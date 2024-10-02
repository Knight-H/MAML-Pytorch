#!/usr/bin/env python
# coding: utf-8
import os
import torch
import json, csv
from torch.nn import CrossEntropyLoss
from scheduler import AnnealingLR
import time, pickle, copy
from datetime import datetime
from knight_utils import FILL_VAL, MODEL_BASE_DIR, TASK_DICT , QADataset, DEVICE, args, MODEL_CONFIG, MODEL_CLASS, TOKENIZER, CONFIG_CLASS, SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, TOKENS_WEIGHT, create_dataloader, get_gen_token, CONFIG_NAME, create_extra_data
from knight_meta import Meta
from fp16 import FP16_Module

import logging
# from maml_pytorch_test import test_all
from collections import OrderedDict
import torch.nn.functional as F
from metrics import compute_metrics


DICT_RUN_TYPE = {
    "SEQ": "SEQ",
    "SEQ_MAML": "SEQ_MAML",
    "LAMOL": "LAMOL",
    "LAMOL_MAML": "LAMOL_MAML",
    "LAMOL_MAML_REAL": "LAMOL_MAML_REAL",
    "LAMOL_MAML_EXP": "LAMOL_MAML_EXP",
}

# tasks = ['movie', 'boolq', 'scifact']
# tasks = ['movie', 'scifact', 'boolq' ]
# tasks = ['scifact', 'movie',  'boolq' ]
# tasks = ['scifact','boolq', 'movie' ]
# tasks = ['boolq','scifact', 'movie' ]
TASK_TRIAD = ''.join([task_name[0:3] for task_name in args.tasks])
DATETIME = datetime.today().strftime('%Y%m%dT%H%M%S')
RUN_TYPE = DICT_RUN_TYPE["LAMOL_MAML"]
if (args.real_sample):
    RUN_TYPE = DICT_RUN_TYPE["LAMOL_MAML_REAL"]
RUN_ID = DATETIME + "_" + TASK_TRIAD + "_" + RUN_TYPE


model_dir = os.path.join(MODEL_BASE_DIR, RUN_ID)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
logging.basicConfig(filename=f'{model_dir}/{RUN_ID}.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
 
    maml = Meta(args, MODEL_CLASS, len(TOKENIZER), DEVICE, is_lamol=True, lm_lambda=args.lm_lambda)
    
    # Sequential Tasks
    for task_id in range(len(args.tasks)):
        tic_TASK = time.time()

        _tasks = [args.tasks[task_id]]
        print(f"Starting task {_tasks[0]}")

        ##### Start training on task_id #####
        gen_token = get_gen_token(_tasks[0])
        TOKENIZER.add_tokens([gen_token])
        TOKENIZER.save_pretrained(model_dir)
        SPECIAL_TOKENS[_tasks[0]] = gen_token
        SPECIAL_TOKEN_IDS[_tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
        logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[_tasks[0]]))
        MODEL_CONFIG.vocab_size = len(TOKENIZER)
        MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
        global TOKENS_WEIGHT
        if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
            TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
        # Resize Token Embeddings after special tokens are appended
        maml.net.resize_token_embeddings(len(TOKENIZER))
        # again because resize_token_embeddings makes embedding layer fp32
        maml.net = FP16_Module(maml.net)

        ##### Get Extra data and that particular dataset #####
        train_extra_data = []
        ##### LAMOL_SPECIFIC! This section adds the data (task, prev_task, model, train_extra_data, model_dir, tasks)
        if task_id > 0:
            prev_task = args.tasks[task_id-1]
            with torch.no_grad():
                create_extra_data(_tasks[0], prev_task, maml.net, train_extra_data, model_dir)
        logger.info('extra training data size: {} at {}'.format(len(train_extra_data), TASK_DICT[_tasks[0]]["train"]))
        train_qadata = QADataset(TASK_DICT[_tasks[0]]["train"], "train", SPECIAL_TOKEN_IDS[_tasks[0]], train_extra_data)
        max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
        train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)

        # Training loss function - here since tokens weight may die! (change at new tokens)
        train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT)
        

        ##### Stream from that dataset's dataloader #####
        iter_dataloader = iter(train_dataloader)
        n_steps = 0

        episode_loss = []
        while True:
            tic_BATCH = time.time()

            # 1. Get the support data from the first batch 
            #    and the query data from the second batch
            try:
                _, _, support_x, _, support_y, support_gen_x, support_gen_y = next(iter_dataloader)
                _, query_x_len, query_x, _, query_y, query_gen_x, query_gen_y = next(iter_dataloader)

                n_inputs = sum(_cqa.shape[0] for _cqa in support_x)

                # Since we only have 1 GPU, just use the first one, it will separate batches according to the device IDS
                support_x = support_x[0]
                support_y = support_y[0]
                query_x = query_x[0]
                query_y = query_y[0]
                query_x_len = query_x_len[0]
                # For LAMOL
                support_gen_x = support_gen_x[0]
                support_gen_y = support_gen_y[0]
                query_gen_x = query_gen_x[0]
                query_gen_y = query_gen_y[0]

                support_x = support_x.to(DEVICE)
                support_y = support_y.to(DEVICE)
                query_x = query_x.to(DEVICE)
                query_y = query_y.to(DEVICE)
                # For LAMOL
                support_gen_x = support_gen_x.to(DEVICE)
                support_gen_y = support_gen_y.to(DEVICE)
                query_gen_x = query_gen_x.to(DEVICE)
                query_gen_y = query_gen_y.to(DEVICE)

            except StopIteration:
                break


            loss_item = maml(support_x, support_y, query_x, query_y, train_loss_fct, support_gen_x, support_gen_y, query_gen_x, query_gen_y)

            # Add loss to episode loss
            episode_loss.append(loss_item)

            ### END Meta-Learning Phase ###
            n_steps += 1

            toc_BATCH = time.time() - tic_BATCH

            torch.cuda.empty_cache()
            if n_steps%10 == 0:
                logger.info(f'{RUN_ID} {_tasks[0]} Steps: {n_steps}/{len(train_qadata)//args.train_batch_size} Episode {n_steps}: Loss: {loss_item:.5f}  Batch: {n_inputs}')
                print(f'{RUN_ID} {_tasks[0]} Steps: {n_steps}/{len(train_qadata)//args.train_batch_size} Episode {n_steps}: Loss: {loss_item:.5f}  Batch: {n_inputs}')
                logger.info(f'[TIME] BATCH {RUN_ID} {_tasks[0]} {toc_BATCH}')


        toc_TASK = time.time() - tic_TASK
        MODEL_SAVE_LOC = os.path.join(model_dir, f'{_tasks[0]}.model')
        LOSS_SAVE_LOC = os.path.join(model_dir, f'{_tasks[0]}_loss.pickle')
        torch.save(maml.net.state_dict(), MODEL_SAVE_LOC)
        logger.info(f'{RUN_ID} {_tasks[0]} Done Saving Model at {MODEL_SAVE_LOC}')
        print(f'{RUN_ID} {_tasks[0]} Done Saving Model at {MODEL_SAVE_LOC}')
        logger.info(f'[TIME] TASK {RUN_ID} {_tasks[0]} {toc_TASK}')
        pickle.dump( episode_loss, open( LOSS_SAVE_LOC, "wb" ), protocol=pickle.HIGHEST_PROTOCOL )
        


if __name__ == "__main__":
    
    print(f"Starting Run with RUN_ID {RUN_ID}")
    print(f"[TIME] Start Run {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    logger.info(f"[TIME] Start Run {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    tic_RUN = time.time()
    
    main()
    
    toc_RUN = time.time() - tic_RUN
    print(f"[TIME] End Run {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
    logger.info(f"[TIME] End Run {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
    