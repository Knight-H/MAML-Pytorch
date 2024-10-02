#!/usr/bin/env python
# coding: utf-8
# python maml_pytorch_clean-SEQ.py --train_batch_size 8

from pytorch_transformers import AdamW
from fp16 import FP16_Module, FP16_Optimizer
import os
import torch
import json
from torch.nn import CrossEntropyLoss
from scheduler import AnnealingLR
import time
from datetime import datetime
import pickle
from knight_utils import FILL_VAL, MODEL_BASE_DIR, TASK_DICT , QADataset, DEVICE, args, MODEL_CONFIG, MODEL_CLASS, TOKENIZER, CONFIG_CLASS, SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, TOKENS_WEIGHT, create_dataloader, get_gen_token, CONFIG_NAME
import logging

DICT_RUN_TYPE = {
    "SEQ": "SEQ",
    "SEQ_MAML": "SEQ_MAML",
    "LAMOL": "LAMOL",
    "LAMOL_MAML": "LAMOL_MAML",
    "LAMOL_MAML_REAL": "LAMOL_MAML_REAL",
    "LAMOL_MAML_EXP": "LAMOL_MAML_EXP",
}

# tasks = ['movie', 'boolq', 'scifact']
# tasks = ['movie', 'scifact', 'boolq']
# tasks = ['scifact', 'movie', 'boolq']
# tasks = ['scifact', 'boolq', 'movie']
tasks = ['boolq', 'movie', 'scifact']
TASK_TRIAD = ''.join([task_name[0] for task_name in tasks])
DATETIME = datetime.today().strftime('%Y%m%dT%H%M%S')
RUN_TYPE = DICT_RUN_TYPE["SEQ"]
RUN_ID = DATETIME + "_" + TASK_TRIAD + "_" + RUN_TYPE


model_dir = os.path.join(MODEL_BASE_DIR, RUN_ID)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
logging.basicConfig(filename=f'{model_dir}/{RUN_ID}.log', level=logging.INFO)
logger = logging.getLogger(__name__)


# ## LAMOL Dataset
# 
# `train_qadata` is a pytorch Dataset.  
# A single datapoint of a returns a list of length 8.  
# ```python
# return cq_example, len(cq_example), cqa_example, len(cqa_example), Y_example, gen_X_example, gen_Y_example, idx
#            0                1               2           3                   4       5              6          7
# # 0 cq_example is context+question+__ans__. ie. [7110, 25, 734, 6036, 11886, 467, 284, 257, 4928, 2151]
# # 1 len(cq_example) is the length ie. 901
# # 2 cqa_example is context+question+__ans__+answer ie. [7110, 25, 734, 6036, 11886, 467, 284, 257, 4928, 2151]
# # 3 len(cqa_example) is the length ie. 903
# # 4 Y_example is FILL_VALUE+answer only. ie. [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
# # 5 gen_X_example is __gen__+context+question+__ans__+answer ie. [50260, 7110, 25, 734, 6036, 11886, 467, 284, 257, 4928]
# # 6 gen_Y_example is context+question+__ans__+answer ie. [7110, 25, 734, 6036, 11886, 467, 284, 257, 4928, 2151]
# # 7 idx is id (supposed to be uuid? but i don't see it) ie. 0
# ```


def main():
    # Initialize 2 models
    print("Initializing Model...")
    net_cls = MODEL_CLASS
    net = net_cls.from_pretrained('gpt2').to(DEVICE)
    net.resize_token_embeddings(len(TOKENIZER))
    net = FP16_Module(net)

    # Optimizer
    max_grad_norm=1
    # the optimizer is to update theta parameters, not theta_pi parameters.

    meta_param_optimizer = list(net.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    meta_optimizer_grouped_parameters = [
        {'params': [p for n, p in meta_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in meta_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    meta_optimizer = AdamW(meta_optimizer_grouped_parameters, lr=6.25e-5, eps=1e-4)
    meta_optimizer = FP16_Optimizer(meta_optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                       dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})

    # Sequential Tasks
    for task_id in range(len(tasks)):

        tic_TASK = time.time()


        _tasks = [tasks[task_id]]
        print(f"Starting task {_tasks[0]}")


        ##### Start training on task_id #####
        gen_token = get_gen_token(_tasks[0])
        TOKENIZER.add_tokens([gen_token])
        TOKENIZER.save_pretrained(model_dir)
        SPECIAL_TOKENS[_tasks[0]] = gen_token
        SPECIAL_TOKEN_IDS[_tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
        logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))
        MODEL_CONFIG.vocab_size = len(TOKENIZER)
        MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
        global TOKENS_WEIGHT
        if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
            TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
        # Resize Token Embeddings after special tokens are appended
        net.resize_token_embeddings(len(TOKENIZER))
        # again because resize_token_embeddings makes embedding layer fp32
        net = FP16_Module(net)
        


        ##### Get Extra data and that particular dataset #####
        train_extra_data = []
        train_dataset = [TASK_DICT[t]["train"] for t in _tasks]
        train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[_tasks[0]], train_extra_data)
        max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
        train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)

        # Training loss function
        train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT)

        # Scheduler - n.train_epochs is 1 for SEQ
        n_train_optimization_steps = len(train_qadata) * 1


        ##### Stream from that dataset's dataloader #####
        iter_dataloader = iter(train_dataloader)
        n_steps = 0

        episode_loss = []
        while True:
            tic_BATCH = time.time()

            # 1. Get the support data from the first batch 
            #    and the query data from the second batch
            try:
                _, _, query_x, _, query_y, query_gen_x, query_gen_y = next(iter_dataloader)

                n_inputs = sum(_cqa.shape[0] for _cqa in query_x)

                # Since we only have 1 GPU, just use the first one, it will separate batches according to the device IDS
                query_x = query_x[0]
                query_y = query_y[0]

                query_x = query_x.to(DEVICE)
                query_y = query_y.to(DEVICE)

            except StopIteration:
                break


            ### START Meta-Learning Phase ###
            # 4. After Adaptation, use the query set for learning
            # Somehow it also returns attentions in [1]?, this is selecting 0 of what WrapModel is doing 
            qa_logits = net(query_x)[0]
            qa_loss = train_loss_fct(qa_logits.transpose(1,2), query_y)
            loss = qa_loss

            # Add loss to episode loss
            episode_loss.append(loss.item())


            # use our sumed gradients_pi to update the theta/net network,
            # since our optimizer receive the self.net.parameters() only.
            # Update Meta Optimizer
            meta_optimizer.backward(loss, update_master_grads=False) # instead of loss.backward() for fp16
            meta_optimizer.update_master_grads()
            meta_optimizer.clip_master_grads(max_grad_norm)
            meta_optimizer.step()
            # DO I NEED SCHEDULER HERE???
            meta_optimizer.zero_grad()



            ### END Meta-Learning Phase ###
            n_steps += 1

            toc_BATCH = time.time() - tic_BATCH

            torch.cuda.empty_cache()
    #         mem = float(torch.cuda.memory_allocated() / (1024 * 1024))
    #         print("memory allocated:", mem, "MiB")
            if n_steps%10 == 0:
                logger.info(f'{RUN_ID} {_tasks[0]} Steps: {n_steps}/{len(train_qadata)//args.train_batch_size} Episode {n_steps}: Loss: {loss:.5f}  Batch: {n_inputs}')
                print(f'{RUN_ID} {_tasks[0]} Steps: {n_steps}/{len(train_qadata)//args.train_batch_size} Episode {n_steps}: Loss: {loss:.5f}  Batch: {n_inputs}')
                logger.info(f'[TIME] BATCH {RUN_ID} {_tasks[0]} {toc_BATCH}')

        toc_TASK = time.time() - tic_TASK
        MODEL_SAVE_LOC = os.path.join(model_dir, f'{_tasks[0]}.model')
        LOSS_SAVE_LOC = os.path.join(model_dir, f'{_tasks[0]}_loss.pickle')
        torch.save(net.state_dict(), MODEL_SAVE_LOC)
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