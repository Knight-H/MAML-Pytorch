#!/usr/bin/env python
# coding: utf-8
# python maml_pytorch_clean-SEQ.py --train_batch_size 8

from pytorch_transformers import AdamW
from fp16 import FP16_Module, FP16_Optimizer
import os, gc
import torch
import torchtext
import numpy as np
import json, csv
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from scheduler import AnnealingLR
import time
from datetime import datetime
import pickle
from knight_utils import FILL_VAL, MODEL_BASE_DIR, TASK_DICT , FewRelQADataset, DEVICE, args, MODEL_CONFIG, MODEL_CLASS, MODEL_CLASSES, TOKENIZER, \
        CONFIG_CLASS, SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, TOKENS_WEIGHT, create_dataloader, get_gen_token, CONFIG_NAME, create_extra_data_rel, \
        TimeFilter, logits_to_tokens, sample_sequence, \
        read_relations, read_rel_data, get_relation_embedding, get_relation_index, create_relation_clusters, LifelongFewRelDataset, prepare_rel_datasets
from metrics import compute_metrics, compute_rel_metrics
import logging
from collections import OrderedDict

MODEL_DIR_NAME = "20210905T173303_smb_SEQ"
MODEL_DIR_NAME = args.model_dir_name
MODEL_DIR = os.path.join(MODEL_BASE_DIR,MODEL_DIR_NAME)

logging.basicConfig(filename=f'{MODEL_DIR}/test_run.log', level=logging.INFO)
logger = logging.getLogger(__name__)

LOGGING_STEPS = 10000


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
# 
# For FewRel, This is the format:
# {__7__} {context}  {question} {__ans__} {answer} {eos}
#           text        Q                  Yes/No 
# Q: Is this the relation of {label}?         
# ```
def test_task(model, test_dataset, order, task_id):
    # This cant be sorted! need to check [1,0,0]
    test_qadata = FewRelQADataset(test_dataset , SPECIAL_TOKEN_IDS[task_id]) # doesn't really matter? gen token isn't used?
    max_a_len = test_qadata.max_a_len
    n_examples = len(test_qadata)
    logger.info("len of test dataset: {}".format(n_examples))
    ##### Make dataloaders for that particular dataset #####
    test_dataloader = create_dataloader(test_qadata, "test")
    ##### Stream from that dataset's dataloader #####
    iter_test_dataloader = iter(test_dataloader)
    pbar = tqdm(total=len(iter_test_dataloader) )

    need_process = OrderedDict()
    # qa_results is qa_results[cnt]
    qa_results = [0 for _ in range(n_examples)]
    # All pasts is shape all_pasts[layer_id][cnt]
    all_pasts = [[0 for _ in range(n_examples)] for __ in range(MODEL_CONFIG.n_layer)]
    # max_tot_lens is qa_results[cnt]
    max_tot_lens = [0 for _ in range(n_examples)]
    cnt = 0
    n_steps = 0
    while True:
        try:
            query_x, query_x_len, query_x_cqa, _, query_y, _, _ = next(iter_test_dataloader) # Let query get the CQ!
            pbar.update(1)
        except StopIteration:
            break
        n_inputs = sum(_cqa.shape[0] for _cqa in query_x)
        # Since we only have 1 GPU, just use the first one, it will separate batches according to the device IDS
        query_x = query_x[0]
        query_y = query_y[0]
        query_x_len = query_x_len[0] # an array of query x lengths, but test batch size is only1??
        query_x_cqa = query_x_cqa[0] #EXTRA DEBUG

        query_x = query_x.to(DEVICE)
        query_y = query_y
        query_x_cqa = query_x_cqa.to(DEVICE) #EXTRA DEBUG
        
        model.eval()
        ### START Meta-Learning Phase ###
        # 4. After Adaptation, use the query set for test (CQ ONLY)
        # model() returns Tuple of length 2: 
        #  The [0] is a  torch.Size([1, 225, 50260]), and the [1] is 12 of torch.Size([2, 1, 12, 225, 64])
        # Thinking that the [0] is the actual output and [1] is the pasts?
        all_outputs = model(query_x)
        outputs = all_outputs[0]
        pasts = all_outputs[1]
        next_logits = outputs[range(n_inputs), query_x_len-1, :] / args.temperature_qa
        next_tokens = logits_to_tokens(next_logits).cpu()
        
        # Maybe this is not needed in testing since n_inputs is only 1??
        for batch_i in range(n_inputs):
            # max total length = max answer length + length of cq
            max_tot_lens[cnt] = max_a_len + test_qadata[cnt][1] 
            # add the cq of that particular batch to qa_results (Change it to cpu first!)
            qa_results[cnt] = query_x.cpu()[batch_i][:query_x_len[batch_i]]
            # If the next tokens is not eos
            if next_tokens[batch_i] != SPECIAL_TOKEN_IDS["eos_token"]:
                # Concat the result
                qa_results[cnt] = torch.cat((qa_results[cnt], next_tokens[batch_i]))
                # if the length is not max yet -> MAXTOT 225 1024
                if len(qa_results[cnt]) not in [max_tot_lens[cnt], args.max_len]:
                    # Append need_process of that cnt
                    need_process.update([[cnt, None]])
                    # Update all pasts
                    for layer_id in range(MODEL_CONFIG.n_layer):
                        all_pasts[layer_id][cnt] = pasts[layer_id][:, batch_i, ..., :query_x_len[batch_i], :].type(torch.half)
            
            # Try sample_sequence here! it will get all need_process (should be only 1 batch, and generate all!)
            sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)
            
            if n_steps % LOGGING_STEPS == 0:
                logger.info(f"[ERROR_ANALYSIS]  {cnt}/{n_examples} Predicted Answer {TOKENIZER.decode(qa_results[cnt].tolist())}")
                logger.info(f"[ERROR_ANALYSIS] {cnt}/{n_examples} Predicted Tokens {qa_results[cnt].tolist()[query_x_len[batch_i]:]}")

            
            # Do the score calculation here
            # The answer of that particular batch to list
            # EDIT
            Y = query_y[batch_i].tolist()
            Y = list(filter(lambda x: x != -1, Y))[:-1]  # remove eos from the answer
            if n_steps % LOGGING_STEPS == 0:
                logger.info(f"[ERROR_ANALYSIS]  {cnt}/{n_examples} Actual Tokens {Y}")
            Y = ' '.join([str(y) for y in Y]).split(str(SPECIAL_TOKEN_IDS["pad_token"]))
            Y = [TOKENIZER.decode(list(map(int, y.split()))) for y in Y]
            
            # Change the QA Results to a decoded version of real answer and predicted answer
            qa_results[cnt] = [TOKENIZER.decode(qa_results[cnt].tolist()[query_x_len[batch_i]:]), Y]
            #print(f"Predict vs Actual {cnt}/{n_examples}", qa_results[cnt])
            if n_steps % LOGGING_STEPS == 0:
                logger.info(f"[ERROR_ANALYSIS]  {cnt}/{n_examples} Actual Answer {Y}")
                print(f"[ERROR_ANALYSIS]  {cnt}/{n_examples} Predict vs Actual {qa_results[cnt]}")
                logger.info(f"[ERROR_ANALYSIS]  {cnt}/{n_examples} Predict vs Actual {qa_results[cnt]}")
            cnt += 1
        n_steps += 1
    
    pbar.close()
    score = compute_rel_metrics(qa_results)

    results_path = os.path.join(MODEL_DIR,f"qa_o{order}-{task_id}.csv")
    with open(results_path, "w",encoding="utf-8") as f:
        qa_writer = csv.writer(f,delimiter=',')
        qa_writer.writerow(["y","pred"])
        for pred, y in qa_results:
            qa_writer.writerow([y,pred]) 

    with open(os.path.join(MODEL_DIR, f"metrics-o{order}-{task_id}.json"),"w") as f:
        json.dump(score, f)

    acc = score
    logger.info('[ACC] Overall test metrics: Accuracy = {:.4f}'.format(acc))
    return acc

def main():
    # Load training and validation data
    logger.info('Loading the dataset')
    data_dir = '/data/omler_data/LifelongFewRel'
    relation_file = os.path.join(data_dir, 'relation_name.txt')
    training_file = os.path.join(data_dir, 'training_data.txt')
    validation_file = os.path.join(data_dir, 'val_data.txt')
    # ie. ['fill', ['place', 'served', 'by', 'transport', 'hub'], ['mountain', 'range'], ['religion'], ['participating', 'team'], ...]
    # Note that 'fill' is the 0 index, can be ignored
    relation_names = read_relations(relation_file) # List of relation names (converted to 1-based index later)
    train_data = read_rel_data(training_file)
    val_data = read_rel_data(validation_file)
    logger.info('Finished loading the dataset')

    # Load GloVe vectors
    logger.info('Loading GloVe vectors')
    glove = torchtext.vocab.GloVe(name='6B', dim=300)
    logger.info('Finished loading GloVe vectors')

    # Get relation embeddings for clustering
    relation_embeddings = get_relation_embedding(relation_names, glove)
    print(relation_embeddings.shape)

    # Generate clusters
    # This essentially goes through all train_data and get label set, which is a list of 1-80 ie. [80, 25, 75, 15, 62, 74, 5, 10...] 
    relation_index = get_relation_index(train_data)  
    # This uses KMeans to divide the label up into 10 disjoint clusters ie. {80: 1, 25: 5, 75: 3, 15: 1, 62: 1, 74: 1, 5: 1, 10: 2...}
    # > relation_embeddings just return a dictionary of relation_index --> Glove embedding ie. { 80: embedding, 25: embedding, ...}
    cluster_labels, relation_embeddings = create_relation_clusters(args.num_clusters, relation_embeddings, relation_index)

    # Validation dataset (Test Dataset)
    val_dataset = LifelongFewRelDataset(val_data, relation_names)
    print(f"Val Dataset Length: {len(val_dataset)}")

    # Run for different orders of the clusters
    accuracies = []
    for order in range(args.order):
        logger.info('Running order {}'.format(order + 1))

        # Generate continual learning training data
        logger.info('Generating continual learning data')
        train_datasets, shuffle_index = prepare_rel_datasets(train_data, relation_names, cluster_labels, args.num_clusters)
        args.shuffle_index = shuffle_index
        print(f"Shuffle Index: {shuffle_index}")
        logger.info(f"Shuffle Index: {shuffle_index}")
        logger.info(f"Train Dataset Length: {[len(x) for x in train_datasets]}")
        logger.info('Finished generating continual learning data')


        # The latest model in the order
        task = args.shuffle_index.index(9) # This is the index of the task!
        model_path = os.path.join(MODEL_DIR, f'o{order}-{task}.model')

        global SPECIAL_TOKENS
        global TOKENIZER
        global SPECIAL_TOKEN_IDS
        global TOKENS_WEIGHT

        # Add all the task tokens 
        for ii in range(10):
            _task_id = args.shuffle_index.index(ii)
            gen_token = get_gen_token(str(_task_id))
            TOKENIZER.add_tokens([gen_token])
            SPECIAL_TOKENS[task] = gen_token
            SPECIAL_TOKEN_IDS[task] = TOKENIZER.convert_tokens_to_ids(gen_token)
        
        print("THIS IS LEN TOK ", len(TOKENIZER))
        model = MODEL_CLASS(MODEL_CONFIG).cuda()

        if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
            TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))

        model.resize_token_embeddings(len(TOKENIZER))
        model = FP16_Module(model)
        model.model_dir = MODEL_DIR
        model.model_path = model_path

        # Try Loading the state dict like this!
        model.state_dict = torch.load(model.model_path, map_location='cuda:0')
        # Load State Dict here! IT's sequential
        model.load_state_dict(model.state_dict)
        
        logger.info(f"task: {task}")
        score_dict = {k:None for k in args.tasks}


        
        #  =================== Testing ==================================== 
        logger.info('----------Testing starts here----------')
        tic_TEST = time.time()
        acc = test_task(model, val_dataset, order, args.shuffle_index.index(9))
        accuracies.append(acc)
        toc_TEST = time.time() - tic_TEST
        logger.info(f"[TIME] Testing Time within {toc_TEST/3600} hours")

        # Delete the model to free memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    logger.info('[ACC] Accuracy across runs = {}'.format(accuracies))
    logger.info('[ACC] Average accuracy across runs: {}'.format(np.mean(accuracies)))
    #################################=====================================================================================
    

    
if __name__ == "__main__":
    print(f"Starting Test for {MODEL_DIR_NAME}")
    print(f"[TIME] Start Test {MODEL_DIR_NAME} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    logger.info(f"[TIME] Start Test {MODEL_DIR_NAME} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    tic_RUN = time.time()
    
    main()
    
    toc_RUN = time.time() - tic_RUN
    print(f"[TIME] End Test {MODEL_DIR_NAME} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
    logger.info(f"[TIME] End Test {MODEL_DIR_NAME} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")