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
from metrics import compute_metrics
import logging
from collections import OrderedDict

DICT_RUN_TYPE = {
    "SEQ": "SEQ",
    "SEQ_MAML": "SEQ_MAML",
    "LAMOL": "LAMOL",
    "LAMOL_MAML": "LAMOL_MAML",
    "LAMOL_MAML_REAL": "LAMOL_MAML_REAL",
    "LAMOL_MAML_EXP": "LAMOL_MAML_EXP",
}

TASK_TRIAD = 'FewRel'
DATETIME = datetime.today().strftime('%Y%m%dT%H%M%S')
RUN_TYPE = DICT_RUN_TYPE["LAMOL"]
RUN_ID = DATETIME + "_" + TASK_TRIAD + "_" + RUN_TYPE


model_dir = os.path.join(MODEL_BASE_DIR, RUN_ID)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# Logging from the way em_in_lll works
logger = logging.getLogger(__name__)
logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(format=logging_format, filename=f'{model_dir}/{RUN_ID}.log', filemode='a', level=logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(logging_format))
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)
for handler in root_logger.handlers:
    handler.addFilter(TimeFilter())

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
    print("Testing >> SPECIAL_TOKEN_IDS", SPECIAL_TOKEN_IDS)
    test_qadata = FewRelQADataset(test_dataset , SPECIAL_TOKEN_IDS[task_id]).sort() # doesn't really matter? gen token isn't used?
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
            
            if batch_i % 100 == 0:
                logger.info(f"[ERROR_ANALYSIS]  {cnt}/{n_examples} Predicted Answer {TOKENIZER.decode(qa_results[cnt].tolist())}")
                logger.info(f"[ERROR_ANALYSIS] {cnt}/{n_examples} Predicted Tokens {qa_results[cnt].tolist()[query_x_len[batch_i]:]}")

            
            # Do the score calculation here
            # The answer of that particular batch to list
            # EDIT
            Y = query_y[batch_i].tolist()
            Y = list(filter(lambda x: x != -1, Y))[:-1]  # remove eos from the answer
            if batch_i % 100 == 0:
                logger.info(f"[ERROR_ANALYSIS]  {cnt}/{n_examples} Actual Tokens {Y}")
            Y = ' '.join([str(y) for y in Y]).split(str(SPECIAL_TOKEN_IDS["pad_token"]))
            Y = [TOKENIZER.decode(list(map(int, y.split()))) for y in Y]
            
            # Change the QA Results to a decoded version of real answer and predicted answer
            qa_results[cnt] = [TOKENIZER.decode(qa_results[cnt].tolist()[query_x_len[batch_i]:]), Y]
            #print(f"Predict vs Actual {cnt}/{n_examples}", qa_results[cnt])
            if batch_i % 100 == 0:
                logger.info(f"[ERROR_ANALYSIS]  {cnt}/{n_examples} Actual Answer {Y}")
                logger.info(f"[ERROR_ANALYSIS]  {cnt}/{n_examples} Predict vs Actual {qa_results[cnt]}")
            cnt += 1
        n_steps += 1
    
    pbar.close()
    
    score = compute_metrics(
            qa_results,
            bleu=False,
            dialogue=False,
            rouge=False,
            logical_form=False,
            corpus_f1=False
    )
    print(score)

    results_path = os.path.join(model_dir,f"qa_o{order}.csv")
    with open(results_path, "w",encoding="utf-8") as f:
        qa_writer = csv.writer(f,delimiter=',')
        qa_writer.writerow(["y","pred"])
        for pred, y in qa_results:
            qa_writer.writerow([y,pred]) 

    with open(os.path.join(model_dir, f"metrics-o{order}.json"),"w") as f:
        json.dump(score, f)

    acc = score["em"] # For text cls, this is accuracy.
    logger.info('Overall test metrics: Accuracy = {:.4f}'.format(acc))
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

        # Initialize Training args and Model, Tokenizer
        pickle.dump(args, open(os.path.join(model_dir, 'train_args'), 'wb'))
        logger.info("args: " + str(args))

        # Initialize models & Reset Tokenizer
        print("Initializing Model & Reset Tokenizer...")
        # ========= Copied from utils.
        global SPECIAL_TOKENS
        global TOKENIZER
        global SPECIAL_TOKEN_IDS
        global TOKENS_WEIGHT
        # Delete all special tokens if have
        # https://stackoverflow.com/questions/11941817/how-can-i-avoid-runtimeerror-dictionary-changed-size-during-iteration-error
        # Casting the dictionary items to list creates a list of its items, so you can iterate over it and avoid the RuntimeError.
        for k, v in list(SPECIAL_TOKENS.items()): 
            if k not in ["ans_token", "pad_token", "unk_token", "eos_token"]:
                del SPECIAL_TOKENS[k]
        model_class, tokenizer_class, config_class = MODEL_CLASSES['gpt2']
        TOKENIZER = tokenizer_class.from_pretrained('gpt2')
        TOKENIZER.add_tokens(list(SPECIAL_TOKENS.values()))
        SPECIAL_TOKEN_IDS = {k:TOKENIZER.convert_tokens_to_ids(v) for k,v in SPECIAL_TOKENS.items()}
        MODEL_CONFIG = config_class.from_pretrained('gpt2')
        MODEL_CONFIG.vocab_size = len(TOKENIZER)
        args.max_len = MODEL_CONFIG.n_positions
        TOKENS_WEIGHT = torch.ones([MODEL_CONFIG.vocab_size], dtype=torch.float).to(DEVICE)
        TOKENS_WEIGHT[SPECIAL_TOKEN_IDS["ans_token"]] = args.tokens_weight  # only answer token has token weight of 5! (default)
        # ============
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
        
        # Generate continual learning training data
        logger.info('Generating continual learning data')
        train_datasets, shuffle_index = prepare_rel_datasets(train_data, relation_names, cluster_labels, args.num_clusters)
        args.shuffle_index = shuffle_index
        logger.info(f"Shuffle Index: {shuffle_index}")
        logger.info(f"Train Dataset Length: {[len(x) for x in train_datasets]}")
        logger.info('Finished generating continual learning data')

        #  =================== Training ==================================== 
        logger.info('----------Training starts here----------')
        tic_TRAIN = time.time()
        # Sequential Tasks
        for train_id, task_dataset in enumerate(train_datasets):
            tic_TASK = time.time()
            task_id = args.shuffle_index.index(train_id)  # Cluster number/ Task ID is the index of the shuffle!
            _tasks = [task_id]                            # The task just use the number itself!
            print(f"Starting task {_tasks[0]}")

            ##### Start training on task_id #####
            gen_token = get_gen_token(str(_tasks[0]))
            TOKENIZER.add_tokens([gen_token])
            TOKENIZER.save_pretrained(model_dir)
            SPECIAL_TOKENS[_tasks[0]] = gen_token
            print("SPECIAL_TOKENS ", SPECIAL_TOKENS)
            SPECIAL_TOKEN_IDS[_tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
            logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[_tasks[0]]))
            MODEL_CONFIG.vocab_size = len(TOKENIZER)
            MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
            # global TOKENS_WEIGHT
            if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
                TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
            # Resize Token Embeddings after special tokens are appended
            net.resize_token_embeddings(len(TOKENIZER))
            # again because resize_token_embeddings makes embedding layer fp32
            net = FP16_Module(net)
            


            ##### Get Extra data and that particular dataset #####
            train_extra_data = []
            ##### LAMOL_SPECIFIC! This section adds the data (task, prev_task, model, train_extra_data, model_dir, tasks)
            if train_id > 0:
                create_extra_data_rel(train_id, train_datasets, train_extra_data, model_dir)
            logger.info('extra training data size: {}'.format(len(train_extra_data)))
            train_qadata = FewRelQADataset(task_dataset, SPECIAL_TOKEN_IDS[_tasks[0]], train_extra_data)
            max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
            train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
            
            # Scheduler - n.train_epochs is 1 for SEQ
            n_train_optimization_steps = len(train_qadata) * 1
            
            # Do we need scheduler here?!?
            scheduler = AnnealingLR(meta_optimizer, start_lr=6.25e-5, warmup_iter=int(0.005*len(train_qadata)),
                num_iters=int(n_train_optimization_steps), decay_style="linear")
            # Training loss function
            train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT)


            ##### Stream from that dataset's dataloader #####
            iter_dataloader = iter(train_dataloader)
            pbar = tqdm(total=len(iter_dataloader) )
            n_steps = 0
            net.train()

            episode_loss = []
            while True:
                tic_BATCH = time.time()

                # 1. Get the support data from the first batch 
                #    and the query data from the second batch
                try:
                    _, _, query_x, _, query_y, query_gen_x, query_gen_y = next(iter_dataloader)
                    pbar.update(1)

                    n_inputs = sum(_cqa.shape[0] for _cqa in query_x)

                    # Since we only have 1 GPU, just use the first one, it will separate batches according to the device IDS
                    query_x = query_x[0]
                    query_y = query_y[0]
                    query_gen_x = query_gen_x[0]
                    query_gen_y = query_gen_y[0]

                    query_x = query_x.to(DEVICE)
                    query_y = query_y.to(DEVICE)
                    query_gen_x = query_gen_x.to(DEVICE)
                    query_gen_y = query_gen_y.to(DEVICE)

                except StopIteration:
                    break


                ### START Meta-Learning Phase ###
                # 4. After Adaptation, use the query set for learning
                # Somehow it also returns attentions in [1]?, this is selecting 0 of what WrapModel is doing 
                
                ##### LAMOL_SPECIFIC! This section adds the data (task, prev_task, model, train_extra_data, model_dir, tasks)
                qa_logits = net(query_x)[0]
                qa_loss = train_loss_fct(qa_logits.transpose(1,2), query_y)
                lm_logits = net(query_gen_x)[0]
                lm_loss = train_loss_fct(lm_logits.transpose(1,2), query_gen_y)
                qa_loss_mean = torch.mean(qa_loss)
                lm_loss_mean = torch.mean(lm_loss)
                loss = qa_loss_mean +  args.lm_lambda * lm_loss_mean

                
                # Loss items for printing
                lm_loss_item = lm_loss_mean.item()
                qa_loss_item = qa_loss_mean.item()
                # Add loss to episode loss
                episode_loss.append(loss.item())


                # use our sumed gradients_pi to update the theta/net network, (This from utils.TrainStep.__call__())
                # since our optimizer receive the self.net.parameters() only.
                # Update Meta Optimizer
                meta_optimizer.backward(loss, update_master_grads=False) # instead of loss.backward() for fp16
                meta_optimizer.update_master_grads()
                meta_optimizer.clip_master_grads(max_grad_norm)
                meta_optimizer.step()
                # DO I NEED SCHEDULER HERE???
                if not meta_optimizer.overflow:
                    for i in range(n_inputs):
                        scheduler.step()
                meta_optimizer.zero_grad()


                ### END Meta-Learning Phase ###
                n_steps += 1

                toc_BATCH = time.time() - tic_BATCH

                #torch.cuda.empty_cache()  # << Uncomment this if there's CUDA out of MEM!
                
        #         mem = float(torch.cuda.memory_allocated() / (1024 * 1024))
        #         print("memory allocated:", mem, "MiB")
    #             if n_steps%10 == 0:
    #                 logger.info(f'{RUN_ID} {_tasks[0]} Steps: {n_steps}/{len(train_qadata)//args.train_batch_size} Episode {n_steps}: Loss: {loss:.5f}  Batch: {n_inputs} qa_loss {qa_loss_item} lm_loss {lm_loss_item}')
    #                 print(f'{RUN_ID} {_tasks[0]} Steps: {n_steps}/{len(train_qadata)//args.train_batch_size} Episode {n_steps}: Loss: {loss:.5f}  Batch: {n_inputs} qa_loss {qa_loss_item} lm_loss {lm_loss_item}')
    #                 logger.info(f'[TIME] BATCH {RUN_ID} {_tasks[0]} {toc_BATCH}')
            pbar.close()
            toc_TASK = time.time() - tic_TASK
            MODEL_SAVE_LOC = os.path.join(model_dir, f'o{order}-{_tasks[0]}.model')
            LOSS_SAVE_LOC = os.path.join(model_dir, f'o{order}-{_tasks[0]}_loss.pickle')
            torch.save(net.state_dict(), MODEL_SAVE_LOC)
            logger.info(f'{RUN_ID} Order {order} {_tasks[0]} Done Saving Model at {MODEL_SAVE_LOC}')
            logger.info(f'[TIME] TASK {RUN_ID} Order {order} {_tasks[0]} {toc_TASK}')
            pickle.dump( episode_loss, open( LOSS_SAVE_LOC, "wb" ), protocol=pickle.HIGHEST_PROTOCOL )

        toc_TRAIN = time.time() - tic_TRAIN
        logger.info(f"[TIME] Training Time within {toc_TRAIN/3600} hours")

        #  =================== Testing ==================================== 
        logger.info('----------Testing starts here----------')
        tic_TEST = time.time()
        acc = test_task(net, val_dataset, order, args.shuffle_index.index(0))
        accuracies.append(acc)
        toc_TEST = time.time() - tic_TEST
        logger.info(f"[TIME] Testing Time within {toc_TEST/3600} hours")

        # Delete the model to free memory
        del net
        gc.collect()
        torch.cuda.empty_cache()

    logger.info('[ACC] Accuracy across runs = {}'.format(accuracies))
    logger.info('[ACC]Average accuracy across runs: {}'.format(np.mean(accuracies)))
    #################################=====================================================================================
    

    
if __name__ == "__main__":
    print(f"Starting Run with RUN_ID {RUN_ID}")
    print(f"[TIME] Start Run {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    logger.info(f"[TIME] Start Run {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"[ARGS] {args}")
    logger.info(f"[ARGS] {args}")
    tic_RUN = time.time()
    
    main()
    
    toc_RUN = time.time() - tic_RUN
    print(f"[TIME] End Run {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
    logger.info(f"[TIME] End Run {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")