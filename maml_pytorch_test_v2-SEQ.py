import time
import os
import torch
import json
import logging
import csv
from collections import OrderedDict
import torch.nn.functional as F
from metrics import compute_metrics
from knight_utils import FILL_VAL, MODEL_BASE_DIR, TASK_DICT , QADataset, DEVICE, args, MODEL_CONFIG, MODEL_CLASS, TOKENIZER, CONFIG_CLASS, SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, TOKENS_WEIGHT, create_dataloader, get_gen_token, CONFIG_NAME, remove_id, sample_sequence, top_k_top_p_filtering, logits_to_tokens
from datetime import datetime
from fp16 import FP16_Module, FP16_Optimizer
from tqdm import tqdm

MODEL_DIR_NAME = "20210905T173303_smb_SEQ"
MODEL_DIR_NAME = args.model_dir_name
MODEL_DIR = os.path.join(MODEL_BASE_DIR,MODEL_DIR_NAME)


logging.basicConfig(filename=f'{MODEL_DIR}/test_run.log', level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    for task in args.tasks:
        model_path = os.path.join(MODEL_DIR, f"{task}.model")
        config_path = os.path.join(MODEL_DIR,CONFIG_NAME)

        gen_token = get_gen_token(task)
        TOKENIZER.add_tokens([gen_token])
        SPECIAL_TOKENS[task] = gen_token
        SPECIAL_TOKEN_IDS[task] = TOKENIZER.convert_tokens_to_ids(gen_token)
    #     model_config = CONFIG_CLASS.from_json_file(config_path) # Already defined
        model = MODEL_CLASS(MODEL_CONFIG).cuda()
        # Don't load state dict here, load for every adaptation phase!

    #     print(model)
        print(model_path)

        global TOKENS_WEIGHT
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

        # Can do this!
        with torch.no_grad():
            for task_eval in args.tasks:
                test_one_to_one(task, task_eval, model, score_dict)
        logger.info("score: {}".format(score_dict))

        with open(os.path.join(MODEL_DIR, f"metrics-{task}.json"),"w") as f:
            json.dump(score_dict, f)


def test_one_to_one(task_load, task_eval, model, score_dict):
    tic_TASK = time.time()
    logger.info("start to test { task: %s (load) %s (eval)}" % (task_load, task_eval))
    print("start to test { task: %s (load) %s (eval)}" % (task_load, task_eval))

    
    test_qadata = QADataset(TASK_DICT[task_eval]["test"] , "test", SPECIAL_TOKEN_IDS[task_load]).sort()
    # EDIT Not sure if Leaving this here is correct
    # I see that in the original code, they sort test_qadata and qa_results AFTER sample_sequence.
    # I think that just sort the data first, and then create dataloader should not do anyharm!!
    # And hopefully the test_qadata.answer[cnt] will also perform well.
    if task_eval in ['wikisql','woz.en','multinli.in.out']:
        test_qadata.sort_by_index()
    
    max_a_len = test_qadata.max_a_len
    n_examples = len(test_qadata)
    logger.info("len of test dataset: {}".format(n_examples))
    print("len of test dataset: {}".format(n_examples))
    
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
            
            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Predicted Answer {TOKENIZER.decode(qa_results[cnt].tolist())}")
            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Predicted Tokens {qa_results[cnt].tolist()[query_x_len[batch_i]:]}")

            
            # Do the score calculation here
            # The answer of that particular batch to list
            # EDIT
            if task_eval in ['wikisql','woz.en']:
                Y = test_qadata.answers[cnt]
            else:
                Y = query_y[batch_i].tolist()
                Y = list(filter(lambda x: x != -1, Y))[:-1]  # remove eos from the answer
                logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Actual Tokens {Y}")
                Y = ' '.join([str(y) for y in Y]).split(str(SPECIAL_TOKEN_IDS["pad_token"]))
                Y = [TOKENIZER.decode(list(map(int, y.split()))) for y in Y]
            
            
            # Change the QA Results to a decoded version of real answer and predicted answer
            qa_results[cnt] = [TOKENIZER.decode(qa_results[cnt].tolist()[query_x_len[batch_i]:]), Y]
            #print(f"Predict vs Actual {cnt}/{n_examples}", qa_results[cnt])
            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Actual Answer {Y}")
            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Predict vs Actual {qa_results[cnt]}")
            
            cnt += 1
        n_steps += 1
    
    pbar.close()
    toc_TASK = time.time() - tic_TASK
    logger.info(f'[TIME] TASK {(task_load, task_eval)} {toc_TASK}')
    
    get_test_score(task_eval, qa_results, score_dict)
    print(score_dict)

    model_dir = model.model_dir
    results_path = os.path.join(model_dir,f"qa_{task_load}_{task_eval}.csv")
    with open(results_path, "w",encoding="utf-8") as f:
        qa_writer = csv.writer(f,delimiter=',')
        qa_writer.writerow(["y","pred"])
        for pred, y in qa_results:
            # EDIT 
            if task_eval == 'wikisql': 
                y = y["answer"]
            elif task_eval == 'woz.en': 
                y = y[1]
            qa_writer.writerow([y,pred]) 

    return model, score_dict

def get_test_score(task_eval,qa_results,score_dict):

    score = compute_metrics(
            qa_results,
            bleu='iwslt.en.de' in task_eval or 'multinli.in.out' in task_eval,
            dialogue='woz.en' in task_eval,
            rouge='cnn_dailymail' in task_eval,
            logical_form='wikisql' in task_eval,
            corpus_f1='zre' in task_eval
    )
    score_dict[task_eval] = score
    

if __name__ == "__main__":
    
    print(f"Starting Test for {MODEL_DIR_NAME}")
    print(f"[TIME] Start Test {MODEL_DIR_NAME} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    logger.info(f"[TIME] Start Test {MODEL_DIR_NAME} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    tic_RUN = time.time()
    
    main()
    
    toc_RUN = time.time() - tic_RUN
    print(f"[TIME] End Test {MODEL_DIR_NAME} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
    logger.info(f"[TIME] End Test {MODEL_DIR_NAME} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")