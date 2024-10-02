import time
import os
import torch
import json
import logging
from collections import OrderedDict
import torch.nn.functional as F
from metrics import compute_metrics
# from maml_pytorch_clean import MODEL_CONFIG, create_dataloader, SPECIAL_TOKEN_IDS, TASK_DICT, QADataset, TOKENIZER


logger = logging.getLogger(__name__)

def test_all(tasks, model, MODEL_DIR):
    
    logger.info(f"START TESTING")
    tic_TEST = time.time()
    
    
    # Iterate for all tasks
    for task in tasks:
        tic_TEST_TASK = time.time()
        
        logger.info(f"task: {task}")
        score_dict = {k:None for k in tasks}
        
        with torch.no_grad():
            for task_eval in tasks:
                test_one_to_one(task, task_eval, model, score_dict)
        logger.info("score: {}".format(score_dict))

        with open(os.path.join(MODEL_DIR, f"metrics-{task}.json"),"w") as f:
            json.dump(score_dict, f)

def test_one_to_one(task_load, task_eval, model, score_dict):

    logger.info("start to test { task: %s (load) %s (eval)}" % (task_load, task_eval))

    test_qadata = QADataset(TASK_DICT[task_eval]["test"] , "test", SPECIAL_TOKEN_IDS[task_load]).sort()
    max_a_len = test_qadata.max_a_len
    test_dataloader = create_dataloader(test_qadata, "test")
    n_examples = len(test_qadata)
    logger.info("len of test dataset: {}".format(n_examples))

    need_process = OrderedDict()
    qa_results = [0 for _ in range(n_examples)]
    all_pasts = [[0 for _ in range(n_examples)] for __ in range(MODEL_CONFIG.n_layer)]
    max_tot_lens = [0 for _ in range(n_examples)]

    cnt = 0
    for n_steps, (cqs, len_cqs, _, _, _, _, _) in enumerate(test_dataloader):
        # assume n_gpus == 1
        cqs = cqs[0]
        len_cqs = len_cqs[0]
        n_inputs = cqs.shape[0]
        all_outputs = model(input_ids=cqs.cuda())
        outputs = all_outputs[0]
        pasts = all_outputs[1]
        next_logits = outputs[range(n_inputs), len_cqs-1, :] / temperature_qa
        next_tokens = logits_to_tokens(next_logits).cpu()

        for i in range(n_inputs):
            max_tot_lens[cnt] = max_a_len + test_qadata[cnt][1]
            qa_results[cnt] = cqs[i][:len_cqs[i]]
            if next_tokens[i] != SPECIAL_TOKEN_IDS["eos_token"]:
                qa_results[cnt] = torch.cat((cqs[i][:len_cqs[i]], next_tokens[i]))
                if len(qa_results[cnt]) not in [max_tot_lens[cnt], max_len]:
                    need_process.update([[cnt, None]])
                    for layer_id in range(MODEL_CONFIG.n_layer):
                        all_pasts[layer_id][cnt] = pasts[layer_id][:, i, ..., :len_cqs[i], :].type(torch.half)
            cnt += 1

#         if len(need_process) > int(12 * args.memory_sizes[0] / cqs.shape[1]):  # dynamic threshold to avoid out of memory
#             sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)
    sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)


    for i in range(len(test_qadata)):
        _, len_cq, _, _, Y, _, _, _ = test_qadata[i]

        Y = list(filter(lambda x: x != -1, Y))[:-1]  # remove eos
        Y = ' '.join([str(y) for y in Y]).split(str(SPECIAL_TOKEN_IDS["pad_token"]))
        Y = [TOKENIZER.decode(list(map(int, y.split()))) for y in Y]
        qa_results[i] = [TOKENIZER.decode(qa_results[i].tolist()[len_cq:]), Y]
    get_test_score(task_eval, qa_results, score_dict)

    model_dir = model.model_dir
    results_path = os.path.join(model_dir,f"qa_{task_load}_{task_eval}.csv")
    with open(results_path, "w",encoding="utf-8") as f:
        qa_writer = csv.writer(f,delimiter=',')
        qa_writer.writerow(["y","pred"])
        for pred, y in qa_results:
            qa_writer.writerow([y,pred])

    return model, score_dict

def remove_id(idx, need_process, all_pasts):
    assert idx in need_process
    del need_process[idx]
    for layer_id in range(MODEL_CONFIG.n_layer):
        all_pasts[layer_id][idx] = 0
def sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens):
    while len(need_process) > 0:
        first_id = next(iter(need_process))
        shortest_len = len(qa_results[first_id])
#         decode_batch_size = int(args.memory_sizes[0] * MEMORY_FACTOR[args.seq_train_type] // (shortest_len+1)**LEN_FACTOR)
        decode_batch_size = test_batch_size
        it = iter(need_process)
        stop = False
        remove_ids = []
        while not stop:
            batch_ids, input_ids, past = [], [], [[] for _ in range(MODEL_CONFIG.n_layer)]
            while True:
                try:
                    cur_id = next(it)
                    if len(qa_results[cur_id]) > shortest_len:
                        stop = True
                        break
                    batch_ids.append(cur_id)
                    input_ids.append(qa_results[cur_id][-1:])
                    for layer_id in range(MODEL_CONFIG.n_layer):
                        past[layer_id].append(all_pasts[layer_id][cur_id])
                    if len(input_ids) == decode_batch_size:
                        break
                except StopIteration:
                    stop = True
                    break

            n_inputs = len(input_ids)
            if n_inputs == 0:
                break
            input_ids = torch.stack(input_ids)
            for layer_id in range(MODEL_CONFIG.n_layer):
                past[layer_id] = torch.stack(past[layer_id], dim=1)
            all_outputs = model(input_ids=input_ids.cuda(), past=past)

            outputs = all_outputs[0]
            pasts = all_outputs[1]

            next_logits = outputs[..., -1, :] / temperature_qa
            next_tokens = logits_to_tokens(next_logits).cpu()

            for i, cur_id in enumerate(batch_ids):
                if next_tokens[i] == SPECIAL_TOKEN_IDS["eos_token"]:
                    remove_ids.append(cur_id)
                else:
                    qa_results[cur_id] = torch.cat((qa_results[cur_id], next_tokens[i]))
                    if len(qa_results[cur_id]) in [max_tot_lens[cur_id], max_len]:
                        remove_ids.append(cur_id)
                    else:
                        for layer_id in range(MODEL_CONFIG.n_layer):
                            all_pasts[layer_id][cur_id] = pasts[layer_id][:, i].type(torch.half)
        for idx in remove_ids:
            remove_id(idx, need_process, all_pasts)

def get_gen_token(task):
    return '__' + task + '__'

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    return logits


def logits_to_tokens(next_logits):
    filtered_logits = top_k_top_p_filtering(next_logits, top_k=top_k_qa, top_p=top_p_qa)
    log_probs = F.softmax(filtered_logits, dim=-1)
    next_tokens = torch.multinomial(log_probs, num_samples=1)
    return next_tokens