#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, CONFIG_NAME 
from pytorch_transformers import AdamW
from fp16 import FP16_Module, FP16_Optimizer
from torch import nn
import os
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import json, csv
from multiprocessing import Pool
from parallel import DataParallelModel, DataParallelCriterion
from torch.nn import CrossEntropyLoss
from scheduler import AnnealingLR
import warnings
import time
from datetime import datetime
import pickle
import copy

# import GPUtil

import logging
# from maml_pytorch_test import test_all
from collections import OrderedDict
import torch.nn.functional as F
from metrics import compute_metrics


# In[2]:

DICT_RUN_TYPE = {
    "SEQ": "SEQ",
    "SEQ_MAML": "SEQ_MAML",
    "LAMOL": "LAMOL",
    "LAMOL_MAML": "LAMOL_MAML",
    "LAMOL_MAML_REAL": "LAMOL_MAML_REAL",
    "LAMOL_MAML_EXP": "LAMOL_MAML_EXP",
}

tasks = ['movie', 'boolq', 'scifact']
# tasks = ['movie', 'scifact', 'boolq' ]
# tasks = ['scifact', 'movie',  'boolq' ]
# tasks = ['scifact','boolq', 'movie' ]
TASK_TRIAD = ''.join([task_name[0] for task_name in tasks])
DATETIME = datetime.today().strftime('%Y%m%dT%H%M%S')
RUN_TYPE = DICT_RUN_TYPE["SEQ_MAML"]
RUN_ID = DATETIME + "_" + TASK_TRIAD + "_" + RUN_TYPE

TEST_TOO = True



# In[2]:


FILL_VAL = -1

n_gpus = 1
device_ids = [1]

train_batch_size = 3
test_batch_size = 1

data_dir = "/root/LAMOL/lamol_data"
MODEL_BASE_DIR = "/data/model_runs"
n_train_epochs  = 5 

min_n_steps = 1500
min_batch_size = 4
n_train_epochs = 3

# Adaptation Phase
num_updates = 5

DEVICE = 'cuda:0'
# DEVICE = "/root/MAML-Pytorch/fp16.py" 'cpu' TypeError: Wrapped parameters must be either torch.cuda.FloatTensor or torch.cuda.HalfTensor. Received torch.HalfTensor

model_dir = os.path.join(MODEL_BASE_DIR, RUN_ID)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
logging.basicConfig(filename=f'{model_dir}/{RUN_ID}.log', level=logging.INFO)
logger = logging.getLogger(__name__)


# In[3]:


MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config),
}


# In[4]:


TASK_DICT = {
    "movie": {
               "train":os.path.join(data_dir,"movie_train.json"),
               "eval":os.path.join(data_dir,"movie_dev.json"),
               "test":os.path.join(data_dir,"movie_test.json"),
               "n_train_epochs": n_train_epochs 
    },
    "boolq": {
               "train":os.path.join(data_dir,"boolq_train.json"),
               "eval":os.path.join(data_dir,"boolq_dev.json"),
               "test":os.path.join(data_dir,"boolq_test.json"),
               "n_train_epochs": n_train_epochs 
    },
    "scifact": {
               "train":os.path.join(data_dir,"scifact_train.json"),
               "eval":os.path.join(data_dir,"scifact_dev.json"),
               "test":os.path.join(data_dir,"scifact_test.json"),
               "n_train_epochs": n_train_epochs 
    }
}


# In[5]:


# In settings.py
special_tokens = {"ans_token":'__ans__', "pad_token":'__pad__', "unk_token":'__unk__', "eos_token": '<|endoftext|>'}

model_class, tokenizer_class, config_class = MODEL_CLASSES['gpt2']
tokenizer = tokenizer_class.from_pretrained('gpt2')
tokenizer.add_tokens(list(special_tokens.values()))
special_token_ids = {k:tokenizer.convert_tokens_to_ids(v) for k,v in special_tokens.items()}


model_config = config_class.from_pretrained('gpt2')
model_config.vocab_size = len(tokenizer)
max_len = model_config.n_positions

tokens_weight = torch.ones([model_config.vocab_size], dtype=torch.float).to(DEVICE)
tokens_weight[special_token_ids["ans_token"]] = 5


MODEL_CLASS = model_class
TOKENIZER = tokenizer
SPECIAL_TOKENS = special_tokens
SPECIAL_TOKEN_IDS = special_token_ids
TOKENS_WEIGHT = tokens_weight
MODEL_CONFIG = model_config

# # Dataset

class QADataset(Dataset):
    def __init__(self, data_paths, data_type, gen_token, extra_data=[]):
        self.data_type = data_type
        self.gen_token = gen_token
        self.ans_token = SPECIAL_TOKEN_IDS["ans_token"]
        self.eos_token = SPECIAL_TOKEN_IDS["eos_token"]
        self.pad_token = SPECIAL_TOKEN_IDS["pad_token"]

        if not isinstance(data_paths, list):
            data_paths = [data_paths]

        data = []
        for data_path in data_paths:
            if not data_path:
                continue
            with open(data_path, "r") as f:
                raw_ds = json.load(f)
            raw_ds = map(lambda x: x["paragraphs"], raw_ds["data"])
            d = []
            for raw_d in raw_ds:
                d.extend(raw_d)
            data += d
        
        self.data = []
        self.max_a_len = 0
        if len(data) > 0:
            self.data_tokenization(data)

        if len(extra_data) > 0:
            extra_data = map(lambda x: self.etl_single_extra_data(x), extra_data)
            extra_data = list(filter(lambda x:x, extra_data))
            if args.gen_lm_sample_percentage > 0. and len(extra_data) == 0:
                logger.warning("No good extra data but sample percentage > 0!")
            self.data += extra_data


    def etl_single_extra_data(self, data):
        gen_token = data[0]
        data = ' '.join([str(datum) for datum in data[1:]])
        try:
            context = ""
            qa = data
            question, answer = re.split(str(SPECIAL_TOKEN_IDS["ans_token"]), qa)
            context = [int(c) for c in context.strip().split()]
            question = [int(q) for q in question.strip().split()]
            answer = [int(a) for a in re.sub(str(SPECIAL_TOKEN_IDS["eos_token"]), "", answer).strip().split()]
            uid = uuid.uuid1().hex
            data = self.parse_example(gen_token, context, question, answer, uid)
        except ValueError:
            return
        return data

    def concat_example(self, gen_token, c, sep_token, q, ans_token, a, eos_token):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > max_len:
            logger.warning('an example with len {} is too long!'.format(len(example) + 1))
            return
        example = gen_token + c[:max_len-len(example)-1] + example + eos_token
        return example

    def parse_example(self, gen_token, context, question, answer, idx):
        cq_example = self.concat_example([], context, [], question, [self.ans_token], [], [])
        cqa_example = self.concat_example([], context, [], question, [self.ans_token], answer, [])
        Y_example = self.concat_example([], [], [], [], [], answer, [self.eos_token])
        Y_example = [FILL_VAL] * (len(cqa_example) - len(Y_example)) + Y_example
        gen_X_example = self.concat_example([gen_token], context, [], question, [self.ans_token], answer, [])
        gen_Y_example = self.concat_example([], context, [], question, [self.ans_token], answer, [self.eos_token])
        return cq_example, len(cq_example), cqa_example, len(cqa_example), Y_example, gen_X_example, gen_Y_example, idx

    def parallel_tokenization(self, d):
        # ADD MAX LENGTH FOR MODEL SO IT DOESNT SHOW WARNING
        # OLD VERSION OF PYTORCH HUGGINGFACE DOESNT HAVE MAX LENGTH!!!!!
        # Suppress the warnings instead! https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
        # Still doesn't work. idk what to do. we can delete all the warnings catches here
        examples = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            context = TOKENIZER.encode(d["context"])
        max_a_len = 0
        for qa in d["qas"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                question = TOKENIZER.encode(qa["question"])

            raw_answers = qa["answers"]
            if len(raw_answers) == 0:
                assert qa["is_impossible"]
                raw_answers.append({"text": ""})

            answer = []
            for i, raw_answer in enumerate(raw_answers):
                answer.extend(TOKENIZER.encode(raw_answer["text"]))
                if i != len(raw_answers) - 1:
                    answer.append(self.pad_token)
            max_a_len = max(max_a_len, len(answer))

            examples.append(self.parse_example(self.gen_token, context, question, answer, qa.get("id", 0)))
        return examples, max_a_len

    def data_tokenization(self, data):
        with Pool(4) as pool:
            data = pool.map(self.parallel_tokenization, data)
        for datum, max_a_len in data:
            self.data.extend(datum)
            self.max_a_len = max(self.max_a_len, max_a_len)

    def sort(self):
        self.data.sort(key=lambda x: len(x[0]))
        return self

    def sort_by_index(self):
        self.data.sort(key=lambda x: x[-1])

    def get_indices(self):
        return [d[-1] for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# In[9]:


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, data_type, max_batch_size):
        self.dataset = dataset
        self.data_type = data_type
        if data_type == "train":
            self.batch_size = train_batch_size
        else:
            self.batch_size = test_batch_size
        self.n_samples = len(dataset)
        self.max_batch_size = max_batch_size

    def __iter__(self):
        if self.data_type == "test":
            indices = range(self.n_samples)
        else:
            indices = np.random.permutation(self.n_samples)
        max_len, cnt, st = 0, 0, 0
        batch = []
        for ed, idx in enumerate(indices):
            ln = len(self.dataset[idx][2])
            if max(max_len, ln)**LEN_FACTOR * (ed - st + 1) > self.batch_size[cnt]:
                st = ed
                cnt += 1
                max_len = 0
                if cnt == args.n_gpus:
                    yield batch
                    cnt = 0
                    batch = []
            max_len = max(max_len, ln)
            batch.append(idx)
            if len(batch) == self.max_batch_size and self.data_type == "train":
                yield batch
                cnt, max_len, st = 0, 0, ed
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        raise NotImplementedError


# In[10]:


def dynamic_collate_fn(data, batch_size):

    def local_collate():
        null_counter = 0
        _cqs, _len_cqs, _cqas, _len_cqas, _Ys, _gen_Xs, _gen_Ys = [], [], [], [], [], [], []
        Y_max_len = max(len(data[j][4]) for j in range(st, ed))
        cq_max_len = max(len(data[j][0]) for j in range(st, ed))
        for j in range(st, ed):
            if None in data[j] or [] in data[j]:
                null_counter+=1
                logger.warning('null example in collate_fn, count: {}'.format(null_counter))
                continue

            pad_len = cqa_max_len - len(data[j][2])

            _cqs.append(pad_to_max_len(data[j][0], cq_max_len-len(data[j][0]), SPECIAL_TOKEN_IDS["pad_token"]))
            _len_cqs.append(data[j][1])
            _cqas.append(pad_to_max_len(data[j][2], pad_len, SPECIAL_TOKEN_IDS["pad_token"]))
            _len_cqas.append(data[j][3])
            _Ys.append(pad_to_max_len(data[j][4], Y_max_len - len(data[j][4]), FILL_VAL))
            _gen_Xs.append(pad_to_max_len(data[j][5], pad_len, SPECIAL_TOKEN_IDS["pad_token"]))
            _gen_Ys.append(pad_to_max_len(data[j][6], pad_len, FILL_VAL))

        cqs.append(torch.tensor(_cqs))
        len_cqs.append(torch.tensor(_len_cqs))
        cqas.append(torch.tensor(_cqas))
        len_cqas.append(torch.tensor(_len_cqas))
        Ys.append(torch.tensor(_Ys))
        gen_Xs.append(torch.tensor(_gen_Xs))
        gen_Ys.append(torch.tensor(_gen_Ys))

    cqs, len_cqs, cqas, len_cqas, Ys, gen_Xs, gen_Ys = [], [], [], [], [], [], []
    cqa_max_len, cnt, st = 0, 0, 0
    for ed, datum in enumerate(data):
        ln = len(datum[2]) # use cqas to calibrate
        if max(cqa_max_len, ln)**LEN_FACTOR * (ed - st + 1) > batch_size[cnt]:
            local_collate()
            cnt += 1
            cqa_max_len = 0
            st = ed
        cqa_max_len = max(cqa_max_len, ln)
    ed += 1  # otherwise ed will be len(data)-1
    local_collate()

    return cqs, len_cqs, cqas, len_cqas, Ys, gen_Xs, gen_Ys


# In[11]:



def varlen_collate_fn(data):
    batch_size = (len(data) + n_gpus - 1) // n_gpus
    cqs = torch.tensor(pad_all_to_max_len([datum[0] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    len_cqs = torch.tensor([datum[1] for datum in data]).split(batch_size)
    cqas = torch.tensor(pad_all_to_max_len([datum[2] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    len_cqas = torch.tensor([datum[3] for datum in data]).split(batch_size)
    Ys = torch.tensor(pad_all_to_max_len([datum[4] for datum in data], FILL_VAL)).split(batch_size)
    gen_Xs = torch.tensor(pad_all_to_max_len([datum[5] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    gen_Ys = torch.tensor(pad_all_to_max_len([datum[6] for datum in data], FILL_VAL)).split(batch_size)
    return list(cqs), list(len_cqs), list(cqas), list(len_cqas), list(Ys), list(gen_Xs), list(gen_Ys)


# In[12]:


def pad_to_max_len(l, pad_len, val):
    return l + [val] * pad_len
def pad_all_to_max_len(ls, val):
    max_len = max(len(l) for l in ls)
    return [pad_to_max_len(l, max_len-len(l), val) for l in ls]


# In[13]:


def create_dataloader(dataset, data_type, max_batch_size=1000000000):
    if data_type == "train":
        batch_size = train_batch_size
    else:
        batch_size = test_batch_size

    if isinstance(batch_size, list):
        collate_fn=lambda x,bs=batch_size: dynamic_collate_fn(x, bs)
        shuffle = False
        batch_size = 1
        batch_sampler = DynamicBatchSampler(dataset, data_type, max_batch_size)
    else:
        collate_fn=lambda x: varlen_collate_fn(x)
#         shuffle = not (data_type != "train" or args.debug)
        shuffle = False
        batch_sampler = None

    dataloader =  DataLoader(dataset, num_workers=4,
                             collate_fn=collate_fn,
                             shuffle=shuffle,
                             batch_size=batch_size,
                             batch_sampler=batch_sampler)
    return dataloader


# In[14]:


def get_gen_token(task):
    return '__' + task + '__'


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

    net_pi = net_cls.from_pretrained('gpt2').to(DEVICE)
    net_pi.resize_token_embeddings(len(TOKENIZER))
    net_pi = FP16_Module(net_pi)
    net_pi.load_state_dict(net.state_dict())
    
    # The size is 161 named params , were the size of the first one is [50260, 768] (token encoder), so when comparing with step0
    # just use [1:-1], ignore token encoder  transformer.wte.weight torch.Size([50260, 768]) AND  lm_head.weight torch.Size([50260, 768])
    net_step0 = list(net.state_dict().copy().values())
    
    total_diff = sum((x - y).abs().sum() for x, y in zip(net.state_dict().values(), net_pi.state_dict().values()))
    print(f"THIS IS DIFF {total_diff}")
    print(net_step0[2].shape)
#     print(net.state_dict().keys())
#     for key,value in net.state_dict().items():
#         print(key, value.shape)
#     raise Exception("BREAK")

    # Optimizer
    max_grad_norm=1
    param_optimizer = list(net_pi.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=6.25e-5, eps=1e-4)
    optimizer = FP16_Optimizer(optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                       dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})


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
        net_pi.resize_token_embeddings(len(TOKENIZER))
        # again because resize_token_embeddings makes embedding layer fp32
        net = FP16_Module(net)
        net_pi = FP16_Module(net_pi)

        ##### Get Extra data and that particular dataset #####
        train_extra_data = []
        train_qadata = QADataset(TASK_DICT[_tasks[0]]["train"], "train", SPECIAL_TOKEN_IDS[_tasks[0]], train_extra_data)
        max_train_batch_size = max(len(train_qadata) // min_n_steps, min_batch_size)
        train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)

        # Training loss function - here since tokens weight may die! (change at new tokens)
        train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT)

        # Scheduler
        n_train_optimization_steps = len(train_qadata) * n_train_epochs
        scheduler = AnnealingLR(optimizer, start_lr=6.25e-5, warmup_iter=int(0.005*len(train_qadata)),
                num_iters=int(n_train_optimization_steps), decay_style="linear")

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

                support_x = support_x.to(DEVICE)
                support_y = support_y.to(DEVICE)
                query_x = query_x.to(DEVICE)
                query_y = query_y.to(DEVICE)

            except StopIteration:
                break


            ### START Adaptation Phase ###
            # 2. Reinitialize net_pi with parameters from net
#             for m_from, m_to in zip(net.modules(), net_pi.modules()):
#                 m_to.load_state_dict(m_from.state_dict().copy())
            # Try new load state dict
    
            total_diff = sum((x - y).abs().sum() for x, y in zip(net.state_dict().values(), net_pi.state_dict().values()))
            print(f"THIS IS net - net_pi DIFF BEFORE ADAPT {total_diff}")
            # This ignores Token Encoder
            total_diff = sum((x - y).abs().sum() for x, y in zip(list(net.state_dict().values())[1:-1], net_step0[1:-1]))
            print(f"THIS IS net - net (step 0) DIFF BEFORE ADAPT {total_diff}")
            
            net_pi.load_state_dict(net.state_dict())
            
            total_diff = sum((x - y).abs().sum() for x, y in zip(net.state_dict().values(), net_pi.state_dict().values()))
            print(f"THIS IS net - net_pi DIFF AFTER ADAPT {total_diff}")
            # This ignores Token Encoder
            total_diff = sum((x - y).abs().sum() for x, y in zip(list(net.state_dict().values())[1:-1], net_step0[1:-1]))
            print(f"THIS IS net - net (step 0) DIFF AFTER ADAPT {total_diff}")


            # 3. Update the weights with the support set
            # May update for several steps
            for i in range(num_updates):

                qa_logits = net_pi(support_x)
                # Somehow it also returns attentions in [1]?, this is selecting 0 of what WrapModel is doing 
                qa_logits = qa_logits[0]
                qa_loss = train_loss_fct(qa_logits.transpose(1,2), support_y)
                loss = qa_loss

                # Update Optimizer
                optimizer.backward(loss, update_master_grads=False) # instead of loss.backward() for fp16
                optimizer.update_master_grads()
                optimizer.clip_master_grads(max_grad_norm)
                optimizer.step()
                if not optimizer.overflow:
                    for i in range(n_inputs):
                        scheduler.step()
                optimizer.zero_grad()

            ### END Adaptation Phase ###

            ### START Meta-Learning Phase ###
            # 4. After Adaptation, use the query set for learning
            # Somehow it also returns attentions in [1]?, this is selecting 0 of what WrapModel is doing 
            qa_logits = net_pi(query_x)[0]
            qa_loss = train_loss_fct(qa_logits.transpose(1,2), query_y)
            loss = qa_loss
            
            answer_indices = query_y[query_y != FILL_VAL]
            
#             logger.info(f'[FIND LOSS] this is qa_logits shape ${qa_logits.transpose(1,2).shape} and this is query _y ${query_y.shape} ${answer_indices}')
#             print(f'[FIND LOSS] this is qa_logits shape ${qa_logits.transpose(1,2).shape} and this is query _y  ${query_y.shape}  ${answer_indices}')
            
            for batch_i in range(n_inputs):
                X = qa_logits.cpu()[batch_i,query_y[batch_i] != FILL_VAL, :] # This will get  EOS too
                X = torch.argmax(X , dim=1).tolist() # Get the max
                X = list(filter(lambda x: x != -1, X))[:-1]  # remove eos from the answer
                print(f"[ERROR_ANALYSIS] MetaLearning query_x Predicted Token {X}")
                X = ' '.join([str(y) for y in X]).split(str(SPECIAL_TOKEN_IDS["pad_token"]))
                X = [TOKENIZER.decode(list(map(int, y.split()))) for y in X]
                print(f"[ERROR_ANALYSIS] MetaLearning query_x  Predicted Answer {X}")
                logger.info(f"[ERROR_ANALYSIS] MetaLearning query_x  Predicted Answer {X}")
                
                
                Y = query_y.cpu()[batch_i].tolist()
                Y = list(filter(lambda x: x != -1, Y))[:-1]  # remove eos from the answer
                print(f"[ERROR_ANALYSIS] MetaLearning query_x Actual Token {Y}")
                Y = ' '.join([str(y) for y in Y]).split(str(SPECIAL_TOKEN_IDS["pad_token"]))
                Y = [TOKENIZER.decode(list(map(int, y.split()))) for y in Y]
                print(f"[ERROR_ANALYSIS] MetaLearning query_x Actual Answer {Y}")
                logger.info(f"[ERROR_ANALYSIS] MetaLearning query_x  Predicted Answer {X}")
                
                # SOMEHOW THIS FAILS X = qa_logits.cpu()[batch_i,query_y[batch_i] != FILL_VAL, :] 
                # IndexError: index 1 is out of bounds for dimension 0 with size 1
                # So I'll just get one of it, and break out
                # I think this errors because of the last batch not having complete batch.
                break
                
            # Add loss to episode loss
            episode_loss.append(loss.item())

            # gradient for validation on theta_pi
            # after call autorad.grad, you can not call backward again except for setting create_graph = True
            # as we will use the loss as dummpy loss to conduct a dummy backprop to write our gradients to theta network,
            # here we set create_graph to true to support second time backward.
            grads_pi = torch.autograd.grad(loss, net_pi.parameters(), create_graph=True)

            # As we already have the grads to update
            # We use a dummy forward / backward pass to get the correct grads into self.net
            # the right grads will be updated by hook, ignoring backward.
            # use hook mechnism to write sumed gradient into network.
            # we need to update the theta/net network, we need a op from net network, so we call self.learner.net_forward
            # to get the op from net network, since the loss from self.learner.forward will return loss from net_pi network.


            # Somehow it also returns attentions in [1]?, this is selecting 0 of what WrapModel is doing 
            qa_logits = net(query_x)[0]
            dummy_loss = train_loss_fct(qa_logits.transpose(1,2), query_y)
            
            
#             if n_steps > 100:
#                 MODEL_SAVE_LOC = os.path.join(model_dir, f'{_tasks[0]}_steps{n_steps}.model')
#                 torch.save(net.state_dict(), MODEL_SAVE_LOC)
#                 raise Exception("BREAKPOINT")

            # Register a hook on each parameter in the net that replaces the current dummy grad
            # with our grads accumulated across the meta-batch
            hooks = []
            for i, v in enumerate(net.parameters()):
                def closure():
                    ii = i
                    return lambda grad: grads_pi[ii]
                # if you write: hooks.append( v.register_hook(lambda grad : sum_grads_pi[i]) )
                # it will pop an ERROR, i don't know why?
                hooks.append(v.register_hook(closure()))

            # use our sumed gradients_pi to update the theta/net network,
            # since our optimizer receive the self.net.parameters() only.
            # Update Meta Optimizer
            meta_optimizer.backward(dummy_loss, update_master_grads=False) # instead of loss.backward() for fp16
            meta_optimizer.update_master_grads()
            meta_optimizer.clip_master_grads(max_grad_norm)
            meta_optimizer.step()
            # DO I NEED SCHEDULER HERE???
            meta_optimizer.zero_grad()

            # if you do NOT remove the hook, the GPU memory will expode!!!
            for h in hooks:
                h.remove()
            
                

            ### END Meta-Learning Phase ###
            n_steps += 1

            toc_BATCH = time.time() - tic_BATCH

            torch.cuda.empty_cache()
    #         mem = float(torch.cuda.memory_allocated() / (1024 * 1024))
    #         print("memory allocated:", mem, "MiB")
            if n_steps%10 == 0:
                logger.info(f'{RUN_ID} {_tasks[0]} Steps: {n_steps}/{len(train_qadata)//max_train_batch_size} Episode {n_steps}: Loss: {loss:.5f} lr {scheduler.get_lr():.1E} Batch: {n_inputs}')
                print(f'{RUN_ID} {_tasks[0]} Steps: {n_steps}/{len(train_qadata)//max_train_batch_size} Episode {n_steps}: Loss: {loss:.5f} lr {scheduler.get_lr():.1E} Batch: {n_inputs}')
                logger.info(f'[TIME] BATCH {RUN_ID} {_tasks[0]} {toc_BATCH}')
            # FOR TEST WITH ONLY ONE BATCH
#             break

        toc_TASK = time.time() - tic_TASK
        MODEL_SAVE_LOC = os.path.join(model_dir, f'{_tasks[0]}.model')
        LOSS_SAVE_LOC = os.path.join(model_dir, f'{_tasks[0]}_loss.pickle')
        torch.save(net.state_dict(), MODEL_SAVE_LOC)
        logger.info(f'{RUN_ID} {_tasks[0]} Done Saving Model at {MODEL_SAVE_LOC}')
        print(f'{RUN_ID} {_tasks[0]} Done Saving Model at {MODEL_SAVE_LOC}')
        logger.info(f'[TIME] TASK {RUN_ID} {_tasks[0]} {toc_TASK}')
        pickle.dump( episode_loss, open( LOSS_SAVE_LOC, "wb" ), protocol=pickle.HIGHEST_PROTOCOL )
        
        if TEST_TOO:
            test_all(tasks, net, model_dir, SPECIAL_TOKEN_IDS)

            
# TESTT PASTING
temperature_qa = 1.0

top_k_qa = 20
top_p_qa = 0.

def test_all(tasks, model, MODEL_DIR, SPECIAL_TOKEN_IDS):
    
    logger.info(f"START TESTING")
    tic_TEST = time.time()
    
    
    # Iterate for all tasks
    for task in tasks:
        tic_TEST_TASK = time.time()
        
        model.model_dir = MODEL_DIR
        logger.info(f"task: {task}")
        score_dict = {k:None for k in tasks}
        
        for task_eval in tasks:
            test_one_to_one(task, task_eval, model, score_dict, SPECIAL_TOKEN_IDS)
        logger.info("score: {}".format(score_dict))
        print("score: {}".format(score_dict))

        with open(os.path.join(MODEL_DIR, f"metrics-{task}.json"),"w") as f:
            json.dump(score_dict, f)

def test_one_to_one(task_load, task_eval, model, score_dict, SPECIAL_TOKEN_IDS):

    logger.info("start to test { task: %s (load) %s (eval)}" % (task_load, task_eval))
    print("start to test { task: %s (load) %s (eval)}" % (task_load, task_eval))

    if task_load in SPECIAL_TOKEN_IDS:
        _special_token_id = SPECIAL_TOKEN_IDS[task_load]
    else:
        # get special token id as the initial + index of that task
        _special_token_id = SPECIAL_TOKEN_IDS[tasks[0]] + tasks.index(task_load)

        
    # Test Dataset : Support (Train QAData) Query (Test QAData)
    support_qadata = QADataset(TASK_DICT[task_eval]["train"], "train", SPECIAL_TOKEN_IDS[task_load])
    test_qadata = QADataset(TASK_DICT[task_eval]["test"] , "test", SPECIAL_TOKEN_IDS[task_load]).sort()
    
    max_a_len = test_qadata.max_a_len
    n_examples = len(test_qadata)
    logger.info("len of test dataset: {}".format(n_examples))
    print("len of test dataset: {}".format(n_examples))
    
    
    ##### Make dataloaders for that particular dataset #####
    support_dataloader = create_dataloader(support_qadata, "train")
    test_dataloader = create_dataloader(test_qadata, "test")
    
    
    ##### Stream from that dataset's dataloader #####
    iter_support_dataloader = iter(support_dataloader)
    iter_test_dataloader = iter(test_dataloader)

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
#     for n_steps, (cq, len_cq, cqa, len_cqa, Y, genX, genY) in enumerate(test_dataloader):
        # 1. Get the support data from the train dataloader
        #    and the query data from the test dataloader
        # Assume that query data >> support data!
        try:
            
            _, _, support_x, _, support_y, _, _ = next(iter_support_dataloader)
            query_x, query_x_len, query_x_cqa, _, query_y, _, _ = next(iter_test_dataloader) # Let query get the CQ!

            # Different inputs for train and test -> train with batch 3 and test with batch 1
            n_inputs_train = sum(_cqa.shape[0] for _cqa in support_x)
            n_inputs = sum(_cqa.shape[0] for _cqa in query_x)

            # Since we only have 1 GPU, just use the first one, it will separate batches according to the device IDS
            support_x = support_x[0]
            support_y = support_y[0]
            query_x = query_x[0]
            query_y = query_y[0]
            query_x_len = query_x_len[0] # an array of query x lengths, but test batch size is only1??
            query_x_cqa = query_x_cqa[0] #EXTRA DEBUG

            support_x = support_x.to(DEVICE)
            support_y = support_y.to(DEVICE)
            query_x = query_x.to(DEVICE)
            query_y = query_y
            query_x_cqa = query_x_cqa.to(DEVICE) #EXTRA DEBUG

        except StopIteration:
            break
        
        
        ### START Adaptation Phase ###
        # 2. Reinitialize model with parameters from model_path
        #    For this case, make new model with copy from model params
#         model_copy = MODEL_CLASS.from_pretrained('gpt2').to(DEVICE)
#         model_copy.resize_token_embeddings(len(TOKENIZER))
#         model_copy = FP16_Module(model_copy)
#         for m_from, m_to in zip(model.modules(), model_copy.modules()):
#             m_to.load_state_dict(m_from.state_dict().copy())
        model_copy = copy.deepcopy(model)
        model_copy.train()
        
        # Training loss function
        train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT)
        
        # Optimizer
        max_grad_norm=1
        param_optimizer = list(model_copy.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=6.25e-5, eps=1e-4)
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                           dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})

        
        # 3. Update the weights with the support set
        # May update for several steps
        for i in range(num_updates):

            qa_logits = model_copy(support_x)
            # Somehow it also returns attentions in [1]?, this is selecting 0 of what WrapModel is doing 
            qa_logits = qa_logits[0]
            qa_loss = train_loss_fct(qa_logits.transpose(1,2), support_y)
            loss = qa_loss

            logger.info(f"[DEBUG] Adaptation loss: {qa_loss.item()}")
            # Update Optimizer
            optimizer.backward(loss, update_master_grads=False) # instead of loss.backward() for fp16
            optimizer.update_master_grads()
            optimizer.clip_master_grads(max_grad_norm)
            optimizer.step()
            # Ignore this for now
#             if not optimizer.overflow:
#                 for i in range(n_inputs):
#                     scheduler.step()
            optimizer.zero_grad()
        ### END Adaptation Phase ###
        
        model_copy.eval()
        
        ### START Meta-Learning Phase ###
        # 4. After Adaptation, use the query set for test (CQ ONLY)
        # model() returns Tuple of length 2: 
        #  The [0] is a  torch.Size([1, 225, 50260]), and the [1] is 12 of torch.Size([2, 1, 12, 225, 64])
        # Thinking that the [0] is the actual output and [1] is the pasts?
        all_outputs = model_copy(query_x)
        outputs = all_outputs[0]
        pasts = all_outputs[1]
        next_logits = outputs[range(n_inputs), query_x_len-1, :] / temperature_qa
        next_tokens = logits_to_tokens(next_logits).cpu()
        
        
        # EXTRA FOR COMPARE
        qa_logits = model_copy(query_x_cqa)[0]
        qa_loss = train_loss_fct(qa_logits.transpose(1,2), query_y.to(DEVICE))
        logger.info(f"[DEBUG] QUERY LOSS: {qa_loss.item()}")
        
        
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
                if len(qa_results[cnt]) not in [max_tot_lens[cnt], max_len]:
                    # Append need_process of that cnt
                    need_process.update([[cnt, None]])
                    # Update all pasts
                    for layer_id in range(MODEL_CONFIG.n_layer):
                        all_pasts[layer_id][cnt] = pasts[layer_id][:, batch_i, ..., :query_x_len[batch_i], :].type(torch.half)
            
            # Try sample_sequence here! it will get all need_process (should be only 1 batch, and generate all!)
            sample_sequence(model_copy, need_process, qa_results, all_pasts, max_tot_lens)
            
            
            
            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Predicted Answer {TOKENIZER.decode(qa_results[cnt].tolist())}")
            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Predicted Tokens {qa_results[cnt].tolist()[query_x_len:]}")

            # Do the score calculation here
            # The answer of that particular batch to list
            Y = query_y[batch_i].tolist()
            Y = list(filter(lambda x: x != -1, Y))[:-1]  # remove eos from the answer
            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Actual Tokens {Y}")
            Y = ' '.join([str(y) for y in Y]).split(str(SPECIAL_TOKEN_IDS["pad_token"]))
            Y = [TOKENIZER.decode(list(map(int, y.split()))) for y in Y]
            # Change the QA Results to a decoded version of real answer and predicted answer
            qa_results[cnt] = [TOKENIZER.decode(qa_results[cnt].tolist()[query_x_len:]), Y]
            print(f"Predict vs Actual {cnt}/{n_examples}", qa_results[cnt])
            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Actual Answer {Y}")
            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Predict vs Actual {qa_results[cnt]}")
            
            cnt += 1
        n_steps += 1
        
    get_test_score(task_eval, qa_results, score_dict)
    print(score_dict)
    
    model_dir = model.model_dir
    results_path = os.path.join(model_dir,f"qa_{task_eval}.csv")
    with open(results_path, "w",encoding="utf-8") as f:
        qa_writer = csv.writer(f,delimiter=',')
        qa_writer.writerow(["y","pred"])
        for pred, y in qa_results:
            qa_writer.writerow([y,pred])

    return model_copy, score_dict

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
    
    print(f"Starting Run with RUN_ID {RUN_ID}")
    print(f"[TIME] Start Run {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    logger.info(f"[TIME] Start Run {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    tic_RUN = time.time()
    
    main()
    
    toc_RUN = time.time() - tic_RUN
    print(f"[TIME] End Run {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
    logger.info(f"[TIME] End Run {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
    