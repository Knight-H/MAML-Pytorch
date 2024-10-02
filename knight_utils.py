import os
import argparse
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from pytorch_transformers import GPT2LMHeadModel, GPT2Config #, GPT2Tokenizer
from transformers import GPT2Tokenizer # Use the new tokenizer for is_split_into_words 
from torch.utils.data import Dataset, DataLoader, Sampler
import json, csv
from collections import OrderedDict
from multiprocessing import Pool
import warnings
import numpy as np
import uuid
import re
import random
import logging
import GPUtil
import datetime
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# data_dir = "/root/LAMOL/lamol_data"
MODEL_BASE_DIR = "/data/model_runs"

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config),
}
CONFIG_NAME = 'model_config.json'

DEVICE = 'cuda:0'
# DEVICE = "/root/MAML-Pytorch/fp16.py" 'cpu' TypeError: Wrapped parameters must be either torch.cuda.FloatTensor or torch.cuda.HalfTensor. Received torch.HalfTensor

LEN_FACTOR = 1.163
FILL_VAL = -1
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train_epochs", type=int, default=1)
    parser.add_argument("--tokens_weight", type=float, default=5)
    parser.add_argument("--test_batch_size", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=0)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--min_batch_size", type=int, default=4)
    parser.add_argument("--min_n_steps", type=int, default=1500)
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--num_updates", type=int, default=5, help='Meta Learning Adaptation phase number of updates')
    parser.add_argument("--update_lr", type=float, default=6.25e-5, help='Adaptation learning rate')
    parser.add_argument("--meta_lr", type=float, default=6.25e-5, help='Meta Learning learning rate')
    parser.add_argument("--use_sep", action="store_true")
    parser.add_argument("--lm_lambda", type=float, default=0.25)
    # Directories
    parser.add_argument("--data_dir", type=str, default="/data/lamol_data")
    #Tasks
    #parser.add_argument("--tasks", nargs='+', default=['movie', 'boolq', 'scifact'])
    parser.add_argument("--tasks", nargs='+', default=["yelp", "ag", "dbpedia", "amazon", "yahoo"])
    # For test
    parser.add_argument("--model_dir_name", type=str)
    # Generation Sample Sequence
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature_lm", type=float, default=1.0)
    parser.add_argument("--temperature_qa", type=float, default=1.0)
    parser.add_argument("--top_k_lm", type=int, default=20)
    parser.add_argument("--top_k_qa", type=int, default=20)
    parser.add_argument("--top_p_lm", type=float, default=0.)
    parser.add_argument("--top_p_qa", type=float, default=0.)
    # Extra Data Arguments
    parser.add_argument("--is_lamol", action="store_true") #FOR USE IN TESTING - Whether to include lm in adaptation
    parser.add_argument("--real_sample", action="store_true")
    parser.add_argument("--gen_lm_sample_percentage", type=float, default=0.05)
    # For Few Rel
    parser.add_argument('--order', type=int, help='Number of task orders to run for', default=5)
    parser.add_argument('--num_clusters', type=int, help='Number of clusters to take', default=10)

    args, unknown = parser.parse_known_args()
    
    
    # Fix Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministric = True # https://stackoverflow.com/questions/66130547/what-does-the-difference-between-torch-backends-cudnn-deterministic-true-and
    
    torch.cuda.set_device(DEVICE)
    args.device_ids = GPUtil.getAvailable(maxLoad=0.1, maxMemory=0.05, limit=args.n_gpus)
    gpus = GPUtil.getGPUs()
    gpu_names = [gpus[device_id].name for device_id in args.device_ids]
    print(gpu_names)
    
    
    # In settings.py
    special_tokens = {"ans_token":'__ans__', "pad_token":'__pad__', "unk_token":'__unk__', "eos_token": '<|endoftext|>'}
    
    model_class, tokenizer_class, config_class = MODEL_CLASSES['gpt2']
    tokenizer = tokenizer_class.from_pretrained('gpt2')
    tokenizer.add_tokens(list(special_tokens.values()))
    special_token_ids = {k:tokenizer.convert_tokens_to_ids(v) for k,v in special_tokens.items()}

    model_config = config_class.from_pretrained('gpt2')
    model_config.vocab_size = len(tokenizer)
    args.max_len = model_config.n_positions

    tokens_weight = torch.ones([model_config.vocab_size], dtype=torch.float).to(DEVICE)
    tokens_weight[special_token_ids["ans_token"]] = args.tokens_weight  # only answer token has token weight of 5! (default)

    # data_attrs -> get from preprocess?!? but where?
    # All the train/dev/test legnth of all datasets. Since we create_extra_data
    data_attrs_path = os.path.join(BASE_DIR,"data_attrs.json")
    assert os.path.exists(data_attrs_path)
    with open(data_attrs_path, "r") as f:
        data_attrs = json.load(f)
    
    
    return args, model_config, model_class, tokenizer, config_class, special_token_ids, special_tokens, data_attrs, tokens_weight


args, MODEL_CONFIG, MODEL_CLASS, TOKENIZER, CONFIG_CLASS, SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, DATA_ATTRS, TOKENS_WEIGHT = parse_args()


TASK_DICT = {
    "movie": {
               "train":os.path.join(args.data_dir,"movie_train.json"),
               "eval":os.path.join(args.data_dir,"movie_dev.json"),
               "test":os.path.join(args.data_dir,"movie_test.json"),
               "n_train_epochs": args.n_train_epochs  
    },
    "boolq": {
               "train":os.path.join(args.data_dir,"boolq_train.json"),
               "eval":os.path.join(args.data_dir,"boolq_dev.json"),
               "test":os.path.join(args.data_dir,"boolq_test.json"),
               "n_train_epochs": args.n_train_epochs  
    },
    "scifact": {
               "train":os.path.join(args.data_dir,"scifact_train.json"),
               "eval":os.path.join(args.data_dir,"scifact_dev.json"),
               "test":os.path.join(args.data_dir,"scifact_test.json"),
               "n_train_epochs": args.n_train_epochs  
    },
    "sst": {
               "train":os.path.join(args.data_dir,"sst_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"sst_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"sst_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "srl": {
               "train":os.path.join(args.data_dir,"srl_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"srl_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"srl_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "woz.en": {
               "train":os.path.join(args.data_dir,"woz.en_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"woz.en_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"woz.en_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "ag": {
               "train":os.path.join(args.data_dir,"ag_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"ag_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"ag_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "dbpedia": {
               "train":os.path.join(args.data_dir,"dbpedia_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"dbpedia_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"dbpedia_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "yahoo": {
               "train":os.path.join(args.data_dir,"yahoo_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"yahoo_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"yahoo_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "amazon": {
               "train":os.path.join(args.data_dir,"amazon_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"amazon_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"amazon_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "yelp": {
               "train":os.path.join(args.data_dir,"yelp_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"yelp_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"yelp_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "ag10k": {
               "train":os.path.join(args.data_dir,"ag_to_squad-train-v2.0-10k.json"),
               "eval":os.path.join(args.data_dir,"ag_to_squad-test-v2.0-10k.json"),
               "test":os.path.join(args.data_dir,"ag_to_squad-test-v2.0-10k.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "dbpedia10k": {
               "train":os.path.join(args.data_dir,"dbpedia_to_squad-train-v2.0-10k.json"),
               "eval":os.path.join(args.data_dir,"dbpedia_to_squad-test-v2.0-10k.json"),
               "test":os.path.join(args.data_dir,"dbpedia_to_squad-test-v2.0-10k.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "yahoo10k": {
               "train":os.path.join(args.data_dir,"yahoo_to_squad-train-v2.0-10k.json"),
               "eval":os.path.join(args.data_dir,"yahoo_to_squad-test-v2.0-10k.json"),
               "test":os.path.join(args.data_dir,"yahoo_to_squad-test-v2.0-10k.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "amazon10k": {
               "train":os.path.join(args.data_dir,"amazon_to_squad-train-v2.0-10k.json"),
               "eval":os.path.join(args.data_dir,"amazon_to_squad-test-v2.0-10k.json"),
               "test":os.path.join(args.data_dir,"amazon_to_squad-test-v2.0-10k.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "yelp10k": {
               "train":os.path.join(args.data_dir,"yelp_to_squad-train-v2.0-10k.json"),
               "eval":os.path.join(args.data_dir,"yelp_to_squad-test-v2.0-10k.json"),
               "test":os.path.join(args.data_dir,"yelp_to_squad-test-v2.0-10k.json"),
               "n_train_epochs": args.n_train_epochs 
    }
}


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
        if len(data_paths)==1 and data_paths[0] is not None and ('wiki' in data_paths[0] or 'woz' in data_paths[0]):
            #data = self._sort_by_index(data)
            #args.n_workers = 1
            if 'wiki' in data_paths[0]:
                answers_file = "wikisql_answers.json" 
            elif 'woz' in data_paths[0]:
                answers_file = "woz.en_answers.json" 
            with open(os.path.join(args.data_dir,answers_file),"r") as f:
                self.answers = json.load(f)
        if len(data) > 0:
            self.data_tokenization(data)

        if len(extra_data) > 0:
            extra_data = map(lambda x: self.etl_single_extra_data(x), extra_data)
            # Filter all with len(cq_example) == 0; happens with Q is concat with C. (Without SEP token, this will happen for real samples!)
            # See how etl_single_extra_data is written for "context"! By default it will be ""-->[]
            # So actual extra data will be filtered... 
            extra_data = list(filter(lambda x: len(x[0]) > 0, extra_data))
            if args.gen_lm_sample_percentage > 0. and len(extra_data) == 0:
                logger.warning("No good extra data but sample percentage > 0!")
            print(f"Actual extra data: {len(extra_data)}")
            logger.info(f"Actual extra data: {len(extra_data)}")
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
            print(f"PARSE ERROR > {gen_token}, {context}, {question}, {answer}, {uid}")
            return
        except:
            print(f"PARSE ERROR >\nGen:{gen_token}\nCTX: {context}\nQ:{question}\nA:{answer}\n{uid}")
            print([TOKENIZER.convert_ids_to_tokens(int(a)) for a in qa.split()])
            print()
            return
        return data

    def concat_example(self, gen_token, c, sep_token, q, ans_token, a, eos_token):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > args.max_len:
            logger.warning('an example with len {} is too long!'.format(len(example) + 1))
            return [] # this is a bug when you use without [], because len() will not be effective!
        example = gen_token + c[:args.max_len-len(example)-1] + example + eos_token
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

# Similar to FewRel, just tokenized and everything
class FewRelQADataset(Dataset):
    def __init__(self, fewrel_dataset, gen_token, extra_data=[]):
        self.gen_token = gen_token
        self.ans_token = SPECIAL_TOKEN_IDS["ans_token"]
        self.eos_token = SPECIAL_TOKEN_IDS["eos_token"]
        self.pad_token = SPECIAL_TOKEN_IDS["pad_token"]

        self.data = []
        self.max_a_len = 0
        if fewrel_dataset:
            self.data_tokenization(fewrel_dataset)

        if len(extra_data) > 0:
            extra_data = map(lambda x: self.etl_single_extra_data(x), extra_data)
            # Filter all with len(cq_example) == 0; happens with Q is concat with C. (Without SEP token, this will happen for real samples!)
            # See how etl_single_extra_data is written for "context"! By default it will be ""-->[]
            # So actual extra data will be filtered... 
            extra_data = list(filter(lambda x: len(x[0]) > 0, extra_data))
            if args.gen_lm_sample_percentage > 0. and len(extra_data) == 0:
                logger.warning("No good extra data but sample percentage > 0!")
            print(f"Actual extra data: {len(extra_data)}")
            logger.info(f"Actual extra data: {len(extra_data)}")
            self.data += extra_data

    def data_tokenization(self, data):
        with Pool(4) as pool:
            data = pool.map(self.parallel_tokenization, data)
        for datum, max_a_len in data:
            self.data.extend(datum)
            self.max_a_len = max(self.max_a_len, max_a_len)

    # d is now a single data instance of fewrel dataset
    def parallel_tokenization(self, d):
        text, label, candidates = d
        examples = []
        # Check if this is correct? because text is a list of string right?? did it expect this?
        context = TOKENIZER.encode(" ".join(text), is_split_into_words=True) # There will only be one text, so there's no need to do it again
        max_a_len = 0
        replicated_text, replicated_relations, ranking_label = replicate_rel_data([text],[label],[candidates])
        # print("this is rep text ", replicated_text)
        for rt, rr, rl in zip(replicated_text, replicated_relations, ranking_label):
            q = "Is this the relation of " + " ".join(rr) + "?"
            question = TOKENIZER.encode(q)
            a = "Yes" if rl == 1 else "No"
            answer = TOKENIZER.encode(a, is_split_into_words=True)
            max_a_len = max(max_a_len, len(answer))
            examples.append(self.parse_example(self.gen_token, context, question, answer, 0))
        return examples, max_a_len
    
    def concat_example(self, gen_token, c, sep_token, q, ans_token, a, eos_token):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > args.max_len:
            logger.warning('an example with len {} is too long!'.format(len(example) + 1))
            return [] # this is a bug when you use without [], because len() will not be effective!
        example = gen_token + c[:args.max_len-len(example)-1] + example + eos_token
        return example

    def parse_example(self, gen_token, context, question, answer, idx):
        cq_example = self.concat_example([], context, [], question, [self.ans_token], [], [])
        cqa_example = self.concat_example([], context, [], question, [self.ans_token], answer, [])
        Y_example = self.concat_example([], [], [], [], [], answer, [self.eos_token])
        Y_example = [FILL_VAL] * (len(cqa_example) - len(Y_example)) + Y_example
        gen_X_example = self.concat_example([gen_token], context, [], question, [self.ans_token], answer, [])
        gen_Y_example = self.concat_example([], context, [], question, [self.ans_token], answer, [self.eos_token])
        return cq_example, len(cq_example), cqa_example, len(cqa_example), Y_example, gen_X_example, gen_Y_example, idx
    
    # data is the output of train_extra_data, which comes out from
    # train_extra_data.extend([TOKENIZER.encode(x) for x in d])  (just a list of tokenized numbers)
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
            print(f"PARSE ERROR > {gen_token}, {context}, {question}, {answer}, {uid}")
            return
        except:
            print(f"PARSE ERROR >\nGen:{gen_token}\nCTX: {context}\nQ:{question}\nA:{answer}\n{uid}")
            print([TOKENIZER.convert_ids_to_tokens(int(a)) for a in qa.split()])
            print()
            return
        return data
    
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


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, data_type, max_batch_size):
        self.dataset = dataset
        self.data_type = data_type
        if data_type == "train":
            self.batch_size = args.train_batch_size
        else:
            self.batch_size = args.test_batch_size
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



def create_dataloader(dataset, data_type, max_batch_size=1000000000):
    if data_type == "train":
        batch_size = args.train_batch_size
    else:
        batch_size = args.test_batch_size

    if isinstance(batch_size, list):
        print("I MADE DATA LOAD DYNAMIC")
        collate_fn=lambda x,bs=batch_size: dynamic_collate_fn(x, bs)
        shuffle = False
        batch_size = 1
        batch_sampler = DynamicBatchSampler(dataset, data_type, max_batch_size)
    else:
        print("I MADE DATA LOAD varlen_collate_fn")
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


def varlen_collate_fn(data):
    batch_size = (len(data) + args.n_gpus - 1) // args.n_gpus
    cqs = torch.tensor(pad_all_to_max_len([datum[0] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    len_cqs = torch.tensor([datum[1] for datum in data]).split(batch_size)
    cqas = torch.tensor(pad_all_to_max_len([datum[2] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    len_cqas = torch.tensor([datum[3] for datum in data]).split(batch_size)
    Ys = torch.tensor(pad_all_to_max_len([datum[4] for datum in data], FILL_VAL)).split(batch_size)
    gen_Xs = torch.tensor(pad_all_to_max_len([datum[5] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    gen_Ys = torch.tensor(pad_all_to_max_len([datum[6] for datum in data], FILL_VAL)).split(batch_size)
    return list(cqs), list(len_cqs), list(cqas), list(len_cqas), list(Ys), list(gen_Xs), list(gen_Ys)

def pad_to_max_len(l, pad_len, val):
    return l + [val] * pad_len
def pad_all_to_max_len(ls, val):
    max_len = max(len(l) for l in ls)
    return [pad_to_max_len(l, max_len-len(l), val) for l in ls]


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
        decode_batch_size = args.test_batch_size
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

            next_logits = outputs[..., -1, :] / args.temperature_qa
            next_tokens = logits_to_tokens(next_logits).cpu()

            for i, cur_id in enumerate(batch_ids):
                if next_tokens[i] == SPECIAL_TOKEN_IDS["eos_token"]:
                    remove_ids.append(cur_id)
                else:
                    qa_results[cur_id] = torch.cat((qa_results[cur_id], next_tokens[i]))
                    if len(qa_results[cur_id]) in [max_tot_lens[cur_id], args.max_len]:
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
    filtered_logits = top_k_top_p_filtering(next_logits, top_k=args.top_k_qa, top_p=args.top_p_qa)
    log_probs = F.softmax(filtered_logits, dim=-1)
    next_tokens = torch.multinomial(log_probs, num_samples=1)
    return next_tokens





###### Extra Data #####
def create_extra_data(task, prev_task, model, train_extra_data, model_dir):
    # Real Sample - why is real samples not os.path.exist check? generate everytime??
    if args.real_sample:
        print(f"using real data as extra data")
        logger.info(f"using real data as extra data")
        return get_real_data(task, train_extra_data, model_dir)
    
    #MODEL_SAVE_LOC = os.path.join(model_dir, f'{_tasks[0]}.model')
    task_cnt = args.tasks.index(task)
    gen_path = os.path.join(model_dir,f"lm-{task}-{prev_task}.csv")  # MAJOR EDIT - NEEDS TO BE 5% of the current task!!!
    if os.path.exists(gen_path):
        print(f"extra data exists in {gen_path}, read it!")
        logger.info(f"extra data exists in {gen_path}, read it!")
        return read_extra_data(gen_path, train_extra_data) 
    # Generation size = Train data * 0.05
    gen_size = DATA_ATTRS[task]["train"]["data_size"]
    gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage))
    gen_size -= (gen_size % task_cnt)

    model.eval()
    print(f"Generating extra data! With gen_size {gen_size}")
    logger.info(f"generating extra data!  With gen_size {gen_size}")
    
    need_process = OrderedDict()
    qa_results = []
    for task_name in args.tasks[:task_cnt]:
        qa_results.extend([torch.tensor([SPECIAL_TOKEN_IDS[task_name]]) for _ in range(gen_size//task_cnt)])
    all_pasts = [[
        torch.empty(2, MODEL_CONFIG.n_head, 0, MODEL_CONFIG.n_embd//MODEL_CONFIG.n_head, dtype=torch.half).cuda()
        for _ in range(gen_size)
    ] for __ in range(MODEL_CONFIG.n_layer)]
    max_tot_lens = [args.max_len for _ in range(gen_size)]

    for i in range(gen_size):
        need_process.update([[i, None]])
        # If it's too long it will error! need to do it first
        # V100 is 32MB, so just make do with this!
        if len(need_process) > int(32510 * 0.12):
            sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)
    sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)

    model.train()

    qa_results = [res.tolist() for res in qa_results]
    train_extra_data.extend(qa_results)
    qa_results = [TOKENIZER.decode(res) for res in qa_results]

    write_extra_data(gen_path, qa_results)

# only real samples
def create_extra_data_rel(train_id, train_datasets, train_extra_data, model_dir):
    if args.real_sample:
        prev_tasks = train_datasets[:train_id]
        logger.info(f"This is prev tasks {[len(x) for x in prev_tasks]}")
        gen_size = sum([len(x) for x in prev_tasks])
        gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage))//len(prev_tasks)
        logger.info(f"generating extra data!  With gen_size {gen_size}")
        datum = []
        # Now prev_dataset is a LifelongFewRelDataset
        for prev_id, prev_dataset in enumerate(prev_tasks):
            task_id = args.shuffle_index.index(prev_id)
            indices = np.random.choice(range(len(prev_dataset)), gen_size)
            for i in indices:
                d = parse_single_real_data_rel(prev_dataset[i],task_id) # d is now multiple because of candidates!
                datum.extend(d)
                train_extra_data.extend([TOKENIZER.encode(x) for x in d])
        dump_path = os.path.join(model_dir,f"real-{train_id}.csv")
        write_extra_data(dump_path, datum)
        return dump_path
    else:
        raise "Not Implemented"
    
    


def get_real_data(task, train_extra_data, model_dir, accum=True, encode=True):
    task_idx = args.tasks.index(task)
    gen_size = DATA_ATTRS[task]["train"]["data_size"]
    if accum:
        prev_tasks = args.tasks[:task_idx]
        gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage))//len(prev_tasks)
    else:
        prev_tasks = [args.tasks[task_idx-1]]
        gen_size = int(gen_size * args.gen_lm_sample_percentage)
        
    print(f"Generating extra data! With gen_size {gen_size}")
    logger.info(f"generating extra data!  With gen_size {gen_size}")

    datum = []
    for prev_task in prev_tasks:
        with open(TASK_DICT[prev_task]["train"],"r") as f:
            data = data_expand(json.load(f)["data"])
        indices = np.random.choice(range(len(data)), gen_size)
        for i in indices:
            d = parse_single_real_data(data[i],prev_task)
            datum.append(d)
            if encode:
                train_extra_data.append(TOKENIZER.encode(d))

    dump_path = os.path.join(model_dir,f"real-{prev_task}.csv")
    write_extra_data(dump_path, datum)
    return dump_path


def read_extra_data(gen_path, train_extra_data):
    with open(gen_path,"r") as lm_file:
        reader = csv.reader(lm_file,delimiter=',')
        next(reader)
        for row in reader: 
            row = TOKENIZER.encode(row[0].strip()) 
            train_extra_data.append(row)

def write_extra_data(dump_path, qa_results):
    print(f"writing extra data in {dump_path} ...")
    logger.info(f"writing extra data in {dump_path} ...")
    with open(dump_path,"w",newline="",encoding="utf-8") as f:
        lm_writer = csv.writer(f,delimiter=',')
        lm_writer.writerow(["gen"])
        for l in qa_results:
            lm_writer.writerow([l])


def parse_single_real_data(data,task):
    c = data["paragraphs"][0]["context"]
    q = data["paragraphs"][0]["qas"][0]["question"]
    a = data["paragraphs"][0]["qas"][0]["answers"][0]["text"]
    if args.use_sep:
        data = "{}{}{}{}{}{}{}".format(SPECIAL_TOKENS[task],c,SPECIAL_TOKENS["sep_token"],q,SPECIAL_TOKENS["ans_token"],a,SPECIAL_TOKENS["eos_token"])
    else:
        data = "{}{} {}{}{}{}".format(SPECIAL_TOKENS[task],c,q,SPECIAL_TOKENS["ans_token"],a,SPECIAL_TOKENS["eos_token"])
    return data

# Data is now a single instance of LifelongFewRelDataset
# 1 data = multiple candidates. so data output will be a list!
def parse_single_real_data_rel(data,task):
    text, label, candidates = data
    replicated_text, replicated_relations, ranking_label = replicate_rel_data([text],[label],[candidates])
    d = []
    for rt, rr, rl in zip(replicated_text, replicated_relations, ranking_label):
        c = " ".join(rt)
        q = "Is this the relation of " + " ".join(rr) + "?"
        a = "Yes" if rl == 1 else "No"
        if args.use_sep:
            d.append("{}{}{}{}{}{}{}".format(SPECIAL_TOKENS[task],c,SPECIAL_TOKENS["sep_token"],q,SPECIAL_TOKENS["ans_token"],a,SPECIAL_TOKENS["eos_token"]))
        else:
            d.append("{}{} {}{}{}{}".format(SPECIAL_TOKENS[task],c,q,SPECIAL_TOKENS["ans_token"],a,SPECIAL_TOKENS["eos_token"]))
    return d


def data_expand(data):
    datum = []
    for d in data:
        para = d["paragraphs"]
        for p in para: 
            for qa in p["qas"]:
                d = {"context": p["context"], "qas": [qa]}
                datum.append({"paragraphs":[d]})
    return datum


## ========== Utils For FewRel======================== 
class LifelongFewRelDataset(Dataset):
    def __init__(self, data, relation_names):
        self.relation_names = relation_names
        self.label = []
        self.candidate_relations = []
        self.text = []

        for entry in data:
            self.label.append(self.relation_names[entry[0]])
            negative_relations = entry[1]
            candidate_relation_names = [self.relation_names[x] for x in negative_relations]
            self.candidate_relations.append(candidate_relation_names)
            self.text.append(entry[2])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.text[index], self.label[index], self.candidate_relations[index]

# Copied from em-in-lll
class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
          last = self.last
        except AttributeError:
          last = record.relativeCreated

        delta = record.relativeCreated/1000 - last/1000
        record.relative = "{:.3f}".format(delta)
        record.uptime = str(datetime.timedelta(seconds=record.relativeCreated//1000))
        self.last = record.relativeCreated
        return True
    

def replicate_rel_data(text, label, candidates):
    replicated_text = []
    replicated_relations = []
    ranking_label = []
    for i in range(len(text)):
        replicated_text.append(text[i])
        replicated_relations.append(label[i])
        ranking_label.append(1)
        for j in range(len(candidates[i])):
            replicated_text.append(text[i])
            replicated_relations.append(candidates[i][j])
            ranking_label.append(0)
    return replicated_text, replicated_relations, ranking_label


def remove_return_sym(str):
    return str.split('\n')[0]
    
def read_relations(relation_file):
    relation_list = ['fill']
    with open(relation_file, encoding='utf8') as file_in:
        for line in file_in:
            line = remove_return_sym(line)
            line = re.sub(r'/', '', line)
            line = line.split()
            relation_list.append(line)
    return relation_list
def read_rel_data(sample_file):
    sample_data = []
    with open(sample_file, encoding='utf8') as file_in:
        for line in file_in:
            items = line.split('\t')
            if len(items[0]) > 0:
                relation_ix = int(items[0])
                if items[1] != 'noNegativeAnswer':
                    candidate_ixs = [int(ix) for ix in items[1].split() if int(ix) != relation_ix]
                    sentence = remove_return_sym(items[2]).split()
                    sample_data.append([relation_ix, candidate_ixs, sentence])
    return sample_data


def get_relation_embedding(relations, glove):
    rel_embed = []
    for rel in relations:
        word_embed = glove.get_vecs_by_tokens(rel, lower_case_backup=True)
        if len(word_embed.shape) == 2:
            rel_embed.append(torch.mean(word_embed, dim=0))
        else:
            rel_embed.append(word_embed)
    rel_embed = torch.stack(rel_embed)
    return rel_embed


def get_relation_index(data):
    relation_pool = []
    for entry in data:
        relation_number = entry[0]
        if relation_number not in relation_pool:
            relation_pool.append(relation_number)
    return relation_pool

def create_relation_clusters(num_clusters, relation_embedding, relation_index):
    ordered_relation_embedding = np.zeros_like(relation_embedding[1:])
    for i, rel_idx in enumerate(relation_index):
        ordered_relation_embedding[i] = relation_embedding[rel_idx]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(ordered_relation_embedding)
    labels = kmeans.labels_
    rel_embed = {}
    cluster_index = {}
    for i in range(len(labels)):
        cluster_index[relation_index[i]] = labels[i]
        rel_embed[relation_index[i]] = relation_embedding[i]
    return cluster_index, rel_embed


# Cluster  Idx  0                        1                      ....  9
#            [ [data0, data1, ...],     [data0, data1, ...] .....   [data0, data1, ...]]
def split_rel_data_by_clusters(data_set, cluster_labels, num_clusters, shuffle_index):
    splitted_data = [[] for i in range(num_clusters)]
    for data in data_set:  # for all 44.5k data
        cluster_number = cluster_labels[data[0]]      # (data[0]) relation idx of that data map--> cluster index
        index_number = shuffle_index[cluster_number]  # cluster index --> order of the shuffle
        splitted_data[index_number].append(data)      # place it in the bucket of num_clusters 
    return splitted_data



def remove_unseen_relations(dataset, seen_relations):
    cleaned_data = []
    for data in dataset:
        neg_cands = [cand for cand in data[1] if cand in seen_relations]
        if len(neg_cands) > 0:
            cleaned_data.append([data[0], neg_cands, data[2]])
        else:
            cleaned_data.append([data[0], data[1][-2:], data[2]])
    return cleaned_data


# Number of clusters = 10
#   shuffle_index       = shuffled num_clusters ( 0, 1, 2, 3 ... 9)
#                         There are 0-9 clusters, the shuffle_index corresponds to which cluster comes first. NOT the cluster idx
#                         ie. [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]  this means
#                         [0 = cluster8, 1 = cluster9, 2= cluster 2,... ]
#   splitted_train_data = Split training dataset by clusters (in shuffled index)
#   For each cluster:
#         add all seen_relations of that cluster
#         current_train_data = remove_unseen_relations in current cluster 
#   train_datasets = [cluster0, cluster1, ... cluster 9]
def prepare_rel_datasets(train_data, relation_names, cluster_labels, num_clusters):
    train_datasets = []

    shuffle_index = list(range(num_clusters))
    random.shuffle(shuffle_index)

    splitted_train_data = split_rel_data_by_clusters(train_data, cluster_labels, num_clusters, shuffle_index)
    seen_relations = []

    for i in range(num_clusters):
        for data_entry in splitted_train_data[i]:
            if data_entry[0] not in seen_relations:
                seen_relations.append(data_entry[0])

        current_train_data = remove_unseen_relations(splitted_train_data[i], seen_relations)

        train_dataset = LifelongFewRelDataset(current_train_data, relation_names)
        train_datasets.append(train_dataset)
    return train_datasets, shuffle_index