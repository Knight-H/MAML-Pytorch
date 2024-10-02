### Changelog v2
# Changed inner optimizer from Adam to SGD lr (update_lr) 3e-3, changed Adam outer to (meta_lr) 3e-5 (per MAML-CL paper)
# Remove LM Loss in Adaptation. (For LAMOL)
# Thinking just truncating sequence length like MAML-CL? but not now. 

import torch
from torch import nn, optim
from fp16 import FP16_Module, FP16_Optimizer
# from pytorch_transformers import AdamW
import gc
from copy import deepcopy
from knight_utils import FILL_VAL, TOKENIZER, SPECIAL_TOKEN_IDS

import logging
logger = logging.getLogger(__name__)

# Copied from 
# https://github.com/dragen1860/MAML-Pytorch/blob/master/meta.py#L118

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, net_cls, vocab_size, device, is_lamol=False, lm_lambda=0.25):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.num_updates = args.num_updates
        self.device = device
        self.max_grad_norm = args.max_grad_norm
        self.is_lamol = is_lamol
        self.lm_lambda = lm_lambda
        
        print("Initializing Model...")
        self.net = net_cls.from_pretrained('gpt2').to(self.device)
        self.net.resize_token_embeddings(vocab_size)
        self.net = FP16_Module(self.net)
        
        
        # Optimizer
        meta_param_optimizer = list(self.net.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        meta_optimizer_grouped_parameters = [
            {'params': [p for n, p in meta_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in meta_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.meta_optimizer = optim.AdamW(meta_optimizer_grouped_parameters, lr=self.meta_lr, eps=1e-4)
        self.meta_optimizer = FP16_Optimizer(self.meta_optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                           dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})
        
    def forward(self, support_x, support_y, query_x, query_y, train_loss_fct, support_gen_x=None, support_gen_y=None, query_gen_x=None, query_gen_y=None):
        
        all_loss = []
        sum_gradients = []
        accs = [] # TODO
        
        ### START Adaptation Phase ###
        # 1. Update the weights with the support set
        
        
        # Copy from https://github.com/mailong25/meta-learning-bert/blob/master/maml.py
        fast_model = deepcopy(self.net)
        fast_model.to(self.device)
        param_optimizer = list(fast_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        inner_optimizer = optim.SGD(optimizer_grouped_parameters, lr=self.update_lr)
        inner_optimizer = FP16_Optimizer(inner_optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                           dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})
        
        fast_model.train()
        

        for k in range(0, self.num_updates):

            qa_logits = fast_model(support_x)[0]
            qa_loss = train_loss_fct(qa_logits.transpose(1,2), support_y)
            loss = qa_loss
            
            ##### LAMOL_SPECIFIC! This section adds the data (task, prev_task, model, train_extra_data, model_dir, tasks)
            # V2 remove this. For optimization.& you don't really need it?!?
#             if self.is_lamol:
#                 lm_logits = fast_model(support_gen_x)[0]
#                 lm_loss = train_loss_fct(lm_logits.transpose(1,2), support_gen_y)
#                 qa_loss_mean = torch.mean(qa_loss)
#                 lm_loss_mean = torch.mean(lm_loss)
#                 loss = qa_loss_mean +  self.lm_lambda * lm_loss_mean
            
            # Update Optimizer
            inner_optimizer.backward(loss, update_master_grads=False) # instead of loss.backward() for fp16
            inner_optimizer.update_master_grads()
            inner_optimizer.clip_master_grads(self.max_grad_norm)
            inner_optimizer.step()
            inner_optimizer.zero_grad()
            
            all_loss.append(loss.item())

        ### END Adaptation Phase ###
        
        
        ### START Meta-Learning Phase ###
        # 4. After Adaptation, use the query set for learning
        # Somehow it also returns attentions in [1]?, this is selecting 0 of what WrapModel is doing 
        qa_logits = fast_model(query_x)[0]
        qa_loss = train_loss_fct(qa_logits.transpose(1,2), query_y)
        qa_loss_mean = torch.mean(qa_loss)
        loss = qa_loss_mean
        
        ##### LAMOL_SPECIFIC! This section adds the data (task, prev_task, model, train_extra_data, model_dir, tasks)
        if self.is_lamol:
            lm_logits = fast_model(query_gen_x)[0]
            lm_loss = train_loss_fct(lm_logits.transpose(1,2), query_gen_y)
            lm_loss_mean = torch.mean(lm_loss)
            loss = qa_loss_mean +  self.lm_lambda * lm_loss_mean
        

        for batch_i in range(qa_logits.size()[0]):
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
            
        # https://github.com/mailong25/meta-learning-bert/blob/master/maml.py
        self.meta_optimizer.backward(loss, update_master_grads=False) # instead of loss.backward() for fp16
        
        fast_model.to(torch.device('cpu'))
        for i, params in enumerate(fast_model.parameters()):
            sum_gradients.append(deepcopy(params.grad))
        del fast_model, inner_optimizer
        torch.cuda.empty_cache()
        
        #Assign gradient for original model, then using optimizer to update its weights
        for i, params in enumerate(self.net.parameters()):
            params.grad = sum_gradients[i].to(self.device)

        self.meta_optimizer.update_master_grads()
        self.meta_optimizer.clip_master_grads(self.max_grad_norm)
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()
        
        del sum_gradients
        gc.collect()
        
        return loss.item()