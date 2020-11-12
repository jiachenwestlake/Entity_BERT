# coding: utf-8

import logging
import time
import math
import os, sys
import random
import itertools
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb

from utils.tokenization import BertTokenizer
from utils.optimization import BertAdam, WarmupLinearSchedule
from data.data_loader import load_lm_data
from modeling_no_entity import BertForPreTraining

from utils.data_parallel import BalancedDataParallel

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Domain-specific Transformer Language Model')

########################################################################################################################
# Prepare Hyper-Parameters and Settings
########################################################################################################################
### Required parameters
parser.add_argument('--transformer_mode', default='transformer_base', type=str,
                    help='transformer models \'transformer_base\', \'transformer_XL\'')
parser.add_argument('--output_dir', default='LM_pretrained', type=str,
                    help='experiment directory.')
parser.add_argument('--data', type=str, default='../data/train_pro.txt',
                    help='location of the data corpus')
parser.add_argument('--entity_dict', type=str, default='../data/entity_dict',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='lm_raw_data',
                    choices=['lm_raw_data_novel', 'lm_raw_data_finance', 'lm_raw_data_novel_half', 'lm_raw_data_novel_open', 'lm_raw_data_novel_ngram', 'lm_raw_data_thuner'],
                    help='dataset type')
parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                    help='pretrained bert model')
parser.add_argument("--do_lower_case", action='store_true',
                    help="Whether to lower case the input text. \n"
                         "True for uncased models, False for cased models.")

### Training params
##  Training scale
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument("--max_seq_length", type=int, default=128,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--max_cls_mems", type=int, default=128,
                    help="The maximum total number of sentsences to be attended during training. \n" )
parser.add_argument("--max_doc_length", type=int, default=128,
                    help="The maximum total input sentsence number of one document. \n"
                         "Documents longer than this will be truncated. \n")
parser.add_argument('--memory_length', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--max_step', type=int, default=100000,
                    help='upper step limit')
parser.add_argument("--num_train_epochs",default=3.0, type=float,
                    help="Total number of training epochs to perform.")
## Algorithms setting
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same positional embeddings after clamp_len')
parser.add_argument('--sample_softmax', type=int, default=-1,
                    help='number of samples in sampled softmax')
## Optimization setting
parser.add_argument('--learn_rate', type=float, default=3e-5,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument("--warmup_proportion",default=0.1,type=float,
                    help="Proportion of training to perform linear learning rate warmup for. ")
parser.add_argument('--decay_rate', type=float, default=0.01,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--loss_scale', type=float, default=0,
                    help='Static loss scale, positive power of 2 values can '
                    'improve fp16 convergence.')
## Hardware and Other setting
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--fp16', action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')

args = parser.parse_args()


# Set the random seed manually for reproducibility (Optional).
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logger.info('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Validate `--fp16` option
if args.fp16:
    if not args.cuda:
        logger.info('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
        args.fp16 = False
    else:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            logger.info('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

# object of <class 'torch.device'>
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
n_gpu = torch.cuda.device_count()

########################################################################################################################
# Load Data
########################################################################################################################
## using the Bert Word Vocabulary
tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
vocab_size = len(tokenizer.vocab)
corpus = load_lm_data(args.entity_dict, args.data, args.output_dir, args.dataset, tokenizer)
## Training Dataset
train_iter = corpus.get_iterator('train', args.batch_size, args.max_seq_length, args.max_doc_length, device=device)

## total batch numbers and optim updating steps
total_train_steps = int(train_iter.batch_steps * args.num_train_epochs)

########################################################################################################################
# Building the model
########################################################################################################################
model = BertForPreTraining.from_pretrained(args.bert_model, entity_num=train_iter.entity_num)

args.n_all_param = sum([p.nelement() for p in model.bert.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.bert.encoder.parameters()])

logger.info('=' * 100)
for k, v in args.__dict__.items():
    logger.info('    - {} : {}'.format(k, v))
logger.info('=' * 100)
logger.info('#params = {}'.format(args.n_all_param))
logger.info('#non emb params = {}'.format(args.n_nonemb_param))

if args.fp16:
    model = model.half()

#n_gpu = 1
if n_gpu > 1:
    device_ids = [0, 1]
    model = nn.DataParallel(model.to(device), device_ids=device_ids)
else:
    model = model.to(device)

########################################################################################################################
# prepare optimizer
########################################################################################################################

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.decay_rate}, # no no_decay params in p
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} # any one of no_decay in p
]
if args.fp16:
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=args.learn_rate,
                          bias_correction=False,
                          max_grad_norm=1.0)
    if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion, t_total=total_train_steps)
else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learn_rate,
                         warmup=args.warmup_proportion,
                         t_total=total_train_steps)


########################################################################################################################
# Training
########################################################################################################################

logger.info("** ** ** Running Training ** ** **")
logger.info(" Num documents = %d", train_iter.num_docs)
logger.info(" Num sentences = %d", train_iter.batch_steps * args.batch_size)
logger.info(" Num steps = %d ", total_train_steps)
# train
def train():
    global train_step, train_loss, train_entity_pred, train_entity_right, log_start_time, \
           step_loss, step_entity_pred, step_entity_right
    model.train()
    model.zero_grad()
    mask_mems = torch.tensor([[]]) # (batch_size, seq_len) use for transformer-xl
    doc_id_mems = torch.tensor([]) # (batch_size) check sentence sequence
    hidden_mems = tuple()
    ## for transformer-Sent length of cls mems
    cls_len = torch.zeros(args.batch_size, device=device).long()  # LongTensor(batch_size)

    lm_train_iter = train_iter.get_iter()
    # step_loss =0
    for batch_idx,  batch_list in enumerate(lm_train_iter):
        batch_list = tuple(t.to(device) for t in batch_list)
        novel_id, doc_id, input_ids, org_ids, input_mask, lm_label, entity_label = batch_list

        if doc_id_mems.size(0) != 0:
            same_doc = doc_id_mems == doc_id # tensor([1/0, 1/0, ...], dtype=Byte) size=batch_size
        else:
            same_doc = None
        mask_mems = mask_mems if mask_mems.size(1) != 0 else None
        if same_doc is not None:
            cls_len += same_doc.long() # same doc + 1
            cls_len = cls_len.masked_fill(1 - same_doc, 0).type_as(input_ids) # not same doc restart
            cls_len.clamp_(max=args.max_cls_mems) # contrain the maximum cls mems numbers

        loss_gpus, entity_pred_gpus, entity_right_gpus = model(input_ids, attention_mask=input_mask, masked_lm_labels=lm_label, entity_label=entity_label)
        # Tensor()
        loss = loss_gpus.float().mean().type_as(loss_gpus)
        entity_pred = entity_pred_gpus.sum()
        entity_right = entity_right_gpus.sum()

        if loss.float().item() > 1e8:
            logger.warning('Training loss is too large: %.4f' % (train_loss))
            logger.info('Step %d gradiant expload!, exit!' % (train_step))
            exit()

        if not np.any(np.isnan(loss.detach().cpu().numpy())) and loss.item() != 0:
            train_loss += loss.float().item()
            train_entity_pred += entity_pred.float().item()
            train_entity_right += entity_right.float().item()
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
        elif np.any(np.isnan(loss.detach().cpu().numpy())):
                print(input_ids)
                print(lm_label)
                print(same_doc)
                print(loss_gpus)
                logger.info('Training Loss is nan ones time !')

        mask_mems = input_mask
        doc_id_mems = doc_id

        if args.fp16:
            lr_this_step = args.learn_rate * warmup_linear.get_lr(train_step, args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step

        optimizer.step()
        optimizer.zero_grad()

        train_step += 1

        if train_step % 100 == 0:
            logger.info('Num of trianing step: %d, steps loss: %.4f, steps entity right rate: %.4f' %
                        (train_step, train_loss-step_loss, (train_entity_right-step_entity_right)/(train_entity_pred-step_entity_pred)))
            step_loss = train_loss
            step_entity_pred = train_entity_pred
            step_entity_right = train_entity_right

########################################################################################################################
# Save models
########################################################################################################################

def save_model(epoch_num):
    ##save the whole model, for restart
    with open(os.path.join(args.output_dir, 'model.pt'), 'wb') as f:
        torch.save(model, f)
    with open(os.path.join(args.output_dir, 'optimizer.pt'), 'wb') as f:
        torch.save(optimizer.state_dict(), f)

    ## save the learned parameters
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, 'epoch' + str(epoch_num) + '_' + WEIGHTS_NAME)
    # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)

########################################################################################################################
# Loop over epochs
########################################################################################################################
#pdb.set_trace()
train_step = 0
train_loss = 0
step_loss = 0

train_entity_pred = 0
train_entity_right = 0
step_entity_pred = 0
step_entity_right = 0

log_start_time = time.time()
epoch_time = time.time()
epoch_iter = 0
epoch_loss = 0
#print (torch.cuda.is_available())
try:
    # for epoch in itertools.count(start=1):
    while epoch_iter < args.num_train_epochs:
        train()
        logger.info("Epoch %d is finished, steps: %d, time: %s s" % (epoch_iter, train_step + 1, time.time() - epoch_time))
        logger.info('Epoch loss: %.4f' % (train_loss - epoch_loss))
        epoch_loss = train_loss
        save_model(epoch_iter)
        epoch_iter += 1
        epoch_time = time.time()
        if train_step >= args.max_step:
            break
    logger.info('*' * 100)
    logger.info('End of Training')
    logger.info('train time: %s s' % (time.time() - log_start_time))
    logger.info('Total loss is: %.4f' % (train_loss))
    logger.info('Total entity prediction rate: %.4f' % (train_entity_right/train_entity_pred))

except KeyboardInterrupt:
    logger.info('*'*100)
    logger.info('Exiting from training early')
    logger.info('train time: %s s' % (time.time() - log_start_time))
    logger.info('Total loss is: %.4f' % (train_loss))
    logger.info('Total entity prediction rate: %.4f' % (train_entity_right/train_entity_pred))

########################################################################################################################
# save model
########################################################################################################################

logger.info("*** *** Save Model *** ***")
save_model(-1) ## interrupt or final model with epoch number = -1






