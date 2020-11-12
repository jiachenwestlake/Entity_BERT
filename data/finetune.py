import os
import logging
import torch
import torch.nn as nn
import numpy as np
import argparse
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from doc_transformer import Doctransformer

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Finetuning Transformer Language Model on Target Tasks')

class TotalModel(nn.Module):
    def __init__(self, vocab_size, cutoffs, args):
        super(TotalModel, self).__init__()
        ## build the pretrained transformer model
        pretrianed_model = Doctransformer(args.transformer_mode, vocab_size, args.n_layer, args.n_head, args.hidden_dim,
                                          args.head_dim, args.d_inner, args.dropout, args.dropatt, args.emb_dim,
                                          args.div_val, args.pre_lnorm, args.max_seq_len, args.max_cls_mems,
                                          args.memory_length, cutoffs, args.clamp_len)

        ## Load the model parameters
        if os.path.exists(args.pretrained_model):
            pretrianed_model.load_state_dict(torch.load(args.pretrained_model))
        else:
            logger.warning("the pretrianed model path does not exist! ")
            exit()


    def forward(self):
        pass


def train(args, model, train_dataset):
    ##
    total_train_steps = len(train_dataset) / args.batch_size * args.num_train_epochs
    # prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.decay_rate},  # no no_decay params in p
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # any one of no_decay in p
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

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




def main():
    ### Required parameters
    parser.add_argument('--transformer_mode', default='transformer_base', type=str,
                        help='transformer models \'transformer_base\', \'transformer_XL\'')
    parser.add_argument('--output_dir', default='Finetune', type=str,
                        help='experiment directory.')
    parser.add_argument('--pretrained_model', type=str, default='LM_pretrained/pretrained_LM.model',
                        help='restart dir')
    parser.add_argument('--data_dir', type=str, default='../data/',
                        help='location of the dataset corpus')
    parser.add_argument('--dataset', type=str, default='fintune_data',
                        choices=['lm_raw_data', 'finetune_data'],
                        help='dataset type')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='pretrained bert model')
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to lower case the input text. \n"
                             "True for uncased models, False for cased models.")
    ### Model architecture params
    parser.add_argument('--n_layer', type=int, default=12,
                        help='number of total layers')
    parser.add_argument('--n_head', type=int, default=10,
                        help='number of heads')
    parser.add_argument('--head_dim', type=int, default=50,
                        help='head dimension')
    parser.add_argument('--emb_dim', type=int, default=-1,
                        help='wored embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=500,
                        help='model hidden dimension')
    parser.add_argument('--d_inner', type=int, default=1000,
                        help='inner dimension in FeedForward')
    ### Training params

    ##  Training scale
    parser.add_argument('--batch_size', type=int, default=60,
                        help='batch size')
    parser.add_argument('--batch_chunk', type=int, default=1,
                        help='split batch into chunks to save memory')
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_cls_mems", type=int, default=128,
                        help="The maximum total number of sentsences to be attended during training. \n")
    parser.add_argument("--max_doc_length", type=int, default=128,
                        help="The maximum total input sentsence number of one document. \n"
                             "Documents longer than this will be truncated. \n")
    parser.add_argument('--memory_length', type=int, default=0,
                        help='length of the retained previous heads')
    parser.add_argument('--max_step', type=int, default=100000,
                        help='upper step limit')
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    ## Algorithms setting
    parser.add_argument('--clamp_len', type=int, default=-1,
                        help='use the same positional embeddings after clamp_len')
    parser.add_argument('--adaptive', action='store_true',
                        help='use adaptive softmax')

    ## Optimization setting
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='global dropout rate')
    parser.add_argument('--dropatt', type=float, default=0.0,
                        help='attention probability dropout rate')
    parser.add_argument('--learn_rate', type=float, default=3e-5,
                        help='initial learning rate (0.00025|5 for adam|sgd)')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. ")
    parser.add_argument('--decay_rate', type=float, default=0.01,
                        help='decay factor when ReduceLROnPlateau is used')
    parser.add_argument('--loss_scale', type=float, default=0,
                        help='Static loss scale, positive power of 2 values can '
                             'improve fp16 convergence.')
    ## Hardware and Other setting
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='use multiple GPU')
    parser.add_argument('--gpu0_bsz', type=int, default=-1,
                        help='batch size on gpu 0')
    parser.add_argument('--div_val', type=int, default=1,
                        help='divident value for adapative input and softmax')
    parser.add_argument('--pre_lnorm', action='store_true',
                        help='apply LayerNorm to the input of multi-head attention and FeedForward instead of the output')
    parser.add_argument('--fp16', action='store_true',
                        help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')

    args = parser.parse_args()

    # Set the random seed manually for reproducibility (Optional).
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            logger.info('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    ## using the Bert Word Vocabulary
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_size = len(tokenizer.vocab)

    # adaptive softmax / embedding
    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        assert args.dataset in ['lm_raw_data']
        if args.dataset == 'lm_raw_data':
            cutoffs = [20000, 40000, 200000]
            tie_projs += [True] * len(cutoffs)
        else:
            cutoffs = [60000, 100000, 640000]
            tie_projs += [False] * len(cutoffs)

    ## build the transformer model
    total_model = TotalModel(vocab_size, cutoffs, args)

    total_model = total_model.to(device)







if __name__ == "__name__":
    main()
