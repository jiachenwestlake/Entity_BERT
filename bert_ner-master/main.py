import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from data_load import NerLabel, NerDataset, pad
from metrics import f1_score
from metrics import classification_report
import os
import numpy as np
import argparse

# from pytorch_pretrained_bert import BertTokenizer
import sys
sys.path.append('../')
from utils.tokenization import BertTokenizer
from data.data_loader import EntityDict

def train_epoch(model, iterator, optimizer, criterion, tokenizer):
    model.train()
    for i, batch in enumerate(iterator):
        words, input_ids, is_heads, tags, input_tags, entity_label, seqlens = batch
        _input_tags = input_tags # for monitoring
        optimizer.zero_grad()
        logits, input_tags, _ = model(input_ids, input_tags, entity_label) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        input_tags = input_tags.view(-1)  # (N*T,)

        loss = criterion(logits, input_tags)
        loss.backward()

        optimizer.step()

        if i == 0:
            print("=====sanity check======")
            print("words:", words[0])
            print("x:", input_ids.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("y:", _input_tags.cpu().numpy()[0][:seqlens[0]])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("=======================")


        if i % 10 == 0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")

def eval(model, iterator, f, ner_label):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open("temp", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [ner_label.idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true = np.array([ner_label.tag2idx[line.split()[1]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])
    y_pred = np.array([ner_label.tag2idx[line.split()[2]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    final = f + ".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
    with open(final, 'w') as fout:
        result = open("temp", "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")

    os.remove("temp")

    print("precision=%.2f"%precision)
    print("recall=%.2f"%recall)
    print("f1=%.2f"%f1)
    return precision, recall, f1

def evaluate(model, iterator, f, ner_label, verbose = False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    y_true = []
    y_pred = []
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, input_ids, is_heads, tags, input_tags, entity_label, seqlens = batch

            _, _, y_hat = model(input_ids, input_tags, entity_label)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(input_tags.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
    ## gets results and save
    with open("temp", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [ner_label.idx2tag[hat] for hat in y_hat]
            if len(preds[1:-1]) > 0:
                y_pred.append(preds[1:-1])
            if len(tags.split()[1:-1]) > 0:
                y_true.append(tags.split()[1:-1])
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    assert len(y_pred) == len(y_true)

    # logging loss, f1 and report
    p, r, f1 = f1_score(y_true, y_pred)

    # metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    # logging.info("- {} metrics: ".format(mark) + metrics_str)
    #
    # if verbose:
    #     report = classification_report(true_tags, pred_tags)
    #     logging.info(report)

    final = f + ".P%.4f_R%.4f_F%.4f" %(p, r, f1)
    with open(final, 'w') as fout:
        result = open("temp", "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={p}\n")
        fout.write(f"recall={r}\n")
        fout.write(f"f1={f1}\n")
        if verbose:
            report = classification_report(y_true, y_pred)
            print(report)

    os.remove("temp")

    print("precision=%.2f"%p)
    print("recall=%.2f"%r)
    print("f1=%.2f"%f1)
    return p, r, f1


def train(hp):
    tokenizer = BertTokenizer.from_pretrained(hp.bert_model_dir, do_lower_case=False)
    if hp.dataset == "lm_raw_data_finance":
        dict_file = "../data/dataset_finance/raw_data/annual_report_entity_list"
    elif hp.dataset == "lm_raw_data_novel":
        dict_file = "../data/dataset_book9/raw_data/entity_book9"
    elif hp.dataset == "lm_raw_data_thuner":
        dict_file = "../data/dataset_thuner/raw_data/thu_entity.txt"
    entity_dict = EntityDict(hp.dataset, dict_file)
    entity_dict.load(os.path.join(hp.bert_model_dir, 'entity.dict'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ner_label = NerLabel([hp.trainset, hp.validset])
    fname = os.path.join(hp.logdir, 'dict.pt')
    ner_label.save(fname)

    train_dataset = NerDataset(hp.trainset, ner_label, tokenizer, entity_dict)
    eval_dataset = NerDataset(hp.validset, ner_label, tokenizer, entity_dict)
    test_dataset = NerDataset(hp.testset, ner_label, tokenizer, entity_dict)

    model = Net(hp.bert_model_dir, hp.top_rnns, len(ner_label.VOCAB), entity_dict.entity_num, device, hp.finetuning).to(device)
    device_ids = [0, 1]
    model = nn.DataParallel(model, device_ids=device_ids)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    ## train the model
    best_eval = -10
    for epoch in range(1, hp.n_epochs + 1):
        train_epoch(model, train_iter, optimizer, criterion, tokenizer)

        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
        fname = os.path.join(hp.logdir, 'model')
        precision, recall, f1 = evaluate(model, eval_iter, fname, ner_label, verbose=False)

        if f1 > best_eval:
            best_eval = f1
            print("epoch{} get the best eval f-score:{}".format(epoch, best_eval))
            torch.save(model.state_dict(), f"{fname}.pt")
            print(f"weights were saved to {fname}.pt")

        print(f"=========test at epoch={epoch}=========")
        if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
        fname = os.path.join(hp.logdir, str(epoch))
        precision, recall, f1 = evaluate(model, test_iter, fname, ner_label, verbose=False)

def decode(hp):
    tokenizer = BertTokenizer.from_pretrained(hp.bert_model_dir, do_lower_case=False)
    if hp.dataset == "lm_raw_data_finance":
        dict_file = "../data/dataset_finance/annual_report_entity_list"
    elif hp.dataset == "lm_raw_data_novel":
        dict_file = "../data/dataset_book9/entity_book9"
    entity_dict = EntityDict(hp.dataset, dict_file)
    entity_dict.load(os.path.join(hp.bert_model_dir, 'entity.dict'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ner_label = NerLabel([hp.decodeset])
    if os.path.exists(os.path.join(hp.logdir, 'dict.pt')):
        ner_label.load(os.path.join(hp.logdir, 'dict.pt'))
    else:
        print('dict.pt is not exit')
        exit()

    decode_dataset = NerDataset(hp.decodeset, ner_label, tokenizer, entity_dict)

    model = Net(hp.bert_model_dir, hp.top_rnns, len(ner_label.VOCAB), entity_dict.entity_num, device, hp.finetuning).to(device)
    model = nn.DataParallel(model)
    ## Load the model parameters
    if os.path.exists(os.path.join(hp.logdir, 'model.pt')):
        model.load_state_dict(torch.load(os.path.join(hp.logdir, 'model.pt')))
    else:
        print("the pretrianed model path does not exist! ")
        exit()

    decode_iter = data.DataLoader(dataset=decode_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)

    fname = os.path.join(hp.logdir, '_')

    precision, recall, f1 = evaluate(model, decode_iter, fname, ner_label, verbose=False)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--form", type=str, default='train', help="train or decode")
    parser.add_argument("--bert_model_dir", type=str, default='bert-base-chinese',
                        help="bert-base-chinese for chinese or bert-model-cased for english or others of your pretrained model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help='random seed')
    parser.add_argument("--logdir", type=str, default="checkpoints/01")
    parser.add_argument("--dataset", type=str, default="finance")
    parser.add_argument("--trainset", type=str, default="../data/dataset_book9/book9_train_title_BIO.txt", help="../data/dataset_finance/MSRA_train_dev_title.char.bmes")
    parser.add_argument("--validset", type=str, default="../data/dataset_book9/book9_evaluation_title_BIO.txt", help="../data/dataset_finance/annual_report_anno_part1&2_title.txt")
    parser.add_argument("--testset", type=str, default="../data/dataset_book9/book9_test_title_BIO.txt", help="")
    parser.add_argument("--decodeset", type=str, default="../data/dataset_finance/annual_report_anno_part1&2_title.txt", help="")

    hp = parser.parse_args()

    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)

    if hp.form == 'train':
        train(hp)
    elif hp.form == 'decode':
        decode(hp)
    else:
        print('form is invalid')



