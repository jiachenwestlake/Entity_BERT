# coding: utf-8
'''
An entry or sent looks like ...

SOCCER NN B-NP O
- : O O
JAPAN NNP B-NP B-LOC
GET VB B-VP O
LUCKY NNP B-NP O
WIN NNP I-NP O
, , O O
CHINA NNP B-NP B-PER
IN IN B-PP O
SURPRISE DT B-NP O
DEFEAT NN I-NP O
. . O O

Each mini-batch returns the followings:
words: list of input sents. ["The 26-year-old ...", ...]
x: encoded input sents. [N, T]. int64.
is_heads: list of head markers. [[1, 1, 0, ...], [...]]
tags: list of tags.['O O B-MISC ...', '...']
y: encoded tags. [N, T]. int64
seqlens: list of seqlens. [45, 49, 10, 50, ...]
'''
import numpy as np
import torch
from torch.utils import data
import codecs


try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle


def match(s, p):
    """
    :param s: tokens: list
    :param p: pattern: list
    :return: start index : list
    """
    start = []

    if s == None or p == None:
        return start

    slen = len(s)
    plen = len(p)

    if slen < plen:
        return start

    i = j = 0
    while i < slen:
        if j >= plen:
            start.append(i - plen)
            i += 1
            j = 0
        elif s[i] == p[j]:
            i += 1
            j += 1
        else:
            i = i - j + 1
            j = 0
            if i > slen - plen:
                return start

    return start

def label_entity(tokens, entity_vocab):
    entity_label = [0] * len(tokens) # padding entity label with 0
    sent = "".join(tokens)
    for entity in entity_vocab.keys():
        if entity not in sent:
            continue
        start = match(tokens, list(entity))
        for en_start in start:
            en_end = en_start + len(entity)
            if entity_label[en_start] == 0 and entity_label[en_end-1] == 0:
                entity_label[en_start:en_end] = [entity_vocab[entity]] * (en_end-en_start)

    return entity_label

class NerLabel(data.Dataset):
    def __init__(self, fpath_list):
        self.VOCAB = []
        self.VOCAB.append('<PAD>')
        for fpath in fpath_list:
            entries = codecs.open(fpath, 'r', encoding='utf-8').read().strip().split("\n\n")
            for entry in entries:
                tags = ([line.split()[-1] for line in entry.splitlines()])
                for tag in tags:
                    if tag not in self.VOCAB:
                        self.VOCAB.append(tag)
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.VOCAB)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.VOCAB)}

        # print(self.tag2idx.keys())
    def save(self, file_name):
        f = open(file_name, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()


    def load(self, file_name):
        f = open(file_name, 'rb')
        temp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(temp_dict)

class NerDataset(data.Dataset):
    def __init__(self, fpath, ner_label, tokenizer, entity_dict):
        """
        fpath: [train|valid|test].txt
        """
        self.tokenizer = tokenizer
        entries = codecs.open(fpath, 'r', encoding= 'utf-8').read().strip().split("\n\n")
        sents, tags_li, titles = [], [], [] # list of lists
        self.ner_label = ner_label
        self.novel_vocab = entity_dict.novel_vocab #{title:{entity:id}}
        for entry in entries:
            title = entry.splitlines()[0][7:] # <title>XXXX
            words = [line.split()[0] for line in entry.splitlines()[1:]]
            tags = ([line.split()[-1] for line in entry.splitlines()[1:]])
            if len(words) <= 180:
                sents.append(["[CLS]"] + words + ["[SEP]"])
                tags_li.append(["<PAD>"] + tags + ["<PAD>"])
                titles.append(title)
        # list[list[]]
        self.sents, self.tags_li, self.titles = sents, tags_li, titles

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags, title = self.sents[idx], self.tags_li[idx], self.titles[idx] # words, tags: string list
        if title not in self.novel_vocab:
            print("".join(words))
        entity_vocab = self.novel_vocab[title]

        # We give credits only to the first piece.
        input_ids, input_tags, input_tokens = [], [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            tag_ids = [self.ner_label.tag2idx[each] for each in t]  # (T,)

            input_tokens.extend(tokens)
            input_ids.extend(token_ids)
            is_heads.extend(is_head)
            input_tags.extend(tag_ids)
        # print(title, words, len(words))
        entity_label = label_entity(input_tokens, entity_vocab)
        # assert len(input_ids)==len(input_tags)==len(is_heads)==len(entity_label), \
        #     f"len(x)={len(input_ids)}, len(y)={len(input_tags)}, len(is_heads)={len(is_heads)}, len(entity_label)={len(entity_label)}"
        if not len(input_ids) == len(input_tags) == len(is_heads):
            return "", [], [], "", [], [], 0
        # seqlen
        seqlen = len(input_tags)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, input_ids, is_heads, tags, input_tags, entity_label, seqlen


def pad(batch):
    '''Pads to the longest sample'''
    ## extract batch
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    ## padding the ids, tags with 0
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    pad_input_ids = f(1, maxlen) # input_ids
    pad_input_tags = f(-3, maxlen) # input_tags
    pad_entity_label = f(-2, maxlen) # entity_label

    f = torch.LongTensor

    return words, f(pad_input_ids), is_heads, tags, f(pad_input_tags), f(pad_entity_label), seqlens


