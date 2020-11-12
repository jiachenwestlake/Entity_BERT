import os
import random
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm, trange

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityDict():
    ## entity load from entity dictionary
    def __init__(self, dataset, dict_path, title_vocab=None, encoding="utf-8", dict_lines=None):
        self.entity_num = 0
        self.novel_vocab_all = {} # all novels in the dict_file {novel_title: entity_list}
        self.novel_vocab = {} # novels appear in the dataset {novel_title: entity_vocab:{entity:entity_id}}
        line_num = 1
        entity_list = []
        if dataset in ["lm_raw_data_novel" , "lm_raw_data_novel_half"]:
            with open(dict_path, 'r', encoding=encoding) as f:
                for line in tqdm(f, desc="Loading Entity Dict", total=dict_lines):
                    line = line.strip()
                    if line_num % 14 == 0: # novel end
                        self.novel_vocab_all[title] = entity_list
                        entity_list = []
                    elif line_num % 14 == 1: # title
                        title = line
                    elif (line_num % 14) in [3, 5, 7, 9, 11, 13]:
                        entities = line.split()
                        #print (entities)
                        entity_list.extend(entities)
                    line_num += 1
        elif dataset in [ "lm_raw_data_finance" , "lm_raw_data_novel_open" , "lm_raw_data_novel_ngram", "lm_raw_data_thuner"]:
            with open(dict_path, 'r', encoding=encoding) as f:
                for line in tqdm(f, desc="Loading Entity Dict", total=dict_lines):
                    line = line.strip()
                    if line_num % 3 == 0: # annual report end
                        self.novel_vocab_all[title] = entity_list
                        entity_list = []
                    elif line_num % 3 == 1: # title
                        title = line
                    else:
                        entities = line.split()
                        entity_list.extend(entities)
                    line_num += 1
        #print(self.novel_vocab_all.keys())
        if title_vocab is not None:
            self.update_novel_vocab(title_vocab)

    def update_novel_vocab(self, titlevocab):
        entity_list_all = [] ## all of the entities appearing in raw data
        entity_list_all.append('[PAD]') # padding index 0
        for title in titlevocab:
            entity_vocab = {}
            if title in self.novel_vocab_all:
                entity_list = self.novel_vocab_all[title]
                for entity in entity_list:
                    entity_vocab[entity] = len(entity_list_all)
                    entity_list_all.append(entity)
            self.novel_vocab[title] = entity_vocab
        self.entity_num = len(entity_list_all)

    def save(self, file_name):
        fout = open(file_name, 'wb')
        pickle.dump(self.__dict__, fout, 2)
        fout.close()

    def load(self, file_name):
        fin = open(file_name, 'rb')
        temp_dict = pickle.load(fin)
        fin.close()
        self.__dict__.update(temp_dict)


## database load from the training file
class DatabaseIterator(Dataset):
    def __init__(self, dataset, dict_path, corpus_path, output_path, tokenizer, batch_size, seq_len, doc_len, encoding="utf-8", device='cpu', corpus_lines=None):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.doc_len = doc_len
        self.max_sent_mask = 10
        self.corpus_lines = corpus_lines  # number of non-empty lines(sents) in input corpus
        self.dataset = dataset
        self.dict_path = dict_path
        self.corpus_path = corpus_path
        self.output_path = output_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc
        self.device = device
        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        # for loading samples in memory
        self.titlevocab = [] # novel title list
        self.num_docs = 0
        self.sample_to_doc = [] # [sent#1[novel_id, doc_id, line_id_in_doc], ...]

        self.all_docs = [] # number of documents [doc#1:[sent#1,sent#2, ...], ...]

        # load samples into memory
        doc = []
        line_num = 0 # line num in one doc

        self.corpus_lines = 0
        with open(corpus_path, "r", encoding=self.encoding) as f:
            for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                line = line.strip()
                if line == "" or "<title>" in line:
                    self.all_docs.append(doc)
                    doc = []
                    line_num = 0
                    if "<title>" in line:
                        self.titlevocab.append(line[7:])
                else:
                    self.corpus_lines = self.corpus_lines + 1
                    # split large doc by max_len
                    if line_num >= self.doc_len:
                        self.all_docs.append(doc)
                        doc = []
                        line_num = 0
                    ## one sent as sample
                    sample = [len(self.titlevocab)-1, len(self.all_docs), len(doc)] # [novel_id, doc_id, line_id]
                    self.sample_to_doc.append(sample)
                    doc.append(line)
                    line_num = line_num + 1

        self.title_num = len(self.titlevocab)
        self.title2idx = {title: idx for idx, title in enumerate(self.titlevocab)}
        self.idx2title = {idx: title for idx, title in enumerate(self.titlevocab)}

        # entity vocab {{titel:{entity:emb_idx}}}
        entity_dict = EntityDict(self.dataset, self.dict_path, self.titlevocab)
        entity_dict.save(os.path.join(self.output_path, 'entity.dict'))
        self.novel_vocab = entity_dict.novel_vocab
        self.entity_num = entity_dict.entity_num
        # if last row in file is not empty
        if self.all_docs[-1] != doc:
            self.all_docs.append(doc)

        self.num_docs = len(self.all_docs)

        while len(self.sample_to_doc) % self.batch_size != 0:
            self.sample_to_doc.pop()

        # (sample_num//batch_size, batch_size, 3)
        self.sample_to_doc_tensor = torch.LongTensor(self.sample_to_doc).view(self.batch_size, -1, 3).permute(1, 0, 2).contiguous().to(self.device)
        self.batch_steps = self.sample_to_doc_tensor.size(0)


    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.batch_steps

    def get_batch(self, sent_iter):
        """
        :param sent_iter: the sent iterator of sample_to_doc
        :return: batch of data for train [input_ids, input_mask, lm_label_ids, doc_id]
                 input_ids:Tensor(seq_len, batch_size)
                 doc_id: document id
        """
        assert sent_iter < self.sample_to_doc_tensor.size(0)
        cur_id = self.sample_counter
        self.sample_counter += 1
        #(batch_size, 2)
        batch_data = self.sample_to_doc_tensor[sent_iter]
        batch_doc_idx = []
        batch_sents = []
        max_seq_len_ = -1
        for in_batch_idx in range(self.batch_size): # for each sent in the batch
            doc_idx = batch_data[in_batch_idx][1].item() # doc id
            batch_doc_idx.append(doc_idx)
            line_idx = batch_data[in_batch_idx][2].item() # sent id in docf
            sent_raw = self.all_docs[doc_idx][line_idx]
            tokens = self.tokenizer.tokenize(sent_raw) # tokenize str into token list
            batch_sents.append(tokens)
            max_seq_len_ = max(len(tokens), max_seq_len_)
        batch_novel_idx = []
        batch_input_ids = []
        batch_org_ids = []
        batch_input_mask = []
        batch_lm_label_ids = []
        batch_entity_label_ids = []
        ## +2 for [CLS] and [SEP]
        max_seq_len = min(max_seq_len_ + 2, self.seq_len)
        for in_batch_idx in range(self.batch_size):
            novel_idx = batch_data[in_batch_idx][0].item() # novel id
            # entity in one novel {entity: entity_idx}
            entity_vocab = self.novel_vocab[self.idx2title[novel_idx]]
            batch_novel_idx.append(novel_idx)
            tokens = batch_sents[in_batch_idx]
            cur_sample = InputExample(guid=cur_id, tokens=tokens)
            cur_features = convert_example_to_features(cur_sample, max_seq_len, self.tokenizer, self.max_sent_mask, entity_vocab)
            batch_input_ids.append(cur_features.input_ids)
            batch_org_ids.append(cur_features.org_sent_ids)
            batch_input_mask.append(cur_features.input_mask)
            batch_lm_label_ids.append(cur_features.lm_label_ids)
            batch_entity_label_ids.append(cur_features.entity_label_ids)
        # batch_doc_idx: (batch_size)
        # (batch_size, seq_len)
        return [torch.tensor(batch_novel_idx).to(self.device), #LongTensor(batch_size) novel id of each sent
                torch.tensor(batch_doc_idx).to(self.device), # LongTensor(batch_size) document id of each sent
                torch.tensor(batch_input_ids).to(self.device), # LongTensor(batch_size, seq_len) vocab idx of each token
                torch.tensor(batch_org_ids).to(self.device), # LongTensor(batch_size, seq_len) vocab idx of each token (no [mask])
                torch.tensor(batch_input_mask).to(self.device), # LongTensor(batch_size, seq_len) mask label of the sent
                torch.tensor(batch_lm_label_ids).to(self.device), # LongTensor(batch_size, seq_len) vocab idx of each token in [mask]
                torch.tensor(batch_entity_label_ids).to(self.device)]

    def get_iter(self, start=0):
        for sent_iter in range(start, self.sample_to_doc_tensor.size(0) - 1, 1):
            yield self.get_batch(sent_iter)


    def __iter__(self):
        self.get_iter()


def random_word(tokens, tokenizer, max_sent_mask):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []
    mask_idx = 0
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and mask_idx < max_sent_mask:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] instead".format(token))
            mask_idx += 1
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label

def label_entity(tokens, entity_vocab):
    entity_label = [0] * len(tokens) # padding entity label with 0
    sent = "".join(tokens)
    for entity in entity_vocab.keys():
        if entity not in sent:
            continue
        start = match(tokens, list(entity)) # list [start index]
        for en_start in start:
            en_end = en_start + len(entity)
            if entity_label[en_start] == 0 and entity_label[en_end-1] == 0:
                entity_label[en_start:en_end] = [entity_vocab[entity]] * (en_end-en_start)

    return entity_label

def _truncate_seq_pair(cur_sent, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        if len(cur_sent) <= max_length:
            break
        else: cur_sent.pop()


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

class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens, lm_labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens: string. The untokenized text of the doc.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens = tokens
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, org_sent_ids, input_mask, lm_label_ids, entity_label_ids):
        self.input_ids = input_ids
        self.org_sent_ids = org_sent_ids
        self.input_mask = input_mask
        self.lm_label_ids = lm_label_ids
        self.entity_label_ids = entity_label_ids

def convert_example_to_features(example, max_seq_length, tokenizer, max_sent_mask, entity_vocab):
    """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
    cur_sent = example.tokens
    # length is less than the specified length.
    # Account for [CLS] and [SEP]  with "- 2"
    _truncate_seq_pair(cur_sent, max_seq_length - 2)
    # entity label
    entity_label = label_entity(cur_sent, entity_vocab)
    # original sent
    org_sent = cur_sent
    org_sent_tokens = []
    org_sent_tokens.append("[CLS]")
    for token in org_sent:
        org_sent_tokens.append(token)
    org_sent_tokens.append("[SEP]")

    cur_sent, cur_sent_label = random_word(cur_sent, tokenizer, max_sent_mask)
    # concatenate lm labels and account for CLS
    sent_lm_label = ([-1] + cur_sent_label + [-1])
    entity_label = ([0] + entity_label + [0])
    # The convention in BERT is:
    #  For single sequences:
    #  tokens:   [CLS] the dog is hairy .
    # the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector".

    sent_tokens = []
    sent_tokens.append("[CLS]")
    for token in cur_sent:
        sent_tokens.append(token)
    sent_tokens.append("[SEP]")
    assert len(sent_tokens) == len(org_sent_tokens)
    sent_input_ids = tokenizer.convert_tokens_to_ids(sent_tokens)
    org_sent_ids = tokenizer.convert_tokens_to_ids(org_sent_tokens)
    sent_input_mask = [1] * len(sent_input_ids) # The mask has 1 for real tokens and 0 for padding tokens. Only real
    while len(sent_input_ids) < max_seq_length:
        sent_input_ids.append(0)
        org_sent_ids.append(0)
        sent_input_mask.append(0)
        sent_lm_label.append(-1)
        entity_label.append(0)

    assert len(sent_input_ids) == max_seq_length
    assert len(org_sent_ids) == max_seq_length
    assert len(sent_input_mask) == max_seq_length
    assert len(sent_lm_label) == max_seq_length
    assert len(entity_label) == max_seq_length

    # if example.guid < 5:
    #     logger.info("*** Example ***")
    #     logger.info("guid: %s" % (example.guid))
    #     logger.info("tokens: %s" % " ".join(
    #         [str(x) for x in sent_tokens]))
    #     logger.info("input_ids: %s" % " ".join([str(x) for x in sent_input_ids]))
    #     logger.info("org_sent_ids: %s" % " ".join([str(x) for x in org_sent_ids]))
    #     logger.info("input_mask: %s" % " ".join([str(x) for x in sent_input_mask]))
    #     logger.info("LM label: %s " % (sent_lm_label))
    #     logger.info("entity label: %s " % (entity_label))

    sent_features = InputFeatures(input_ids=sent_input_ids, # list: [word_id] * seq_len
                                  org_sent_ids=org_sent_ids, # list: [word_id] * seq_len
                                  input_mask=sent_input_mask, # list: [1/0] * seq_len
                                  lm_label_ids=sent_lm_label, #  list: [word_label] * seq_len
                                  entity_label_ids=entity_label # list: [entity_label] * seq_len
                                  )

    return sent_features



class Corpus(object):
    def __init__(self, dict_path, corpus_path, output_path, dataset, tokenizer):
        self.tokenizer = tokenizer
        self.dict_path = dict_path
        self.corpus_path = corpus_path
        self.output_path = output_path
        self.dataset = dataset

    def get_iterator(self, split, batch_size, seq_len, max_doc_len, device):
        if split == "train":
            if self.dataset == "lm_raw_data_novel" or self.dataset == "lm_raw_data_finance" or self.dataset == "lm_raw_data_novel_half" or self.dataset == "lm_raw_data_novel_open" or self.dataset == "lm_raw_data_novel_ngram" or self.dataset == "lm_raw_data_thuner":
                data_iter = DatabaseIterator(self.dataset, self.dict_path, self.corpus_path, self.output_path, self.tokenizer, batch_size, seq_len, max_doc_len, device=device)

        return data_iter



def load_lm_data(dictdir, datadir, outputdir, dataset, tokenizer):

    corpus = Corpus(dictdir, datadir, outputdir, dataset, tokenizer)

    return corpus


#if __name__ == '__main__':
#    from pytorch_pretrained_bert import BertTokenizer
#    dictdir = "entity_dict"
#    datadir = "book15_raw"
#    dataset = "lm_raw_data_novel"
#    outputdir = "LM_pretrained"
#    # entity_dict = EntityDict(dictdir)
#    # entity_dict.load(os.path.join(outputdir, 'entity.dict'))
#    tokenizer = BertTokenizer.from_pretrained("../bert_model", do_lower_case=False)
#    corpus = load_lm_data(dictdir, datadir, outputdir, dataset, tokenizer)
#    data_iter = corpus.get_iterator("train", 100, 128, 100, 'cuda')
#    lm_train_iter = data_iter.get_iter()
#    total_steps = len(data_iter)
#    print(total_steps)
#    for idx, batch in enumerate(lm_train_iter):
#        if idx % 100 == 0:
#            print('0k')
