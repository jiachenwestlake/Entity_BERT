import torch
from torch import nn
import numpy as np

class LogUniformSampler(nn.Module):

    def __init__(self, n_sample, embeddings, word_frequency=False):
        super(LogUniformSampler, self).__init__()
        """
        Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))

        Our implementation fixes num_tries at 2 * n_sample, and the actual #samples will vary from run to run
        """
        # self.embeddings = embeddings
        self.vocab_size = embeddings.weight.size(0)
        self.hidden_dim = embeddings.weight.size(1)
        self.embeddings = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.embeddings.weight = embeddings.weight
        self.bias = nn.Parameter(torch.zeros(self.vocab_size))
        self.word_frequency = word_frequency
        with torch.no_grad():
            self.range_max = self.vocab_size
            log_indices = torch.arange(1., self.range_max+2., 1.).log_()
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
            # print('P', self.dist.numpy().tolist()[-30:])

            self.log_q = (- (-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()

        self.n_sample = n_sample

    def sample(self, labels):
        """
            labels: [token_len]
        Return
            true_log_probs: [token_len]
            samp_log_probs: [n_sample]
            neg_samples: [n_sample]
        """

        # neg_samples = torch.empty(0).long()
        n_sample = self.n_sample
        n_tries = 2 * n_sample

        with torch.no_grad():
            neg_samples = torch.multinomial(self.dist, n_tries, replacement=True).unique()
            device = labels.device
            neg_samples = neg_samples.to(device)
            true_log_probs = self.log_q[labels].to(device)
            samp_log_probs = self.log_q[neg_samples].to(device)
            return true_log_probs, samp_log_probs, neg_samples

    def forward(self, labels, inputs):
        """
            embedding: an nn.Embedding layer
            bias: [vocab_size]
            labels: [token_len]
            inputs: [token_len, hidden_dim]
            sampler: you may use a LogUniformSampler
        Return
            logits: [token_len, 1 + n_sample]
        """
        true_log_probs, samp_log_probs, neg_samples = self.sample(labels)
        n_sample = neg_samples.size(0)
        token_len = labels.size(0)
        all_ids = torch.cat([labels.view(-1), neg_samples])
        all_w = self.embeddings(all_ids)
        # (token_len, hidden_dim)
        true_w = all_w[: -n_sample].view(token_len, -1)
        # (n_sample, hidden_dim)
        sample_w = all_w[-n_sample:].view(n_sample, -1)

        all_b = self.bias[all_ids]
        true_b = all_b[: -n_sample].view(token_len)
        sample_b = all_b[-n_sample:]

        hit = (labels[:, None] == neg_samples).detach()
        # (token_len)
        true_logits = torch.einsum('ik,ik->i',
            [true_w, inputs]) + true_b # - true_log_probs
        if self.word_frequency:
            true_logits -= true_log_probs
        # (token_len, n_sample)
        sample_logits = torch.einsum('lk,ik->il',
            [sample_w, inputs]) + sample_b # - samp_log_probs
        if self.word_frequency:
            sample_logits -= samp_log_probs
        sample_logits.masked_fill_(hit, -1e30)
        logits = torch.cat([true_logits[:, None], sample_logits], -1)

        return logits # (token_len, 1 + n_sample)

if __name__ == '__main__':
    S, B = 3, 4
    n_vocab = 10000
    n_sample = 5
    H = 32 # hidden_dim = emb_dim
    embedding = nn.Embedding(n_vocab, H)
    labels = torch.LongTensor(S).random_(0, n_vocab)
    # labels[0,:] = -1 # masked LM pretrain
    sampler = LogUniformSampler(n_vocab, n_sample, embedding)

    bias = torch.zeros(n_vocab)
    inputs = torch.Tensor(S, H).normal_()

    logits = sampler(bias, labels, inputs)
    print('logits', logits.detach().numpy().tolist())
    print('logits shape', logits.size())
    # print('out_labels', out_labels.detach().numpy().tolist())
    # print('out_labels shape', out_labels.size())

