# Entity_BERT
Entity Enhanced BERT Pre-training for Chinese NER, code for EMNLP 2020 paper

A semi-supervised entity enhanced BERT pre-training method for Chinese NER. It is released with our EMNLP 2020 paper: [Entity Enhanced BERT Pre-training for Chinese NER](https://www.aclweb.org/anthology/2020.emnlp-main.518.pdf). This repo contains the code, model and data used in our paper. 


### Introduction
Character-level BERT pre-trained in Chinese suffers a limitation of lacking lexicon information, which shows effectiveness for Chinese NER. To integrate the lexicon into pre-trained LMs for Chinese NER, we investigate a semi-supervised entity enhanced BERT pre-training method. In particular, we first extract an entity lexicon from the relevant raw text using a new-word discovery method. We then integrate the entity information into BERT using Char-Entity-Transformer, which augments the self-attention using a combination of character and entity representations. In addition, an entity classification task helps inject the entity information into model parameters in pre-training. The pre-trained models are used for NER fine-tuning. Experiments on a news dataset and two datasets annotated by ourselves for NER in long-text show that our method is highly effective and achieves the best results.

### Requirements
```
Python 3.6 
PyTorch 1.0.0
```

### Datasets
 - The **Novel dataset** can be downloaded from [Google Drive]() or [Baidu Disk](https://pan.baidu.com/s/1GESFNbZ0r0m1TYVqaZttnA) with key `hcb7`
 - The **Finance dataset** can be downloaded from [Google Drive]() or [Baidu Disk](https://pan.baidu.com/s/10nmORk-OfQoSV2kT50jiYg) with key `5tmz`
 - The **News dataset** can be downloaded from [Google Drive]() or [Baidu Disk](https://pan.baidu.com/s/1E5ZySyTbz5yY_RofWKBAkQ) with key `6w0e`. The NER data is sampled from [CLUENER-2020](https://github.com/CLUEbenchmark/CLUENER2020) and the raw data is taken from [THUCNews](https://github.com/thunlp/THUCTC)


### Pretrained LM Weights and Models
 - The BERT-base weights can be downloaded from [Google Drive]() or [Baidu Disk](https://pan.baidu.com/s/1NAkhlpZp38zEWDn080ly0A) with key `3xic`
 - The pre-trained model by **our method** with **Novel dataset** can be downloaded from [Google Drive]() or [Baidu Disk](https://pan.baidu.com/s/1JQO70-XDlqrINAJnXzOrSA) with  key `ek4e`
 - The pre-trained model by **our method** with **Finance dataset** can be downloaded from [Google Drive]() or [Baidu Disk](https://pan.baidu.com/s/1PIBWCQX9qCQ0APGS_xmmxw) with key `eand`
 - The pre-trained model by **our method** with **News dataset** can be downloaded from [Google Drive]() or [Baidu Disk](https://pan.baidu.com/s/1472Zb90mvxF4Xz9GO5vxIw) with key `4oo0`


### Usage
#### Pre-training
  - Download the data from the above links based on your demands and put the download file under `document_level_ner\data` dictionary.
  - Download the BERT-base weights and put the download file under `document_level_ner` dictionary. 
  - Using the following command to run the pre-training code
  ```
  bash lm_pretrain.sh train
  ```
  The file `lm_pretrain.sh` contains dataset path and model hyperparameters.

#### Fine-tuning
  - Download the data from the above links based on your demands and put the download file under `document_level_ner\data` dictionary.
  - Download our pre-trained BERT weights(Novel, Finance or News) and put the download file under `document_level_ner` dictionary. Or you can use your own pre-trained weights and put the file under `document_level_ner` dictionary.
  - Using the following command to run the code
  ```
  cd bert_ner-master
  bash run.sh finetune
  ```
  The file `run.sh` contains dataset path and model hyperparameters.


  


### Citation
When you use the our paper or dataset, we would appreciate it if you cite the following:
```
@inproceedings{jia2020entity,
  title={Entity Enhanced BERT Pre-training for Chinese NER},
  author={Jia, Chen and Shi, Yuefeng and Yang, Qinrong and Zhang, Yue},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={6384--6396},
  publisher="Association for Computational Linguistics",
  year={2020}
}
```
