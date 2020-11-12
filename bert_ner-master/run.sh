#!/bin/bash
## finance dataset
# --dataset lm_raw_data_finance
# --bert_model_dir ../LM_pretrained_finance
# --trainset ../data/dataset_finance/ner/MSRA_train_dev_title.char.bmes
# --validset ../data/dataset_finance/ner/annual_report_anno_part1\&2_title.txt
# --testset ../data/dataset_finance/ner/test_finance_all_title.txt
# --logdir checkpoints/finetuning/finance

## novel dataset
# --dataset lm_raw_data_novel
# --bert_model_dir ../LM_pretrained_book9
# --trainset ../data/dataset_book9/ner/book9_6500_train_out.txt
# --validset ../data/dataset_book9/ner/book9_evaluation_title_BIO.txt
# --testset ../data/dataset_book9/ner/book9_test_title_BIO.txt
# --logdir checkpoints/finetuning/novel

## news dataset
# --dataset lm_raw_data_thuner
# --bert_model_dir ../LM_pretrained_thuner
# --trainset ../data/dataset_thuner/ner/train_thuner.txt
# --validset ../data/dataset_thuner/ner/dev_thuner.txt
# --testset ../data/dataset_thuner/ner/test_thuner.txt
# --logdir checkpoints/finetuning/news

if [[ $1 == 'finetune' ]]; then
    echo 'Run finetuning...'
    python main.py \
        --form train \
        --bert_model_dir ../LM_pretrained_finance \
        --logdir checkpoints/finetuning/finance \
        --dataset lm_raw_data_finance \
        --trainset ../data/dataset_finance/ner/MSRA_train_dev_title.char.bmes \
        --validset ../data/dataset_finance/ner/annual_report_anno_part1\&2_title.txt \
        --testset ../data/dataset_finance/ner/test_finance_all_title.txt \
        --finetuning \
        --batch_size 32 \
        --lr 5e-5 \
        --n_epochs 10 \
        ${@:2}
elif [[ $1 == 'feature' ]]; then
    echo 'Run finetuning with feature...'
    python main.py \
        --form train \
        --logdir checkpoints/feature \
        --batch_size 128 \
        --top_rnns \
        --lr 1e-4 \
        --n_epochs 30 \
        ${@:2}
elif [[ $1 == 'decode' ]]; then
    echo 'Run decoding...'
    python main.py \
        --form decode \
        --dataset lm_raw_data_finance \
        --bert_model_dir ../LM_pretrained_finance \
        --logdir checkpoints/finetuning/finance \
        --batch_size 32 \
        ${@:2}
else
    echo 'unknown argment 1'
fi

