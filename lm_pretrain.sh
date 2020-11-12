#!/bin/bash
## finance dataset
# --data data/dataset_finance/raw_data/annual_report_raw
# --entity_dict data/dataset_finance/raw_data/annual_report_entity_list
# -- dataset lm_raw_data_finance
# -- output_dir LM_pretrained_finance
## novel dataset
# -- data data/dataset_book9/raw_data/book9_raw
# -- entity_list data/dataset_book9/raw_data/entity_book9_all
# -- dataset lm_raw_data_novel
# -- output_dir LM_pretrained_book9
## news dataset 
# -- data data/dataset_thuner/raw_data/thuner_rawdata_small.txt 
# -- entity_list data/dataset_thuner/raw_data/thu_entity.txt 
# -- dataset lm_raw_data_thuner
# -- output_dir LM_pretrained_thuner



if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --transformer_mode transformer_base \
        --cuda \
        --multi_gpu \
        --output_dir LM_pretrained_thuner \
        --data data/dataset_thuner/raw_data/thuner_rawdata_small.txt \
        --entity_dict data/dataset_thuner/raw_data/thu_entity.txt \
        --dataset lm_raw_data_thuner \
        --bert_model bert_model \
        --learn_rate 3e-5 \
        --warmup_proportion 0.1 \
        --decay_rate 0.01 \
        --num_train_epochs 3.0 \
        --max_step 1000000000 \
        --max_seq_length 180 \
        --max_doc_length 100 \
        --max_cls_mems 100 \
        --memory_length 128 \
        --batch_size 32 \
        --sample_softmax -1 \
        ${@:2}
    #>result.log \
    #2>&1

else
    echo 'unknown argment 1'
fi
