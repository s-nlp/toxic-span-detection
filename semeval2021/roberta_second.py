from transformers import BertTokenizer
from pathlib import Path
import torch


from box import Box
import pandas as pd
import collections
#import apex
import os
from tqdm import tqdm, trange
import sys
import random
import numpy as np
from sklearn.model_selection import train_test_split

import datetime

from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc

torch.cuda.empty_cache()
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
DATA_PATH = Path('./data/')
LABEL_PATH = Path('./labels/')

#AUG_DATA_PATH = Path('../data/data_augmentation/')

MODEL_PATH=Path('./models/')
LOG_PATH=Path('./logs/')
MODEL_PATH.mkdir(exist_ok=True)

model_state_dict = None

# BERT_PRETRAINED_PATH = Path('../../bert_models/pretrained-weights/cased_L-12_H-768_A-12/')
#BERT_PRETRAINED_PATH = Path('./bert_models/base_model')
BERT_PRETRAINED_PATH = None
# BERT_PRETRAINED_PATH = Path('../../bert_fastai/pretrained-weights/uncased_L-24_H-1024_A-16/')
#FINETUNED_PATH = Path('../models/finetuned_model.bin')
FINETUNED_PATH = None
# model_state_dict = torch.load(FINETUNED_PATH)

LOG_PATH.mkdir(exist_ok=True)

OUTPUT_PATH = MODEL_PATH/'roberta_second'
OUTPUT_PATH.mkdir(exist_ok=True)

args = Box({
    "run_text": "roberta_nocontext",
    "train_size": -1,
    "val_size": -1,
    "log_path": LOG_PATH,
    "full_data_dir": DATA_PATH,
    "data_dir": DATA_PATH,
    "no_cuda": False,
    "bert_model": BERT_PRETRAINED_PATH,
    "output_dir": OUTPUT_PATH,
    "max_seq_length": 512,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 9,
    "eval_batch_size": 16,
    "learning_rate": 5e-5,
    "num_train_epochs": 6,
    "warmup_proportion": 0.0,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "fp16_opt_level": "O1",
    "weight_decay": 0.0,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "max_steps": -1,
    "multi_gpu": False,
    "warmup_steps": 500,
    "logging_steps": 50,
    "eval_all_checkpoints": True,
    "overwrite_output_dir": True,
    "overwrite_cache": True,
    "seed": 42,
    "loss_scale": 128,
    "task_name": 'nocontext_toxic_roberta',
    "model_name": 'roberta-base',
    "model_type": 'roberta'
})

import logging

logfile = str(LOG_PATH/'log-{}-{}.txt'.format(run_start_time, args["run_text"]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger()
torch.cuda.set_device(3)
device = torch.device('cuda:3')
label_col = 'toxic'
#create databunch object
databunch = BertDataBunch(args['data_dir'], LABEL_PATH, tokenizer=args.model_name, 
                          train_file='second_part_train.csv', val_file='val.csv',
                          test_data='test.csv', label_file='labels.csv', 
                          text_col="comment_text", label_col=label_col,
                          batch_size_per_gpu=args['train_batch_size'], max_seq_length=args['max_seq_length'], 
                          multi_gpu=args.multi_gpu, multi_label=False, model_type=args.model_type)


metrics = []
metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
metrics.append({'name': 'roc_auc', 'function': roc_auc})
metrics.append({'name': 'fbeta', 'function': fbeta})

learner = BertLearner.from_pretrained_model(databunch, args.model_name, metrics=metrics, 
                                            device=device, logger=logger, output_dir=args.output_dir, 
                                            finetuned_wgts_path=FINETUNED_PATH, warmup_steps=args.warmup_steps,
                                            multi_gpu=args.multi_gpu, is_fp16=args.fp16, 
                                            multi_label=False, logging_steps=0)

#train our learner

learner.fit(args.num_train_epochs, args.learning_rate, validate=True)
learner.validate()

learner.save_model()