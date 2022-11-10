import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import WEIGHTS_NAME, AdamW, AutoModelForQuestionAnswering, AutoTokenizer, BertTokenizer, \
    get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import argparse
import os
import glob
from run import evaluate

class Args():
    def __init__(self):
        self.trian_file = '../data/websrc1.0_train_.json'
        self.predict_file = '../data/websrc1.0_dev_.json'
        self.root_dir = '../data'
        self.model_type = "Bert"
        self.method = "H-PLM"
        self.model_name_or_path = "bert-base-uncased"
        self.output_dir = 'result/H-PLM_bert/'
        self.config_name = ""
        self.tokenizer_name = ""
        self.cache_dir = None
        self.do_lower_case = True
        self.cnn_feature_dir = None
        self.num_node_block = 3
        self.cnn_feature_dim = 1024
        self.max_seq_length = 384
        self.doc_stride = 128
        self.max_query_length = 64
        self.max_answer_length = 30
        self.verbose_logging = True
        self.do_train = False
        self.do_eval = False
        self.local_rank = -1
        self.overwrite_cache = False
        self.per_gpu_train_batch_size = 8
        self.per_gpu_eval_batch_size = 8
        self.evaluate_during_training = False
        self.eval_all_checkpoints = False
        self.eval_from_checkpoint = 0
        self.eval_to_checkpoint = None
        self.learning_rate = 1e-5
        self.gradient_accumulation_steps = 1
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 5
        self.max_steps = -1
        self.warmup_steps = 0
        self.n_best_size = 20
        self.logging_steps = 3000
        self.save_steps = 3000
        self.no_cuda = True
        self.overwrite_output_dir = False
        self.overwrite_cache = False
        self.save_features = True
        self.seed = 42
        self.fp16 = False
        self.fp16_opt_level = '01'


if __name__ == "__main__":
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    args.device = device


    checkpoint = '/content/drive/MyDrive/WebSRC-Baseline/src/result/H-PLM_bert/model_step_1800'
    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
    model.to(device)

    result = evaluate(args, model, tokenizer, prefix="test")
