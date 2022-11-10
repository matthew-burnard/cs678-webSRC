import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import WEIGHTS_NAME, AdamW, AutoModelForQuestionAnswering, AutoTokenizer, BertTokenizer, \
    get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import argparse
import os
import glob
from run import evaluate


class NewDataset(Dataset):
    def __init__(self, *tensors):
        tensors = tuple(tensor for tensor in tensors)
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors

    def __len__(self): return len(self.tensors[0])

    def __getitem__(self, index):
        output = [tensor[index] for tensor in self.tensors]
        return tuple(item for item in output)


def train(train_dataset):
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=10)

    t_total = len(train_dataloader) // 3

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    global_step = 0

    model.zero_grad()
    train_iterator = trange(int(3), desc="Epoch", disable=False)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            outputs = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2],
                            start_positions=batch[3], end_positions=batch[4])
            loss = outputs[0]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            # Print Loss
            if (step + 1) % 1 == 0:
                print("\nLoss: ", loss.item(), "\n")

                # Save model checkpoint
                if (step + 1) % 200 == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    path = './src/result/H-PLM_bert/model_step_' + str(step + 1)
                    model_to_save.save_pretrained(path)
                    print("\nModel Saved\n")


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
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

    features = torch.load('../data/cached/cached_train_bert-base-uncased_384_HTML')

    # Build the dataset from the features
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
    train_dataset = NewDataset(input_ids, input_mask, segment_ids, start_positions, end_positions)

    # Resize model
    tokenizer = BertTokenizer.from_pretrained('../src/result/V-PLM_bert')
    model.resize_token_embeddings(len(tokenizer))

    # Set for gpu/cpu
    model.to(device)

    # Train Model
    train(train_dataset)

    #checkpoint = '/content/drive/MyDrive/WebSRC-Baseline/src/result/H-PLM_bert/model_step_1800'
    #model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
    #model.to(device)

    #result = evaluate(args, model, tokenizer, prefix="test")
