from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering, ElectraConfig, ElectraTokenizer, ElectraForQuestionAnswering
import torch
from torch.utils.data import Dataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer, BertTokenizer, ElectraTokenizer,
    get_linear_schedule_with_warmup,
)

config = AutoConfig.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")




class NewDataset(Dataset):
    def __init__(self, *tensors):
      tensors = tuple(tensor for tensor in tensors)
      assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
      self.tensors = tensors

    def __len__(self): return len(self.tensors[0])

    def __getitem__(self, index):
      output = [tensor[index] for tensor in self.tensors]
      return tuple(item for item in output)

features = torch.load('../data/cached/cached_train_bert-base-uncased_384_HTML')

all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
train_dataset = NewDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions)

tokenizer = BertTokenizer.from_pretrained('./src/result/V-PLM_bert')
model.resize_token_embeddings(len(tokenizer))

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=10)

from tqdm import tqdm, trange

t_total = len(train_dataloader) // 3

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=t_total)

global_step = 0
tr_loss, logging_loss = 0.0, 0.0
model.zero_grad()
train_iterator = trange(int(3), desc="Epoch", disable=False)
# set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=-1 not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        model.train()
        batch = tuple(t.to(torch.device("cpu")) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'start_positions': batch[3],
                  'end_positions': batch[4]}
        ##print(inputs)
        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        loss.backward()

        tr_loss += loss.item()
        if (step + 1) % 1 == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            print("\n\n\nLoss: ", loss.item(), "\n\n\n")

            if (step + 1) % 200 == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                path = './src/result/H-PLM_bert/model_step_' + str(step + 1)
                model_to_save.save_pretrained(path)
                # torch.save(model_to_save,)
                print("Saved")


        if 0 < -1 < global_step:
            epoch_iterator.close()
            break
    if 0 < -1 < global_step:
        train_iterator.close()
        break
