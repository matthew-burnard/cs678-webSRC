import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, AutoModelForQuestionAnswering, BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm, trange


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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

    os.chdir('/content/drive/MyDrive/WebSRC-Baseline/src')
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