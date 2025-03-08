import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, AutoTokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import pandas as pd

class StoryDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.tokenizer = tokenizer
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = data

        df = pd.DataFrame(data, index=[10 * i for i in range(len(self.data))])
        labels = df.columns
        self.labels = labels


    def __getitem__(self, idx):
        input_text = self.data[idx]['input']
        output_text = self.data[idx]['target']

        encoding = self.tokenizer.encode_plus(
            input_text, add_special_tokens=True, max_length=1024, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        if self.labels is not None:  # Assuming you have labels in your dataset
            label_ids = self.tokenizer.encode(output_text, add_special_tokens=False, padding='max_length', truncation=True)
            item['labels'] = torch.tensor(label_ids)
        return item

    def __len__(self):
        return len(self.data)

# Set up tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

#train_dataset = StoryDataset('data/syntdatatrain.json', tokenizer)
#val_dataset = StoryDataset('data/syntdataval.json', tokenizer)
#test_dataset = StoryDataset('data/syndatatest.json', tokenizer)

# Prepare training data
dataset = StoryDataset("data/syntdatatrain.json", tokenizer)
train_dataloader = DataLoader(dataset, batch_size=4, num_workers=0)
eval_dataset = StoryDataset("data/syntdataval.json", tokenizer)
#eval_dataloader
# Set up training arguments and trainer
training_args = TrainingArguments(
    output_dir='output',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    evaluation_strategy="epoch",
    logging_dir="logs",
    dataloader_num_workers=1,
)

#def data_collator(batch):
#    #inputs = [ex['inputs'] for ex in batch]
#    input_ids = [ex['input_ids'] for ex in batch]
#    decoder_input_ids = [ex['decoder_input_ids'] for ex in batch]
#    return {'input_ids': torch.stack(input_ids), 'decoder_input_ids': torch.stack(decoder_input_ids)}

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, return_tensors='pt')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
    )
# ...

#loss = trainer.compute_loss(model, (batch['input_ids'], batch['decoder_input_ids']))
# Train the model
if __name__ == '__main__':
    trainer.train()