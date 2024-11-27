#Transformer model: https://huggingface.co/openai-community/gpt2

from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle

class TextClassifierDataset(Dataset):
    def __init__(self, text_examples, text_labels, text_tokenizer, text_model):
        self.X = text_examples
        self.y = text_labels
        self.tokenizer = text_tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token
        self.model = text_model
        self.model.to('cuda')

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        to_pass = self.X[idx]
        label = self.y[idx]
        
        encoded_input = self.tokenizer(to_pass, padding='max_length', truncation=True, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        last_token_embedding = torch.squeeze(output.last_hidden_state[:, -1, :])

        return last_token_embedding, label

tokenizer = AutoTokenizer.from_pretrained('gpt2', force_download=True)
model = AutoModel.from_pretrained('gpt2', force_download=True)

#Texts from https://huggingface.co/datasets/artem9k/ai-text-detection-pile

base_path = '../data/texts'

df = pd.read_parquet(base_path)

X = df['text'].values.tolist()
y_raw = df['source'].values.tolist()

y = []

for i in y_raw:
    if i == 'human':
        y.append(1)
    elif i == 'ai':
        y.append(0)
    else:
        raise

dataset = TextClassifierDataset(X, y, tokenizer, model)

dl = DataLoader(dataset, batch_size=4096, shuffle=False)

X_store = []
y_store = []

for X, y in dl:   # gives batch data
    X_store.append(X)
    y_store.append(y)

X_save = torch.cat(X_store)
y_save = torch.cat(y_store)

with open('../data/texts/gpt_encoded.pkl', 'wb') as f:
    pickle.dump([X_save, y_save], f)

