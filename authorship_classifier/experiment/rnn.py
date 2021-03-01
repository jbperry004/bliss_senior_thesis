import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import copy

from tqdm import tqdm

import wandb

class LSTMClassifier(nn.Module):
    def __init__(self, text, author, embedding_size, hidden_size, experiment):
        super().__init__()
        self.text = text
        self.V = len(text.vocab.itos)
        self.K = len(author.vocab.itos)
        self.text_pad_state_id = text.vocab.stoi[text.pad_token]

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(self.V, embedding_size, padding_idx=self.text_pad_state_id)
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        self.hidden2output = nn.Linear(hidden_size*2, self.K)
    
        self.loss_criterion = nn.CrossEntropyLoss()
    
        wandb.init(project="authorship-classifier")

        self.init_parameters()
    
    def init_parameters(self, init_low=-0.30, init_high=0.30):
        for p in self.parameters():
            p.data.uniform_(init_low, init_high)
        self.embedding.weight.data[self.text_pad_state_id] = 0

        
    def forward(self, text_batch, hidden_0):
        text, seq_lens = text_batch
        
        embeddings = self.embedding(text)
        packed = pack_padded_sequence(embeddings, seq_lens, batch_first=False)
        output, (h_n, c_n) = self.rnn(packed)

        h_n = self.dropout(torch.transpose(h_n, 0, 1).reshape(-1, self.hidden_size * 2))
        logits = self.hidden2output(h_n)
        return torch.squeeze(logits, dim=1)

    
    def compute_loss(self, logits, ground_truth):
        loss = self.loss_criterion(logits, ground_truth)
        return loss
    
    def train_all(self, train_iter, val_iter, epochs=10, learning_rate=0.001):
        wandb.config.update({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size})

        self.train()
        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        best_validation_loss = float('inf')
        best_model = None
        # Run the optimization for multiple epochs
        for epoch in range(epochs): 
            total = 0
            running_loss = 0.0
            for batch in tqdm(train_iter):
                self.zero_grad()
                logits = self.forward(batch.text, None)
                loss = self.compute_loss(logits, batch.author)
                loss.backward()

                optim.step()

                total += 1
                running_loss += loss.item()
            
            valid_total = 0
            valid_running_loss = 0
            for batch in tqdm(val_iter):
                logits = self.forward(batch.text, None)
                loss = self.compute_loss(logits, batch.author)
                valid_total += 1
                valid_running_loss += loss.item()


            epoch_loss = running_loss / total
            valid_loss = valid_running_loss / valid_total
            validation_accuracy = self.evaluate(val_iter)

            wandb.log({
                "Loss/train": epoch_loss,
                "Loss/valid": valid_loss,
                "Accuracy/valid": validation_accuracy})

            if valid_loss < best_validation_loss:
                best_validation_loss = valid_loss
                self.best_model = copy.deepcopy(self.state_dict())

            print (f'Epoch: {epoch} Training Loss: {epoch_loss:.4f} '
                f'Validation loss: {valid_loss:.4f}'
                 f'Validation accuracy: {validation_accuracy:.4f}')
                 
    def predict(self, text_batch):
        logits = self.forward(text_batch, None)
        _, preds = torch.max(logits, 1)
        return preds
            
    def evaluate(self, iterator):
        self.eval()
        correct = 0
        total = 0
        for batch in tqdm(iterator):
            words = batch.text
            authors = batch.author
            authors_pred = self.predict(words)

            correct += torch.sum(authors == authors_pred)
            total += len(authors)
            

        return correct/total