import torch
import torch.nn as nn
import copy

from tqdm import tqdm


class RNNLanguageModel(nn.Module):
    def __init__(self, text, embedding_size, hidden_size):
        super().__init__()
        self.text = text
        self.V = len(text.vocab.itos)
        self.text_pad_state_id = text.vocab.stoi[text.pad_token]

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(self.V, embedding_size, padding_idx=self.text_pad_state_id)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size)
        self.hidden2output = nn.Linear(hidden_size, self.V)
    
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.text_pad_state_id)
        
        self.init_parameters()
    
    def init_parameters(self, init_low=-0.15, init_high=0.15):
        for p in self.parameters():
            p.data.uniform_(init_low, init_high)

        
    def forward(self, text_batch, hidden_0):
        embeddings = self.embedding(text_batch)
        output, h_n = self.rnn(embeddings, hidden_0)
        logits = self.hidden2output(output)
        
        return logits
    
    def compute_loss(self, logits, ground_truth):
        # used transpose as per https://discuss.pytorch.org/t/for-beginners-do-not-use-view-or-reshape-to-swap-dimensions-of-tensors/75524
        loss = self.loss_criterion(torch.transpose(torch.transpose(logits, 0, 1), 1, 2), ground_truth.T)
        return loss
    
    def train_all(self, train_iter, val_iter, epochs=10, learning_rate=0.001):
        self.train()
        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        best_validation_accuracy = -float('inf')
        best_model = None
        # Run the optimization for multiple epochs
        for epoch in range(epochs): 
            total = 0
            running_loss = 0.0
            for batch in tqdm(train_iter):
                self.zero_grad()

                words = batch.text
                
                logits = self.forward(words, None)
                loss = self.compute_loss(torch.index_select(logits, 0, torch.tensor(range(0, len(words) - 1))), torch.index_select(words, 0, torch.tensor(range(1, len(words)))))

                loss.backward()

                optim.step()

                total += 1
                running_loss += loss.item()
            
            validation_accuracy = self.evaluate(val_iter)
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                self.best_model = copy.deepcopy(self.state_dict())
            epoch_loss = running_loss / total
            print (f'Epoch: {epoch} Loss: {epoch_loss:.4f} '
                 f'Validation accuracy: {validation_accuracy:.4f}')
    
    def predict(self, text_batch):
        logits = self.forward(text_batch, None)
        preds = torch.argmax(logits, dim=2)
        return preds.view(-1, text_batch.size(1))
            
    def evaluate(self, iterator):
        correct = 0
        total = 0
        for batch in tqdm(iterator):
            words = batch.text
            nexts = torch.index_select(words, 0, torch.tensor(range(1, len(words))))
            nexts_pred = torch.index_select(self.predict(words), 0, torch.tensor(range(0, len(words) - 1)))
            
            mask = nexts.ne(self.text_pad_state_id)
            
            cor = (nexts == nexts_pred)[mask]
            correct += cor.float().sum().item()
            total += mask.float().sum().item()

        return correct/total