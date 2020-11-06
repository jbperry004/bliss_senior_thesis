import torch
import torchtext as tt
import os

from greekdataset import GreekCorpus
from rnn import RNNLanguageModel

BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT = tt.data.Field(sequential=True)
train, test, val = GreekCorpus.splits(TEXT, train="greek_corpus_train_encoded.txt", test="greek_corpus_test_encoded.txt", valid="greek_corpus_valid_encoded.txt")

TEXT.build_vocab(train.text)

train_iter, val_iter, test_iter = tt.data.BucketIterator.splits(
    (train, val, test), 
    batch_size=BATCH_SIZE, 
    repeat=False, 
    device=device)

language_model = RNNLanguageModel(TEXT, embedding_size=36, hidden_size=36).to(device)
language_model.train_all(train_iter, val_iter, epochs=5, learning_rate=0.001)
language_model.load_state_dict(rnn_tagger.best_model)

# Evaluate model performance
print(f'Training accuracy: {language_model.evaluate(train_iter):.3f}\n'
      f'Test accuracy:     {language_model.evaluate(test_iter):.3f}')
  
