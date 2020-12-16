import torch
import torchtext as tt
import os

from rnn import LSTMClassifier

BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT = tt.data.Field(sequential=True, include_lengths=True)
LABEL = tt.data.Field(sequential=False, unk_token=None)

train, val, test = tt.data.TabularDataset.splits(
        path='../data/',
        format='csv',
        train="greek_corpus_train_charngram_noword.csv",
        validation="greek_corpus_valid_charngram_noword.csv", 
        test="greek_corpus_test_charngram_noword.csv",
        fields=[('text', TEXT), ('label', LABEL)],
        csv_reader_params={'delimiter':'\t'},
        )

TEXT.build_vocab(train.text)
LABEL.build_vocab(train.label)

train_iter, val_iter, test_iter = tt.data.BucketIterator.splits(
    (train, val, test), 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.text),
    repeat=False,
    device=device)

classifier = LSTMClassifier(TEXT, LABEL, embedding_size=128, hidden_size=64).to(device)
classifier.train_all(train_iter, val_iter, epochs=10, learning_rate=0.001)
classifier.load_state_dict(classifier.best_model)

print(f'Training accuracy: {classifier.evaluate(train_iter):.3f}\n'
      f'Test accuracy:     {classifier.evaluate(test_iter):.3f}')