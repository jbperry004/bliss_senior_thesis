import torchtext as tt
import os

class GreekCorpus(tt.data.Dataset):
  @staticmethod
  def sort_key(ex):
    return len(ex.text)

  def __init__(self, path, text_field, **kwargs):
    fields = [('text', text_field)]
    examples = []
    # Get text
    with open(path, 'r') as f:
        for line in f:
            ex = tt.data.Example()
            ex.text = text_field.preprocess(line.strip()) 
            examples.append(ex)
            
    super(GreekCorpus, self).__init__(examples, fields, **kwargs)

  @classmethod
  def splits(cls, text_field, path='../data/',
              train='train', test='test', valid='valid',
              **kwargs):
    train_data = None if train is None else cls(
        os.path.join(path, train), text_field, **kwargs)
    test_data = None if test is None else cls(
        os.path.join(path, test), text_field, **kwargs)
    valid_data = None if valid is None else cls(
        os.path.join(path, valid), text_field, **kwargs)
    return tuple(d for d in (train_data, test_data, valid_data)
                  if d is not None)