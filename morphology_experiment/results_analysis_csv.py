import pandas as pd

df = pd.read_csv("./results.csv", index_col=0, names=['train_acc_mean', 'train_acc_std', 'test_acc_mean', 'test_acc_std', 'ap_acc_mean', 'app_acc_std'])

print(df['train_acc_mean'] - df['test_acc_mean'])