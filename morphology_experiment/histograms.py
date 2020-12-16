import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# accuracies = pd.read_csv("./apology_accuracies.csv")

# plt.figure(figsize=[10, 5])
# for i, col in enumerate(accuracies.columns):
#     plt.subplot(2, 2, i + 1)
#     plt.hist(accuracies[col], 50, range=[0.0, 1.0])
#     plt.ylim(ymin=0, ymax=3)
#     plt.title(col)
#     plt.xlabel("Apology Accuracy")
#     plt.ylabel("Frequency")

# plt.tight_layout()
# plt.show()


runs = ["lemma_concat_fixed_9", "lemma_fixed_9", "byte_pair_fixed_9", "original_9"]
datasets = ["train", "valid"]

rundict = {"lemma_concat_fixed_9": "Lemma_Concat", "lemma_fixed_9": "Lemma", "byte_pair_fixed_9": "Byte_Pair", "original_9": "Plaintext"}
datasetdict = {"valid": "Validation Loss", "train": "Train Loss"}

plt.figure(figsize=[15, 5])
for i, dataset in enumerate(datasets):
    plt.subplot(1, 3, i + 1)
    for run in runs:
        history = pd.read_csv(f"./results/run-{run}-tag-Loss_{dataset}.csv")
        plt.plot(history["Step"], history["Value"], label=rundict[run])

    plt.title(datasetdict[dataset])  
    plt.xlabel("Epochs")
    plt.ylabel(datasetdict[dataset])
    plt.ylim(ymin=0, ymax=1)
    plt.legend()


plt.subplot(1, 3, 3)
for run in runs:
    train_history = pd.read_csv(f"./results/run-{run}-tag-Loss_train.csv")
    valid_history = pd.read_csv(f"./results/run-{run}-tag-Loss_valid.csv")

    plt.plot(train_history["Step"], valid_history["Value"] - train_history["Value"], label=rundict[run])
    plt.title("Difference between Validation Loss and Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Difference in Loss")
    plt.ylim(ymin=-0.5, ymax=0.5)
    plt.legend()


plt.tight_layout()
plt.show()