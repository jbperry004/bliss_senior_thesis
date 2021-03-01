from statsmodels.stats.weightstats import ttest_ind
import numpy as np
import pandas as pd
import csv

SIGNIFICANCE = './results/statistical_significance.csv'

PATHS = ['./results/lemma_results.csv', './results/lemma_concat_results.csv', './results/encoded_labeled_results.csv']

path_to_method = {
    './results/plaintext_results.csv': "Plaintext",
    './results/lemma_results.csv': "Lemmatization",
    './results/lemma_concat_results.csv': "Lemma Concatenation",
    './results/encoded_labeled_results.csv': "Encoded",
}

dataset_to_col = {
    'train': 'Train Accuracy',
    'test': 'Test Accuracy',
    'apology': 'Apology Accuracy'
}

DIFFERENCE_SIGNIFICANCE = './results/difference_significance.csv'

# datasets = ['train', 'test', 'apology']
# for i, dataset in enumerate(datasets):
#     baseline = pd.read_csv('./results/plaintext_results.csv')[dataset_to_col[dataset]]
#     with open(f'./results/{dataset}_significance.csv', 'w') as f:
#         csv_writer = csv.writer(f, delimiter=',')
#         csv_writer.writerow(['Method', 't-statistic', 'p-value'])
#         for method in PATHS:
#             comparison = pd.read_csv(method)[dataset_to_col[dataset]]
#             t_stat, p_value, _ = ttest_ind(comparison, baseline, usevar='unequal')
#             csv_writer.writerow([path_to_method[method], t_stat, p_value])
    
for i, path in enumerate(PATHS):
    baseline = pd.read_csv('./results/plaintext_results.csv')
    baseline_diff = baseline['Train Accuracy'] - baseline['Test Accuracy']
    comparison = pd.read_csv(path)
    comparison_diff = comparison['Train Accuracy'] - comparison['Test Accuracy']
    t_stat, p_value, _ = ttest_ind(comparison_diff, baseline_diff, usevar='unequal')
    with open(DIFFERENCE_SIGNIFICANCE, 'a') as f:
        csv_writer = csv.writer(f, delimiter=',')
        if i == 0:
            csv_writer.writerow(['Method', 't-statistic', 'p-value'])
        csv_writer.writerow([path_to_method[path], t_stat, p_value])




