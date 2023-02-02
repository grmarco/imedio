from datasets import load_dataset
import pandas as pd


def text_to_binary(examples):
    if examples == 'non-sexist':
        binary_vec = 0
    else:
        binary_vec = 1

    return binary_vec


train = pd.read_csv("../datasets/EXIST2021/training/EXIST2021_training.csv")
test = pd.read_csv("../datasets/EXIST2021/test/EXIST2021_test_labeled.csv")

train['label'] = train['task1'].apply(text_to_binary)
test['label'] = test['task1'].apply(text_to_binary)
train[train["language"] == 'en'].to_csv('../datasets/preprocessed/exist/en_train.csv')
train[train["language"] == 'es'].to_csv('../datasets/preprocessed/exist/es_train.csv')

test[test["language"] == 'en'].to_csv('../datasets/preprocessed/exist/en_test.csv')
test[test["language"] == 'es'].to_csv('../datasets/preprocessed/exist/es_test.csv')
