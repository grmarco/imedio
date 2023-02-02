from datasets import load_dataset


def text_to_binary(examples):
    if examples["Propaganda"] == 'No' or examples["Propaganda"] == 'Not':
        binary_vec = 0
    else:
        binary_vec = 1

    return {"label": binary_vec}


import pandas as pd

path = "/media/gmarco/data/workspace-local/imedio/datasets/propaganda/"

emojis = pd.read_csv(path + 'emojis.csv')


def reformat_emojis(tuit):
    for emoji, meaning in zip(emojis.emoji.to_list(), emojis.meaning.to_list()):
        tuit = tuit.replace(emoji, "_%s_" % meaning)
    return tuit


moral = pd.read_csv(path + "moral.csv")
moral.Text = moral.Text.apply(reformat_emojis)
moral = moral.rename(columns={'label': 'label_cat', 'Text': 'text'})
mapping = {'Not': 0, 'Group': 1, 'Group II Discrediting the opponent': 2, 'Group III Loaded Language': 3,
           'Group IV Appeal to authority': 4}
mapping_binary = {'No': 0, 'Not': 0, 'Yes': 1}
mapping_without_neg = {'Not': -1, 'Group': 0, 'Group II Discrediting the opponent': 1, 'Group III Loaded Language': 2,
           'Group IV Appeal to authority': 3}


moral['label_b'] = moral['Propaganda'].apply(lambda x: mapping_binary[x])
moral['label_without'] = moral['label_cat'].apply(lambda x: mapping_without_neg[x])
moral['label_cat'] = moral['label_cat'].apply(lambda x: mapping[x])

print(moral['label_cat'].value_counts())

moral = moral.sample(frac=1.0)
split = round(len(moral) * 0.8)
moral_train = moral[:split].to_csv(path + 'train.csv')
moral_test = moral[split:].to_csv(path + 'test.csv')
