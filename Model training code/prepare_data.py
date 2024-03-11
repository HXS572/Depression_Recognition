"""
For preparing data
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os
# import sys

# Used to organize and create a unified dataframe for all datasets.
class CorpusDataFrame():

    def __init__(self):
        self.data = []
        self.exceptions = 0

    # def append_file(self, path, name, label1, label2):
    def append_file(self, path, name, label1, label2, label3):
        # Append filename, filepath, and emotion label to the data list.
        try:
            # avoid broken files
            s = torchaudio.load(path)
            self.data.append({
                "name": name,
                "path": path,
                "class_2": label1,   # For binary classification
                "class_4": label2,   # For multiclassification
                "score": label3      # PHQ-8 score
            })
        except Exception as e:
            print('Could not load ', str(path), e)
            self.exceptions+=1
            pass

    def data_frame(self):
        if self.exceptions > 0: print(f'{exceptions} files could not be loaded')

        # Create the dataframe from the organized data list
        df = pd.DataFrame(self.data)
        return df


# Depression list of daic-woz data for creating unified data frames
dep = [308, 309, 311,319,320,321,325,330,332,335,337,338,339,344,345,346,
     347,348,350,351,352,353,354,355,356,359,362,365,367,372,376,377,
     380,381,384,386,388,389,402,405,410,412,413,414,418,421,422,426,
     433,440,441,448,453,459,461,483]


def DAIC_WOZ(data_path):
    print('PREPARING DAIC_WOZ DATA PATHS')

    cdf = CorpusDataFrame()

    for path in tqdm(Path(data_path).glob("**/*.wav")):
        name = str(path).split('/')[-1].split('.')[0]

        label = str(path).split('/')[-1].split('_')[0]  # 第1个下划线之后的标签
        label3 = str(path).split('_')[1]  # 第3个下划线之后的标签

        # based on binary, 0, 1
        if int(label) in dep:
            label1 = 'dep'
        else:
            label1 = 'ndep'

        # Classification of depression severity based on PHQ-8 score
        if 0 <= int(label3) <= 4:
            label2 = 'non'
        elif 5 <= int(label3) <= 9:
            label2 = 'mild'
        elif 10 <= int(label3) <= 14:
            label2 = 'moderate'
        else:
            label2 = 'severe'

        # cdf.append_file(path, name, label1, label2)
        cdf.append_file(path, name, label1, label2, label3)
    df = cdf.data_frame()
    return df


# Use the correct function to iterate through the named dataset.
def get_df(corpus, data_path, i=None):
    if corpus == 'daic_woz':
        df = DAIC_WOZ(data_path)
    try:
        return df
    except:
        raise ValueError("Invalid corpus name")

# To get the datasets names and their file paths
def df(corpora, data_path):

    # In case more than one dataset is used.
    if type(corpora) == list:
        df = pd.DataFrame()
        for i, corpus in enumerate(corpora):
            df_ = get_df(corpus, data_path[i])
            df = pd.concat([df, df_], axis = 0)
    else:
        df = get_df(corpora, data_path)

    print(f"Step 0: {len(df)}")

    # Filter out non-existing files.
    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop("status", 1)
    print(f"Step 1: {len(df)}")

    df = df.sample(frac=1)
    df = df.reset_index(drop=True)

    # Explore the number of emotion lables in the dataset with what distribution.
    print("Labels: ", df["class_4"].unique())
    # print("Labels: ", df["class_2"].unique(), df["class_4"].unique())
    # print()
    # df.groupby("class_2").count()[["path"]]
    print()
    df.groupby("class_4").count()[["path"]]

    return df


def prepare_splits(df, config, evaluation=False):
    output_dir = config['output_dir']
    save_path = output_dir + "/splits/"


    # Create splits directory
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Create train, test, and validation splits.
    random_state = config['seed']  # 103
    # 6:2:2
    train_df, test_df = train_test_split(df, test_size=0.2, train_size=0.8, random_state=random_state, stratify=df["class_4"])
    train_df, valid_df = train_test_split(train_df, test_size=0.25, train_size=0.75, random_state=random_state, stratify=train_df["class_4"])

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    # Save each to file
    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
    valid_df.to_csv(f"{save_path}/valid.csv", sep="\t", encoding="utf-8", index=False)

    print(f'train: {train_df.shape},\t validate: {valid_df.shape},\t test: {test_df.shape}')


# Match eval_df to df by removing additional labels in eval_df
def remove_additional_labels(df, eval_df):
    df_labels = df["class_4"].unique()
    eval_df_labels = eval_df["class_4"].unique()

    print("Default dataset labels: ", df_labels)
    print("Evaluation dataset labels: ", eval_df_labels)

    additional_labels = []
    for label in eval_df_labels:
        if label not in df_labels:
            additional_labels.append(label)

    print("Length of evaluation dataset: \t", len(eval_df))

    # Remove labels not in the orginal df
    # Modification based on classification tasks
    eval_df = eval_df[eval_df.class_4.isin(additional_labels) == False]
    # eval_df = eval_df[eval_df.class_2.isin(additional_labels) == False]

    # print(f"Length of evaluation dataset after removing {additional_labels}: \t", len(eval_df))
    eval_df_labels = eval_df["class_4"].unique()
    # print("Evaluation dataset labels: ", eval_df_labels)

    return eval_df


if __name__ == '__main__':
    import argparse
    import yaml

    # Get the configuration file containing dataset name, path, and other configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='yaml configuration file path')
    args = parser.parse_args()
    config_file = args.config

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_filepath = config['output_dir'] + "/splits/train.csv"
    test_filepath = config['output_dir'] + "/splits/test.csv"
    valid_filepath = config['output_dir'] + "/splits/valid.csv"

    # Create a dataframe
    df = df(config['corpora'], config['data_path'])

    # Create train, test, and validation splits and save them to file
    prepare_splits(df, config)


    # # If a different dataset is used to test the model:
    # if config['test_corpora'] is not None:
    #     # Create a dataframe
    #     eval_df = df(config['test_corpora'], config['test_corpora_path'])
    #
    #     # Match eval_df to df
    #     eval_df = remove_additional_labels(df, eval_df)
    #
    #     # Create train, test, and validation splits and save them to file
    #     prepare_splits(eval_df, config, evaluation=True)
