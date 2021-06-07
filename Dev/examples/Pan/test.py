from collections import Counter
import pandas as pd
import csv
import torch
import os.path as osp
import io

path ="/usr/src/kdd/data_test6/pcqm4m_kddcup2021/raw/data.csv.gz"
outpath = "/usr/src/kdd/suball.csv"
extra = "/data/processed/"


def read_subgraphs(idx):
    try:
        with open(osp.join(extra, 'geometric_data_processed_data_{}.pt'.format(idx)), 'rb') as f:
            sub = torch.load(io.BytesIO(f.read()))

        # sub = torch.load(osp.join(self.extra, 'geometric_data_processed_data_{}.pt'.format(idx))) too many open files
    except Exception as ex:
        sub = None

    return sub

5.681737
def get_sublist():
    data_df = pd.read_csv(path)
    with open(outpath, 'w', newline='') as csvfile:
        fieldnames = ['idx', 'main_smiles','sub_smiles','gap']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for index, row in data_df.iterrows():

            idx = row["idx"]
            print(idx)
            sub = read_subgraphs(idx)
            if sub is not None:
                for s in sub.smiles :
                    writer.writerow({'idx':idx, 'main_smiles' : row["smiles"], "sub_smiles" : s , "gap":row["homolumogap"]})

pd.options.display.max_columns = None
pd.options.display.max_rows = None
sorted_df[(sorted_df.freq > 1000)&((sorted_df.level_1 == 0.1) & (sorted_df.gap > 5))]

def get_stats_mean():
    df = pd.read_csv(outpath)
    count = df["sub_smiles"].value_counts()
    a = df.groupby("sub_smiles").mean()
    a = df.groupby("sub_smiles").quantile([0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
    a.reset_index(inplace=True)
    df_m = pd.merge(a, count, how="left", left_on="sub_smiles", right_index=True)
    sorted_df = a.sort_values(by="gap")
    count = df["sub_smiles"].value_counts()
    df_m = pd.merge(sorted_df, count, how="left", left_index=True, right_index=True)
    df_m = df_m.rename(columns={"sub_smiles": "freq"})
    s_freq = df_m.sort_values(by="freq")
 sorted_df = df_m.sort_values(["sub_smiles_y", "level_1"])

ma = sorted_df[(sorted_df.freq > 10000)&((sorted_df.level_1 == 0.1) & (sorted_df.gap > 5))]

 mi =sorted_df[(sorted_df.freq > 10000)&((sorted_df.level_1 == 0.9) & (sorted_df.gap < 5))]


df["homolumogap"].quantile([0.1,0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
0.1    4.315726
0.2    4.740223
0.3    5.053154
0.5    5.575613
0.6    5.839563
0.7    6.166100
0.8    6.612367
0.9    7.257276
