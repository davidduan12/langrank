import pandas as pd
import numpy as np


def create_bleu_dict(csv_file_path):
    # Load the data into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Create a dictionary with (Source lang, Transfer lang) as keys and BLEU as values
    result_dict = {(row['Source lang'], row['Transfer lang']): round(row['BLEU'],4) for idx, row in df.iterrows()}

    return result_dict

def create_task_bleu_dict(csv_file_path):
    df = pd.read_csv(csv_file_path)
    col = list(df.columns[1:])
    cols = df.values
    result_dict = {}
    for i in cols:
        if type(i[0]) != type(0.1):
            trans = i[0]
            for j,v in enumerate(i[1:]):
                result_dict[(col[j], trans)] = float(v)
        else:
            break
    return result_dict


bleu_dict = create_bleu_dict('data_tsfmt.csv')
task_bleu_dict = create_task_bleu_dict('LangRank Transfer Language Raw Data - MT Results.csv')
print(bleu_dict)
print(task_bleu_dict)

for i in bleu_dict:
    if bleu_dict[i] != task_bleu_dict[i]:
        print(bleu_dict[i])
        print(task_bleu_dict[i])


shared_items = {k: bleu_dict[k] for k in bleu_dict if k in task_bleu_dict and bleu_dict[k] == task_bleu_dict[k]}
print(len(shared_items))
#
# for i in task_bleu_dict:
#     for j in task_bleu_dict[i]:
#         if i != j:
#             if task_bleu_dict[i][j] != bleu_dict[(j,i)]:
#                 # print("ah-oh")
#                 pass
