import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRanker
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score

# Load the data
data = pd.read_csv('data_tsfmt.csv')
groups = data['Source lang']
logo = LeaveOneGroupOut()
# Define feature columns and target column
# features = [
#     'Overlap word-level', 'Overlap subword-level', 'Transfer lang dataset size',
#     'Target lang dataset size', 'Transfer over target size ratio', 'Transfer lang TTR',
#     'Target lang TTR', 'Transfer target TTR distance', 'GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC'
# ]
features = ['GEOGRAPHIC']
target = 'BLEU'

data['relevance'] = 0
for source_lang in data['Source lang'].unique():
    source_lang_data = data[data['Source lang'] == source_lang]
    top_indices = source_lang_data.nlargest(10, target).index
    data.loc[top_indices, 'relevance'] = range(10,0,-1)

data.to_csv('data.csv', index=False)
groups = data['Source lang']
ndcg_scores = []

# Parameters for LightGBM
ranker = LGBMRanker(
    boosting_type='gbdt',
    objective ='lambdarank',
    n_estimators=100,
    metric ='ndcg',
    num_leaves=16,
    min_data_in_leaf=5,
    verbose=-1
)

query = [53] * 53


# Leave-one-source-language-out cross-validation
for train_idx, test_idx in logo.split(data, groups=groups):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    train_X = train_data[features]
    train_y = train_data['relevance']
    test_X = test_data[features]
    test_y = test_data['relevance']


    # Prepare LightGBM dataset with query information
    train_dataset = lgb.Dataset(train_X, label=train_y,group=query)
    test_dataset = lgb.Dataset(test_X, label=test_y, group=[53], reference=train_dataset)


    # Train the model
    ranker.fit(train_X, train_y, group=query, eval_set=[(test_X, test_y)], eval_group=[[53]])

    # Predict and evaluate NDCG@3
    y_pred = ranker.predict(test_X)
    ndcg = ndcg_score([test_y], [y_pred], k=3)
    ndcg_scores.append(ndcg)

ranker.booster_.save_model('LightGBM_model.txt')
# Calculate the average NDCG@3 score
average_ndcg = np.mean(ndcg_scores)
print(f'Average NDCG@3: {average_ndcg}')
