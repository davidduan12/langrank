import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRanker
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score

# Load the data
data = pd.read_csv('data_tsfmt.csv')

# Define feature columns and target column
features = [
    'Overlap word-level', 'Overlap subword-level', 'Transfer lang dataset size',
    'Target lang dataset size', 'Transfer over target size ratio', 'Transfer lang TTR',
    'Target lang TTR', 'Transfer target TTR distance', 'GENETIC', 'SYNTACTIC',
    'FEATURAL', 'PHONOLOGICAL', 'INVENTORY', 'GEOGRAPHIC'
]
target = 'BLEU'

# Assign relevance scores
data['relevance'] = 0
for source_lang in data['Source lang'].unique():
    source_lang_data = data[data['Source lang'] == source_lang]
    top_indices = source_lang_data.nlargest(10, target).index
    data.loc[top_indices, 'relevance'] = range(10, 0, -1)

# Save the processed data
data.to_csv('data_processed.csv', index=False)

groups = data['Source lang']
ndcg_scores = []

# Parameters for LightGBM
params = {
    'boosting_type' : 'gbdt',
    'objective': 'lambdarank',
    'n_estimators':100,
    'metric' : 'ndcg',
    'num_leaves': 16,
    'min_data_in_leaf' : 5,
    'output_model' :'LightGBM_model.txt',
    'verbose ':-1,
    'eval_at': [1, 2, 3, 4, 5],
    'early_stopping_round':10,
}

query = [53] * (len(data) // 53)

# Leave-one-source-language-out cross-validation
logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(data, groups=groups):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    train_X = train_data[features]
    train_y = train_data['relevance']
    test_X = test_data[features]
    test_y = test_data['relevance']

    # Define the groups for LightGBM (same for training and testing)
    train_query = [53] * (len(train_X) // 53)
    test_query = [53]

    # Prepare LightGBM dataset with query information
    train_dataset = lgb.Dataset(train_X, label=train_y, group=train_query)
    test_dataset = lgb.Dataset(test_X, label=test_y, group=test_query, reference=train_dataset)

    # Train the model
    ranker = lgb.train(params, train_dataset, valid_sets=[test_dataset])

    # Predict and evaluate NDCG@3
    y_pred = ranker.predict(test_X, num_iteration=ranker.best_iteration)
    ndcg = ndcg_score([test_y], [y_pred], k=3)
    ndcg_scores.append(ndcg)

    # Save the model
    ranker.save_model('LightGBM_model.txt')

# Calculate the average NDCG@3 score
average_ndcg = np.mean(ndcg_scores)
print(f'Average NDCG@3: {average_ndcg}')