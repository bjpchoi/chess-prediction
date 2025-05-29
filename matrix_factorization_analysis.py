import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib  # for saving model

# -------------------------
# 1) Load data with memory optimization
# -------------------------
# Use chunked reading for large files
chunk_size = 100000  # Adjust based on available memory
games_chunks = pd.read_csv('games.csv', chunksize=chunk_size)
player_map = pd.read_csv('player_mapping.csv')
n_players = player_map.shape[0]
id2idx = dict(zip(player_map.player_id, player_map.matrix_index))

# Process games in chunks to build match-level DataFrame
rows = []
for chunk in games_chunks:
    for _, r in chunk.iterrows():
        try:
            i = id2idx[r.white_id]
            j = id2idx[r.black_id]
            if r.winner == 'white':
                rows.append((i, j, 1))
            elif r.winner == 'black':
                rows.append((i, j, 0))
        except KeyError:
            continue  # Skip if player not in mapping

df = pd.DataFrame(rows, columns=['white_idx', 'black_idx', 'y'])
del rows  # Free memory

# -------------------------
# 2) Train/test split
# -------------------------
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.y
)
del df  # Free memory

# -------------------------
# 3) Build feature matrix with memory optimization
# -------------------------
def make_design_matrix(df):
    X = np.zeros((len(df), n_players), dtype=np.float32)  # Use float32 instead of float64
    for k, (i, j) in enumerate(zip(df.white_idx, df.black_idx)):
        X[k, i] += 1
        X[k, j] -= 1
    return X

X_train = make_design_matrix(train_df)
y_train = train_df.y.values.astype(np.int8)  # Use int8 for binary labels
X_test = make_design_matrix(test_df)
y_test = test_df.y.values.astype(np.int8)

# -------------------------
# 4) Fit baseline BT logistic regression
# -------------------------
clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    fit_intercept=False,
    solver='lbfgs',
    max_iter=1000,
    n_jobs=-1  # Use all available cores
)
clf.fit(X_train, y_train)
player_scores = clf.coef_.ravel()

# -------------------------
# 5) Compute schedule-strength for each player (Phase 2)
# -------------------------
sum_opp = np.zeros(n_players, dtype=np.float32)  # Use float32
count_opp = np.zeros(n_players, dtype=np.int32)

# Process training data in chunks
chunk_size = 100000
for i in range(0, len(train_df), chunk_size):
    chunk = train_df.iloc[i:i + chunk_size]
    for _, r in chunk.iterrows():
        i, j = r.white_idx, r.black_idx
        sum_opp[i] += player_scores[j]
        sum_opp[j] += player_scores[i]
        count_opp[i] += 1
        count_opp[j] += 1

sched_strength = sum_opp / np.maximum(count_opp, 1)

# -------------------------
# 6) Build augmented features & refit logistic (Phase 3)
# -------------------------
white_idx = train_df.white_idx.values
black_idx = train_df.black_idx.values

delta_s = player_scores[white_idx] - player_scores[black_idx]
delta_sched = sched_strength[white_idx] - sched_strength[black_idx]

X_aug_train = np.vstack([delta_s, delta_sched]).T

clf2 = LogisticRegression(
    penalty='l2',
    C=1.0,
    fit_intercept=True,
    solver='lbfgs',
    max_iter=1000,
    n_jobs=-1
)
clf2.fit(X_aug_train, y_train)

# -------------------------
# 7) Evaluate on test set
# -------------------------
wi_test = test_df.white_idx.values
bi_test = test_df.black_idx.values

delta_s_test = player_scores[wi_test] - player_scores[bi_test]
delta_sched_test = sched_strength[wi_test] - sched_strength[bi_test]
X_aug_test = np.vstack([delta_s_test, delta_sched_test]).T

y_prob2 = clf2.predict_proba(X_aug_test)[:, 1]
y_pred2 = (y_prob2 > 0.5).astype(np.int8)

acc2 = accuracy_score(y_test, y_pred2)
roc_auc2 = roc_auc_score(y_test, y_prob2)

# Compute standard errors
n_test = len(y_test)
acc1 = accuracy_score(y_test, (clf.predict_proba(X_test)[:, 1] > 0.5).astype(np.int8))
se1 = np.sqrt(acc1 * (1-acc1) / n_test)
se2 = np.sqrt(acc2 * (1-acc2) / n_test)

print(f'Baseline BT →    Acc = {acc1:.4f} ± {se1:.4f}, '
      f'AUC = {roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]):.4f}')
print(f'Augmented Model → Acc = {acc2:.4f} ± {se2:.4f}, AUC = {roc_auc2:.4f}')

print('\nAugmented Model Weights:')
print(f'Player Strength Difference Weight: {clf2.coef_[0][0]:.4f}')
print(f'Schedule Strength Difference Weight: {clf2.coef_[0][1]:.4f}')
print(f'Intercept: {clf2.intercept_[0]:.4f}')

# -------------------------
# 8) Save results
# -------------------------
joblib.dump(clf2, 'bt_augmented_model.pkl')

out = pd.DataFrame({
    'player_id': player_map.player_id,
    'bt_score': player_scores,
    'sched_strength': sched_strength,
})
out.to_csv('player_bt_with_schedule.csv', index=False)
print('Saved augmented model and player schedule strengths.')
