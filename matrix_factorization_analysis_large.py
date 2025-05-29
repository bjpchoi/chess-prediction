import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import sparse
import joblib
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# Enable tqdm for pandas
tqdm.pandas()

def process_chunk(chunk, id2idx):
    """Process a chunk of games and return valid matches."""
    rows = []
    for _, r in chunk.iterrows():
        try:
            i = id2idx[r.White]
            j = id2idx[r.Black]
            if r.Result == '1-0':
                rows.append((i, j, 1))
            elif r.Result == '0-1':
                rows.append((i, j, 0))
        except (KeyError, ValueError):
            continue
    return rows

print("Loading data...")
# First pass: count total games and collect unique players
print("First pass: counting games and collecting players...")
chunk_size = 25000  # Even smaller chunks
games = pd.read_csv('chess_games.csv', chunksize=chunk_size)
all_players = set()
total_games = 0

for chunk in tqdm(games, desc="First pass"):
    all_players.update(chunk['White'].unique())
    all_players.update(chunk['Black'].unique())
    total_games += len(chunk)
    gc.collect()

# Create player mapping
print("Creating player mapping...")
player_map = pd.DataFrame({
    'player_id': list(all_players),
    'matrix_index': np.arange(len(all_players))
})
n_players = len(all_players)
id2idx = dict(zip(player_map.player_id, player_map.matrix_index))
del all_players
gc.collect()

print(f"Total unique players: {n_players}")

# Second pass: process games in chunks
print("Second pass: processing games...")
games = pd.read_csv('chess_games.csv', chunksize=chunk_size)
all_rows = []

for chunk in tqdm(games, desc="Processing games"):
    rows = process_chunk(chunk, id2idx)
    all_rows.extend(rows)
    gc.collect()

df = pd.DataFrame(all_rows, columns=['white_idx', 'black_idx', 'y'])
del all_rows
gc.collect()

print(f"\nProcessed {len(df)} valid games out of {total_games} total games")

print("Splitting data...")
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.y
)
del df
gc.collect()

print("Building feature matrices...")
def make_sparse_design_matrix(df, n_players):
    """Build sparse design matrix in chunks to save memory."""
    chunk_size = 5000  # Smaller chunks for sparse matrix construction
    rows = []
    cols = []
    data = []
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        for k, (w, b) in enumerate(zip(chunk.white_idx, chunk.black_idx)):
            # White player (1)
            rows.append(i + k)
            cols.append(w)
            data.append(1)
            # Black player (-1)
            rows.append(i + k)
            cols.append(b)
            data.append(-1)
        gc.collect()
    
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(df), n_players))

X_train = make_sparse_design_matrix(train_df, n_players)
y_train = train_df.y.values.astype(np.int8)
del train_df
gc.collect()

X_test = make_sparse_design_matrix(test_df, n_players)
y_test = test_df.y.values.astype(np.int8)
del test_df
gc.collect()

print("Training baseline model...")
clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    fit_intercept=False,
    solver='lbfgs',
    max_iter=1000,
    n_jobs=-1
)
clf.fit(X_train, y_train)
player_scores = clf.coef_.ravel().astype(np.float32)

print("Computing schedule strengths...")
sum_opp = np.zeros(n_players, dtype=np.float32)
count_opp = np.zeros(n_players, dtype=np.int32)

# Process in chunks using sparse matrix
chunk_size = 5000
for i in range(0, X_train.shape[0], chunk_size):
    chunk = X_train[i:i + chunk_size]
    # Get non-zero elements
    rows, cols = chunk.nonzero()
    data = chunk.data
    
    # Update sums and counts
    for r, c, v in zip(rows, cols, data):
        if v == 1:  # White player
            sum_opp[c] += player_scores[cols[rows == r][1]]  # Get black player's score
            count_opp[c] += 1
        else:  # Black player
            sum_opp[c] += player_scores[cols[rows == r][0]]  # Get white player's score
            count_opp[c] += 1
    gc.collect()

sched_strength = (sum_opp / np.maximum(count_opp, 1)).astype(np.float32)

print("Building augmented features...")
# Process in chunks using sparse matrix
chunk_size = 5000
X_aug_train = []
y_train_chunks = []

for i in range(0, X_train.shape[0], chunk_size):
    chunk = X_train[i:i + chunk_size]
    rows, cols = chunk.nonzero()
    data = chunk.data
    
    # Get white and black indices for each game
    white_idx = []
    black_idx = []
    for r in range(chunk_size):
        if i + r >= X_train.shape[0]:
            break
        game_cols = cols[rows == r]
        game_data = data[rows == r]
        white_idx.append(game_cols[game_data == 1][0])
        black_idx.append(game_cols[game_data == -1][0])
    
    white_idx = np.array(white_idx)
    black_idx = np.array(black_idx)
    
    delta_s = player_scores[white_idx] - player_scores[black_idx]
    delta_sched = sched_strength[white_idx] - sched_strength[black_idx]
    
    X_aug_train.append(np.vstack([delta_s, delta_sched]).T)
    y_train_chunks.append(y_train[i:i + chunk_size])
    gc.collect()

X_aug_train = np.vstack(X_aug_train)
y_train = np.concatenate(y_train_chunks)
del X_train, y_train_chunks
gc.collect()

print("Training augmented model...")
clf2 = LogisticRegression(
    penalty='l2',
    C=1.0,
    fit_intercept=True,
    solver='lbfgs',
    max_iter=1000,
    n_jobs=-1
)
clf2.fit(X_aug_train, y_train)

print("Evaluating models...")
# Process test set in chunks using sparse matrix
X_aug_test = []
y_test_chunks = []

for i in range(0, X_test.shape[0], chunk_size):
    chunk = X_test[i:i + chunk_size]
    rows, cols = chunk.nonzero()
    data = chunk.data
    
    # Get white and black indices for each game
    white_idx = []
    black_idx = []
    for r in range(chunk_size):
        if i + r >= X_test.shape[0]:
            break
        game_cols = cols[rows == r]
        game_data = data[rows == r]
        white_idx.append(game_cols[game_data == 1][0])
        black_idx.append(game_cols[game_data == -1][0])
    
    white_idx = np.array(white_idx)
    black_idx = np.array(black_idx)
    
    delta_s = player_scores[white_idx] - player_scores[black_idx]
    delta_sched = sched_strength[white_idx] - sched_strength[black_idx]
    
    X_aug_test.append(np.vstack([delta_s, delta_sched]).T)
    y_test_chunks.append(y_test[i:i + chunk_size])
    gc.collect()

X_aug_test = np.vstack(X_aug_test)
y_test = np.concatenate(y_test_chunks)

y_prob2 = clf2.predict_proba(X_aug_test)[:,1]
y_pred2 = (y_prob2 > 0.5).astype(np.int8)

# Compute metrics
n_test = len(y_test)
acc1 = accuracy_score(y_test, (clf.predict_proba(X_test)[:,1] > 0.5).astype(np.int8))
acc2 = accuracy_score(y_test, y_pred2)
se1 = np.sqrt(acc1 * (1-acc1) / n_test)
se2 = np.sqrt(acc2 * (1-acc2) / n_test)

print("\nResults:")
print(f'Baseline BT →    Acc = {acc1:.4f} ± {se1:.4f}, '
      f'AUC = {roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]):.4f}')
print(f'Augmented Model → Acc = {acc2:.4f} ± {se2:.4f}, AUC = {roc_auc_score(y_test, y_prob2):.4f}')

print('\nAugmented Model Weights:')
print(f'Player Strength Difference Weight: {clf2.coef_[0][0]:.4f}')
print(f'Schedule Strength Difference Weight: {clf2.coef_[0][1]:.4f}')
print(f'Intercept: {clf2.intercept_[0]:.4f}')

del X_test, y_test_chunks
gc.collect()

print("\nSaving results...")
# Save results in chunks
joblib.dump(clf2, 'bt_augmented_model_large.pkl')

# Save player data in chunks
chunk_size = 5000
for i in range(0, n_players, chunk_size):
    chunk = pd.DataFrame({
        'player_id': player_map.player_id[i:i + chunk_size],
        'bt_score': player_scores[i:i + chunk_size],
        'sched_strength': sched_strength[i:i + chunk_size],
    })
    chunk.to_csv('player_bt_with_schedule_large.csv', 
                mode='a' if i > 0 else 'w', 
                header=i == 0,
                index=False)
    gc.collect()

print('Saved augmented model and player schedule strengths.') 