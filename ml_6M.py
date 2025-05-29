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
from dataclasses import dataclass
from typing import Tuple, Dict, List, Iterator, Optional
import logging
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GameData:
    white_idx: int
    black_idx: int
    result: int

class MemoryEfficientDataLoader:
    def __init__(self, file_path: str, chunk_size: int = 25000):
        self.file_path = file_path
        self.chunk_size = chunk_size
    
    def get_chunks(self) -> Iterator[pd.DataFrame]:
        return pd.read_csv(self.file_path, chunksize=self.chunk_size)
    
    def count_total_games(self) -> int:
        return sum(len(chunk) for chunk in self.get_chunks())

class PlayerMapper:
    def __init__(self):
        self.player_map: Optional[pd.DataFrame] = None
        self.n_players: int = 0
        self.id2idx: Dict[str, int] = {}
    
    def build_from_games(self, games_path: str, chunk_size: int = 25000) -> None:
        logger.info("Building player mapping from games...")
        all_players = set()
        loader = MemoryEfficientDataLoader(games_path, chunk_size)
        
        for chunk in tqdm(loader.get_chunks(), desc="Collecting players"):
            all_players.update(chunk['White'].unique())
            all_players.update(chunk['Black'].unique())
            gc.collect()
        
        self.player_map = pd.DataFrame({
            'player_id': list(all_players),
            'matrix_index': np.arange(len(all_players))
        })
        self.n_players = len(all_players)
        self.id2idx = dict(zip(self.player_map.player_id, self.player_map.matrix_index))
        del all_players
        gc.collect()
        
        logger.info(f"Found {self.n_players} unique players")

class GameProcessor:
    def __init__(self, games_path: str, player_mapper: PlayerMapper, chunk_size: int = 25000):
        self.games_path = games_path
        self.player_mapper = player_mapper
        self.chunk_size = chunk_size
        self.processed_games: List[GameData] = []
    
    def process_games(self) -> pd.DataFrame:
        logger.info("Processing games...")
        loader = MemoryEfficientDataLoader(self.games_path, self.chunk_size)
        
        for chunk in tqdm(loader.get_chunks(), desc="Processing games"):
            for _, game in chunk.iterrows():
                try:
                    white_idx = self.player_mapper.id2idx[game.White]
                    black_idx = self.player_mapper.id2idx[game.Black]
                    result = 1 if game.Result == '1-0' else 0 if game.Result == '0-1' else None
                    if result is not None:
                        self.processed_games.append(GameData(white_idx, black_idx, result))
                except KeyError:
                    continue
            gc.collect()
        
        return pd.DataFrame([
            (g.white_idx, g.black_idx, g.result) 
            for g in self.processed_games
        ], columns=['white_idx', 'black_idx', 'y'])

class SparseFeatureBuilder:
    def __init__(self, n_players: int):
        self.n_players = n_players
    
    def build_sparse_matrix(self, df: pd.DataFrame) -> sparse.csr_matrix:
        rows = []
        cols = []
        data = []
        
        for k, (i, j) in enumerate(zip(df.white_idx, df.black_idx)):
            rows.extend([k, k])
            cols.extend([i, j])
            data.extend([1, -1])
        
        return sparse.csr_matrix(
            (data, (rows, cols)), 
            shape=(len(df), self.n_players)
        )

class ScheduleStrengthCalculator:
    def __init__(self, n_players: int):
        self.n_players = n_players
        self.sum_opp = np.zeros(n_players, dtype=np.float32)
        self.count_opp = np.zeros(n_players, dtype=np.int32)
    
    def calculate(self, X_train: sparse.csr_matrix, player_scores: np.ndarray) -> np.ndarray:
        chunk_size = 5000
        for i in range(0, X_train.shape[0], chunk_size):
            chunk = X_train[i:i + chunk_size]
            rows, cols = chunk.nonzero()
            data = chunk.data
            
            for r, c, v in zip(rows, cols, data):
                if v == 1:  # White player
                    self.sum_opp[c] += player_scores[cols[rows == r][1]]
                    self.count_opp[c] += 1
                else:  # Black player
                    self.sum_opp[c] += player_scores[cols[rows == r][0]]
                    self.count_opp[c] += 1
            gc.collect()
        
        return (self.sum_opp / np.maximum(self.count_opp, 1)).astype(np.float32)

class ChessRatingModel:
    def __init__(self):
        self.baseline_model = LogisticRegression(
            penalty='l2',
            C=1.0,
            fit_intercept=False,
            solver='lbfgs',
            max_iter=1000,
            n_jobs=-1
        )
        self.augmented_model = LogisticRegression(
            penalty='l2',
            C=1.0,
            fit_intercept=True,
            solver='lbfgs',
            max_iter=1000,
            n_jobs=-1
        )
        self.player_scores = None
        self.schedule_strength = None
    
    def train_baseline(self, X_train: sparse.csr_matrix, y_train: np.ndarray) -> None:
        self.baseline_model.fit(X_train, y_train)
        self.player_scores = self.baseline_model.coef_.ravel().astype(np.float32)
    
    def train_augmented(self, X_aug_train: np.ndarray, y_train: np.ndarray) -> None:
        self.augmented_model.fit(X_aug_train, y_train)
    
    def build_augmented_features(self, X: sparse.csr_matrix) -> np.ndarray:
        chunk_size = 5000
        X_aug = []
        
        for i in range(0, X.shape[0], chunk_size):
            chunk = X[i:i + chunk_size]
            rows, cols = chunk.nonzero()
            data = chunk.data
            
            white_idx = []
            black_idx = []
            for r in range(chunk_size):
                if i + r >= X.shape[0]:
                    break
                game_cols = cols[rows == r]
                game_data = data[rows == r]
                white_idx.append(game_cols[game_data == 1][0])
                black_idx.append(game_cols[game_data == -1][0])
            
            white_idx = np.array(white_idx)
            black_idx = np.array(black_idx)
            
            delta_s = self.player_scores[white_idx] - self.player_scores[black_idx]
            delta_sched = self.schedule_strength[white_idx] - self.schedule_strength[black_idx]
            
            X_aug.append(np.vstack([delta_s, delta_sched]).T)
            gc.collect()
        
        return np.vstack(X_aug)
    
    def evaluate(self, X_test: sparse.csr_matrix, X_aug_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, float]:
        y_prob1 = self.baseline_model.predict_proba(X_test)[:, 1]
        y_prob2 = self.augmented_model.predict_proba(X_aug_test)[:, 1]
        
        acc1 = accuracy_score(y_test, (y_prob1 > 0.5).astype(np.int8))
        acc2 = accuracy_score(y_test, (y_prob2 > 0.5).astype(np.int8))
        
        n_test = len(y_test)
        se1 = np.sqrt(acc1 * (1-acc1) / n_test)
        se2 = np.sqrt(acc2 * (1-acc2) / n_test)
        
        return acc1, se1, acc2, se2

class ResultSaver:
    def __init__(self, player_map: pd.DataFrame, player_scores: np.ndarray, schedule_strength: np.ndarray, model: LogisticRegression):
        self.player_map = player_map
        self.player_scores = player_scores
        self.schedule_strength = schedule_strength
        self.model = model
    
    def save(self, model_path: str, scores_path: str) -> None:
        joblib.dump(self.model, model_path)
        
        chunk_size = 5000
        for i in range(0, len(self.player_map), chunk_size):
            chunk = pd.DataFrame({
                'player_id': self.player_map.player_id[i:i + chunk_size],
                'bt_score': self.player_scores[i:i + chunk_size],
                'sched_strength': self.schedule_strength[i:i + chunk_size],
            })
            chunk.to_csv(
                scores_path,
                mode='a' if i > 0 else 'w',
                header=i == 0,
                index=False
            )
            gc.collect()

def main():
    logger.info("Initializing player mapper...")
    player_mapper = PlayerMapper()
    player_mapper.build_from_games('chess_games.csv')
    
    logger.info("Processing games...")
    game_processor = GameProcessor('chess_games.csv', player_mapper)
    df = game_processor.process_games()
    
    logger.info("Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.y)
    del df
    gc.collect()
    
    logger.info("Building feature matrices...")
    feature_builder = SparseFeatureBuilder(player_mapper.n_players)
    X_train = feature_builder.build_sparse_matrix(train_df)
    X_test = feature_builder.build_sparse_matrix(test_df)
    y_train = train_df.y.values.astype(np.int8)
    y_test = test_df.y.values.astype(np.int8)
    del train_df, test_df
    gc.collect()
    
    logger.info("Training baseline model...")
    model = ChessRatingModel()
    model.train_baseline(X_train, y_train)
    
    logger.info("Computing schedule strengths...")
    schedule_calc = ScheduleStrengthCalculator(player_mapper.n_players)
    model.schedule_strength = schedule_calc.calculate(X_train, model.player_scores)
    
    logger.info("Building augmented features...")
    X_aug_train = model.build_augmented_features(X_train)
    X_aug_test = model.build_augmented_features(X_test)
    del X_train
    gc.collect()
    
    logger.info("Training augmented model...")
    model.train_augmented(X_aug_train, y_train)
    del y_train
    gc.collect()
    
    logger.info("Evaluating models...")
    acc1, se1, acc2, se2 = model.evaluate(X_test, X_aug_test, y_test)
    
    logger.info("\nResults:")
    logger.info(f'Baseline BT →    Acc = {acc1:.4f} ± {se1:.4f}, '
                f'AUC = {roc_auc_score(y_test, model.baseline_model.predict_proba(X_test)[:,1]):.4f}')
    logger.info(f'Augmented Model → Acc = {acc2:.4f} ± {se2:.4f}, '
                f'AUC = {roc_auc_score(y_test, model.augmented_model.predict_proba(X_aug_test)[:,1]):.4f}')
    
    logger.info('\nAugmented Model Weights:')
    logger.info(f'Player Strength Difference Weight: {model.augmented_model.coef_[0][0]:.4f}')
    logger.info(f'Schedule Strength Difference Weight: {model.augmented_model.coef_[0][1]:.4f}')
    logger.info(f'Intercept: {model.augmented_model.intercept_[0]:.4f}')
    
    logger.info("\nSaving results...")
    saver = ResultSaver(
        player_mapper.player_map, 
        model.player_scores, 
        model.schedule_strength,
        model.augmented_model
    )
    saver.save('bt_augmented_model_large.pkl', 'player_bt_with_schedule_large.csv')
    logger.info('Saved augmented model and player schedule strengths.')

if __name__ == "__main__":
    main() 
