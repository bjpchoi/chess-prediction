import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from dataclasses import dataclass
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GameData:
    white_idx: int
    black_idx: int
    result: int

class PlayerMapper:
    def __init__(self, player_map_path: str):
        self.player_map = pd.read_csv(player_map_path)
        self.n_players = self.player_map.shape[0]
        self.id2idx = dict(zip(self.player_map.player_id, self.player_map.matrix_index))
    
    def get_index(self, player_id: str) -> int:
        return self.id2idx[player_id]

class GameProcessor:
    def __init__(self, games_path: str, player_mapper: PlayerMapper):
        self.games = pd.read_csv(games_path)
        self.player_mapper = player_mapper
        self.processed_games: List[GameData] = []
    
    def process_games(self) -> pd.DataFrame:
        for _, game in self.games.iterrows():
            try:
                white_idx = self.player_mapper.get_index(game.white_id)
                black_idx = self.player_mapper.get_index(game.black_id)
                result = 1 if game.winner == 'white' else 0 if game.winner == 'black' else None
                if result is not None:
                    self.processed_games.append(GameData(white_idx, black_idx, result))
            except KeyError:
                continue
        
        return pd.DataFrame([
            (g.white_idx, g.black_idx, g.result) 
            for g in self.processed_games
        ], columns=['white_idx', 'black_idx', 'y'])

class FeatureBuilder:
    def __init__(self, n_players: int):
        self.n_players = n_players
    
    def build_design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = np.zeros((len(df), self.n_players), dtype=np.float32)
        for k, (i, j) in enumerate(zip(df.white_idx, df.black_idx)):
            X[k, i] += 1
            X[k, j] -= 1
        return X

class ScheduleStrengthCalculator:
    def __init__(self, n_players: int):
        self.n_players = n_players
        self.sum_opp = np.zeros(n_players, dtype=np.float64)
        self.count_opp = np.zeros(n_players, dtype=np.int32)
    
    def calculate(self, train_df: pd.DataFrame, player_scores: np.ndarray) -> np.ndarray:
        for _, game in train_df.iterrows():
            i, j = game.white_idx, game.black_idx
            self.sum_opp[i] += player_scores[j]
            self.sum_opp[j] += player_scores[i]
            self.count_opp[i] += 1
            self.count_opp[j] += 1
        
        return self.sum_opp / np.maximum(self.count_opp, 1)

class ChessRatingModel:
    def __init__(self):
        self.baseline_model = LogisticRegression(
            penalty='l2',
            C=1.0,
            fit_intercept=False,
            solver='lbfgs',
            max_iter=1000
        )
        self.augmented_model = LogisticRegression(
            penalty='l2',
            C=1.0,
            fit_intercept=True,
            solver='lbfgs',
            max_iter=1000
        )
        self.player_scores = None
        self.schedule_strength = None
    
    def train_baseline(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.baseline_model.fit(X_train, y_train)
        self.player_scores = self.baseline_model.coef_.ravel()
    
    def train_augmented(self, X_aug_train: np.ndarray, y_train: np.ndarray) -> None:
        self.augmented_model.fit(X_aug_train, y_train)
    
    def build_augmented_features(self, X: np.ndarray, white_idx: np.ndarray, black_idx: np.ndarray) -> np.ndarray:
        delta_s = self.player_scores[white_idx] - self.player_scores[black_idx]
        delta_sched = self.schedule_strength[white_idx] - self.schedule_strength[black_idx]
        return np.vstack([delta_s, delta_sched]).T
    
    def evaluate(self, X_test: np.ndarray, X_aug_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, float]:
        y_prob1 = self.baseline_model.predict_proba(X_test)[:, 1]
        y_prob2 = self.augmented_model.predict_proba(X_aug_test)[:, 1]
        
        acc1 = accuracy_score(y_test, (y_prob1 > 0.5).astype(int))
        acc2 = accuracy_score(y_test, (y_prob2 > 0.5).astype(int))
        
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
        
        out = pd.DataFrame({
            'player_id': self.player_map.player_id,
            'bt_score': self.player_scores,
            'sched_strength': self.schedule_strength,
        })
        out.to_csv(scores_path, index=False)

def main():
    logger.info("Loading and processing data...")
    player_mapper = PlayerMapper('player_mapping.csv')
    game_processor = GameProcessor('games.csv', player_mapper)
    df = game_processor.process_games()
    
    logger.info("Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.y)
    
    logger.info("Building features...")
    feature_builder = FeatureBuilder(player_mapper.n_players)
    X_train = feature_builder.build_design_matrix(train_df)
    X_test = feature_builder.build_design_matrix(test_df)
    y_train = train_df.y.values
    y_test = test_df.y.values
    
    logger.info("Training baseline model...")
    model = ChessRatingModel()
    model.train_baseline(X_train, y_train)
    
    logger.info("Computing schedule strengths...")
    schedule_calc = ScheduleStrengthCalculator(player_mapper.n_players)
    model.schedule_strength = schedule_calc.calculate(train_df, model.player_scores)
    
    logger.info("Building augmented features...")
    X_aug_train = model.build_augmented_features(
        X_train, train_df.white_idx.values, train_df.black_idx.values
    )
    X_aug_test = model.build_augmented_features(
        X_test, test_df.white_idx.values, test_df.black_idx.values
    )
    
    logger.info("Training augmented model...")
    model.train_augmented(X_aug_train, y_train)
    
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
    saver.save('bt_augmented_model.pkl', 'player_bt_with_schedule.csv')
    logger.info('Saved augmented model and player schedule strengths.')

if __name__ == "__main__":
    main()
