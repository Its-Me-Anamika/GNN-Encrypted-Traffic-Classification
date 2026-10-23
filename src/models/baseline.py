"""
Conventional Machine Learning Baselines for Flow Classification.
Implements Random Forest and Multi-Layer Perceptron (MLP) on raw tabular flow features
to answer the core research question: Does GNN improve over non-graph ML?
"""
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import BASELINE_RF_CONFIG, BASELINE_MLP_CONFIG

logger = logging.getLogger(__name__)


class TabularBaseline:
    """Manages training and evaluation of conventional tabular classifiers."""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type.lower()
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=BASELINE_RF_CONFIG["n_estimators"],
                max_depth=BASELINE_RF_CONFIG["max_depth"],
                random_state=BASELINE_RF_CONFIG["random_state"],
                class_weight="balanced",
            )
        elif self.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=BASELINE_MLP_CONFIG["hidden_layer_sizes"],
                max_iter=BASELINE_MLP_CONFIG["max_iter"],
                random_state=BASELINE_MLP_CONFIG["random_state"],
                early_stopping=True,
            )
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'random_forest' or 'mlp'.")

        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TabularBaseline":
        """Trains the baseline classifier."""
        if self.model_type == "mlp" and len(X_train) < 30:
            self.model.early_stopping = False
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        """Returns feature importances if supported by the model."""
        if self.model_type == "random_forest" and hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))
        return {}
