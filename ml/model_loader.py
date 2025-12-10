"""
SentinEV - Model Loader
=======================
Unified loader for ml_pipeline models (LSTM-AE, LightGBM, thresholds).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

# Try importing model libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Path to ml_pipeline models
ML_PIPELINE_DIR = Path(__file__).parent.parent / "ml_pipeline"
MODELS_DIR = ML_PIPELINE_DIR / "models"


class LSTMAutoencoder(nn.Module):
    """LSTM-based Autoencoder for anomaly detection (architecture from ml_pipeline)."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], latent_dim: int = 16):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)
        self.encoder_lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dims[1], latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dims[1])
        self.decoder_lstm1 = nn.LSTM(hidden_dims[1], hidden_dims[0], batch_first=True)
        self.decoder_lstm2 = nn.LSTM(hidden_dims[0], input_dim, batch_first=True)
    
    def encode(self, x):
        out, _ = self.encoder_lstm1(x)
        out, (h, c) = self.encoder_lstm2(out)
        latent = self.encoder_fc(h.squeeze(0))
        return latent
    
    def decode(self, latent, seq_len):
        out = self.decoder_fc(latent)
        out = out.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm1(out)
        out, _ = self.decoder_lstm2(out)
        return out
    
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent, x.size(1))
        return reconstructed


@dataclass
class LoadedModels:
    """Container for all loaded models."""
    failure_predictor: Optional[object] = None
    severity_classifier: Optional[object] = None
    lstm_autoencoder: Optional[object] = None
    lstm_config: Optional[Dict] = None
    thresholds: Optional[Dict] = None
    feature_columns: Optional[List[str]] = None
    device: str = "cpu"


class ModelLoader:
    """
    Unified model loader for ml_pipeline models.
    
    Loads:
    - LightGBM failure predictor
    - LightGBM severity classifier
    - LSTM Autoencoder for anomaly detection
    - Thresholds and feature columns
    """
    
    _instance: Optional['ModelLoader'] = None
    _models: Optional[LoadedModels] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._models is None:
            self._models = self._load_all_models()
    
    def _load_all_models(self) -> LoadedModels:
        """Load all models from ml_pipeline."""
        print("🔄 Loading ml_pipeline models...")
        
        models = LoadedModels()
        models.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        # Load LightGBM failure predictor
        failure_path = MODELS_DIR / "lgbm_failure_predictor.txt"
        if LIGHTGBM_AVAILABLE and failure_path.exists():
            models.failure_predictor = lgb.Booster(model_file=str(failure_path))
            print("  ✅ Loaded LightGBM failure predictor")
        else:
            print("  ⚠️ LightGBM failure predictor not found")
        
        # Load LightGBM severity classifier
        severity_path = MODELS_DIR / "lgbm_severity_classifier.txt"
        if LIGHTGBM_AVAILABLE and severity_path.exists():
            models.severity_classifier = lgb.Booster(model_file=str(severity_path))
            print("  ✅ Loaded LightGBM severity classifier")
        else:
            print("  ⚠️ LightGBM severity classifier not found")
        
        # Load LSTM Autoencoder
        lstm_path = MODELS_DIR / "lstm_ae_best.pt"
        lstm_config_path = MODELS_DIR / "lstm_ae_config.json"
        
        if TORCH_AVAILABLE and lstm_path.exists() and lstm_config_path.exists():
            with open(lstm_config_path, 'r') as f:
                models.lstm_config = json.load(f)
            
            # Create model architecture
            lstm_model = LSTMAutoencoder(
                input_dim=models.lstm_config['input_dim'],
                hidden_dims=models.lstm_config['hidden_dims'],
                latent_dim=models.lstm_config['latent_dim']
            )
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=models.device))
            lstm_model.to(models.device)
            lstm_model.eval()
            models.lstm_autoencoder = lstm_model
            print("  ✅ Loaded LSTM Autoencoder")
        else:
            print("  ⚠️ LSTM Autoencoder not found")
        
        # Load thresholds
        thresholds_path = MODELS_DIR / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                models.thresholds = json.load(f)
            print(f"  ✅ Loaded thresholds")
        else:
            models.thresholds = {
                'failure_predictor': {'high_precision_threshold': 0.5},
                'anomaly_detector': {'threshold': 0.95}
            }
            print("  ⚠️ Using default thresholds")
        
        # Load feature columns
        features_path = MODELS_DIR / "feature_columns.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                models.feature_columns = json.load(f)
            print(f"  ✅ Loaded {len(models.feature_columns)} feature columns")
        else:
            models.feature_columns = []
            print("  ⚠️ Feature columns not found")
        
        print("✅ Model loading complete")
        return models
    
    @property
    def failure_predictor(self):
        return self._models.failure_predictor
    
    @property
    def severity_classifier(self):
        return self._models.severity_classifier
    
    @property
    def lstm_autoencoder(self):
        return self._models.lstm_autoencoder
    
    @property
    def lstm_config(self):
        return self._models.lstm_config
    
    @property
    def thresholds(self):
        return self._models.thresholds
    
    @property
    def feature_columns(self):
        return self._models.feature_columns
    
    @property
    def device(self):
        return self._models.device
    
    def predict_failure(self, features: np.ndarray) -> Tuple[float, bool]:
        """
        Predict failure probability using LightGBM.
        
        Args:
            features: Feature array matching feature_columns
            
        Returns:
            (probability, is_failure_predicted)
        """
        if self.failure_predictor is None:
            return 0.0, False
        
        proba = self.failure_predictor.predict(features.reshape(1, -1))[0]
        threshold = self.thresholds.get('failure_predictor', {}).get('high_precision_threshold', 0.5)
        return float(proba), bool(proba >= threshold)
    
    def predict_severity(self, features: np.ndarray) -> str:
        """
        Predict severity class using LightGBM.
        
        Args:
            features: Feature array
            
        Returns:
            Severity string: 'low', 'medium', 'high', 'critical'
        """
        if self.severity_classifier is None:
            return "low"
        
        proba = self.severity_classifier.predict(features.reshape(1, -1))[0]
        severity_idx = int(np.argmax(proba))
        severity_map = {0: "low", 1: "medium", 2: "high", 3: "critical"}
        return severity_map.get(severity_idx, "low")
    
    def compute_anomaly_score(self, sequence: np.ndarray) -> float:
        """
        Compute anomaly score using LSTM Autoencoder.
        
        Args:
            sequence: Shape (seq_len, features) or (batch, seq_len, features)
            
        Returns:
            Reconstruction error (anomaly score)
        """
        if self.lstm_autoencoder is None or not TORCH_AVAILABLE:
            return 0.0
        
        # Ensure 3D input
        if len(sequence.shape) == 2:
            sequence = sequence[np.newaxis, :, :]
        
        with torch.no_grad():
            x = torch.FloatTensor(sequence).to(self.device)
            reconstructed = self.lstm_autoencoder(x)
            error = ((x - reconstructed) ** 2).mean().item()
        
        return error
    
    def is_anomaly(self, anomaly_score: float) -> bool:
        """Check if score exceeds anomaly threshold."""
        threshold = self.lstm_config.get('threshold', 0.5) if self.lstm_config else 0.5
        return anomaly_score > threshold


# Singleton accessor
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get or create the singleton ModelLoader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def reload_models():
    """Force reload all models."""
    global _model_loader
    _model_loader = None
    return get_model_loader()


if __name__ == "__main__":
    # Test model loading
    print("\n" + "=" * 50)
    print("Testing Model Loader")
    print("=" * 50)
    
    loader = get_model_loader()
    
    print(f"\nModels loaded:")
    print(f"  Failure predictor: {loader.failure_predictor is not None}")
    print(f"  Severity classifier: {loader.severity_classifier is not None}")
    print(f"  LSTM Autoencoder: {loader.lstm_autoencoder is not None}")
    print(f"  Device: {loader.device}")
    print(f"  Feature columns: {len(loader.feature_columns)}")
    
    # Test predictions
    if loader.feature_columns:
        test_features = np.zeros(len(loader.feature_columns))
        prob, is_failure = loader.predict_failure(test_features)
        severity = loader.predict_severity(test_features)
        print(f"\nTest prediction (zeros):")
        print(f"  Failure prob: {prob:.4f}, Is failure: {is_failure}")
        print(f"  Severity: {severity}")
