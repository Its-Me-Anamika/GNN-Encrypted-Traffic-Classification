import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Fine-Grained Encrypted Traffic Classification using GNNs"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Environment & Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-key-change-in-production")
    API_KEY: str = os.getenv("API_KEY", "gnn-encrypted-traffic-key-2026")
    
    # CORS Settings
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "*"
    ]
    
    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./encrypted_traffic.db")
    
    # ML & Model Settings
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "GraphSAGE")  # "GraphSAGE" or "GAT"
    MODEL_PATH: str = os.getenv("MODEL_PATH", "backend/app/ml/saved_models/gnn_traffic_model.pt")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "backend/app/ml/saved_models/feature_scaler.pkl")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
    MC_DROPOUT_SAMPLES: int = int(os.getenv("MC_DROPOUT_SAMPLES", "20"))
    
    # Application Categories
    CLASSES: list = [
        "Streaming",
        "Chat",
        "VoIP",
        "File Transfer",
        "Browsing",
        "P2P",
        "Email",
        "VPN",
        "Unknown"
    ]
    
    # Feature Configuration
    NUM_FEATURES: int = 28

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
