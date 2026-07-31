from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TrafficFlow(Base):
    __tablename__ = "traffic_flows"

    id = Column(Integer, primary_key=True, index=True)
    flow_id = Column(String(64), unique=True, index=True)
    
    # 5-Tuple Network Identifier
    src_ip = Column(String(45), index=True)
    src_port = Column(Integer)
    dst_ip = Column(String(45), index=True)
    dst_port = Column(Integer)
    protocol = Column(String(10), index=True)
    
    # Flow Meta
    start_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    duration = Column(Float)
    total_packets = Column(Integer)
    total_bytes = Column(Integer)
    is_vpn = Column(Boolean, default=False, index=True)
    
    # Statistical Features (JSON dictionary or JSON array)
    feature_vector = Column(JSON)
    
    # Classification Results
    predicted_category = Column(String(32), index=True)
    confidence = Column(Float)
    uncertainty_score = Column(Float)  # Monte Carlo Dropout Variance/Entropy
    is_unknown = Column(Boolean, default=False, index=True)
    
    # Class Probabilities (JSON dictionary: {Category: float})
    class_probabilities = Column(JSON)
    
    # GNN Attribution & Explainability (JSON dictionary)
    explanation = Column(JSON, nullable=True)

class ClassificationMetric(Base):
    __tablename__ = "classification_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    model_name = Column(String(50))
    accuracy = Column(Float)
    macro_f1 = Column(Float)
    weighted_f1 = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    per_class_metrics = Column(JSON)
    confusion_matrix = Column(JSON)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
