from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

# 5-Tuple Schema
class FlowTuple(BaseModel):
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: str

# Create Flow Payload
class FlowCreate(BaseModel):
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: str
    duration: float = 0.0
    total_packets: int = 1
    total_bytes: int = 64
    is_vpn: bool = False
    feature_vector: Optional[List[float]] = None

# Flow Details Response
class FlowResponse(BaseModel):
    id: int
    flow_id: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: str
    start_timestamp: datetime
    duration: float
    total_packets: int
    total_bytes: int
    is_vpn: bool
    predicted_category: str
    confidence: float
    uncertainty_score: float
    is_unknown: bool
    class_probabilities: Dict[str, float]
    explanation: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

# Flow List Response with Pagination
class FlowListResponse(BaseModel):
    total: int
    page: int
    limit: int
    flows: List[FlowResponse]

# Classify Request
class ClassifyRequest(BaseModel):
    src_ip: str = "192.168.1.105"
    src_port: int = 44332
    dst_ip: str = "142.250.190.46"
    dst_port: int = 443
    protocol: str = "TCP"
    duration: float = 12.5
    total_packets: int = 145
    total_bytes: int = 98230
    is_vpn: bool = False
    raw_pcap_b64: Optional[str] = None
    custom_features: Optional[List[float]] = None

# Classify Response
class ClassifyResponse(BaseModel):
    flow_id: str
    predicted_category: str
    confidence: float
    uncertainty_score: float
    is_unknown: bool
    class_probabilities: Dict[str, float]
    explanation: Dict[str, Any]

# Real-Time Dashboard Metrics Response
class MetricsResponse(BaseModel):
    total_flows: int
    classified_today: int
    active_connections: int
    vpn_ratio: float
    unknown_ratio: float
    avg_confidence: float
    category_distribution: Dict[str, int]
    protocol_distribution: Dict[str, int]
    top_talking_ips: List[Dict[str, Any]]
    confidence_stats: Dict[str, float]

# Analytics & Historical Data
class AnalyticsResponse(BaseModel):
    hourly_traffic: List[Dict[str, Any]]
    category_trends: List[Dict[str, Any]]
    vpn_vs_nonvpn: Dict[str, int]
    bandwidth_usage_mb: float

# Model Performance Response
class ModelPerformanceResponse(BaseModel):
    model_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    precision: float
    recall: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    classes: List[str]
