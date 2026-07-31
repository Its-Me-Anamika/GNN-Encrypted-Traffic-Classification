# REST & WebSocket API Specifications

The FastAPI backend exposes endpoints for flow classification, real-time metrics, analytics, model performance, and live WebSocket streaming.

---

## 🟢 Health Check
### `GET /api/health`
- **Response**:
```json
{
  "status": "healthy",
  "service": "Fine-Grained Encrypted Traffic Classification using GNNs",
  "version": "1.0.0",
  "model_loaded": true,
  "model_type": "GraphSAGE",
  "timestamp": "2026-07-31T14:40:00"
}
```

---

## 📊 Real-Time Metrics
### `GET /api/metrics`
- **Response**:
```json
{
  "total_flows": 120,
  "classified_today": 120,
  "active_connections": 24,
  "vpn_ratio": 0.18,
  "unknown_ratio": 0.05,
  "avg_confidence": 0.942,
  "category_distribution": {
    "Streaming": 34,
    "Browsing": 22,
    "VPN": 18,
    "Chat": 14,
    "VoIP": 10,
    "File Transfer": 8,
    "P2P": 6,
    "Email": 5,
    "Unknown": 3
  },
  "protocol_distribution": { "TCP": 85, "UDP": 35 }
}
```

---

## 🔍 Traffic Flows
### `GET /api/flows?page=1&limit=50&category=Streaming`
- **Query Params**: `page` (default 1), `limit` (default 50), `category`, `is_vpn`, `search`

### `GET /api/flows/{flow_id}`
- **Response**: Includes full GNNExplainer feature importance breakdown and connected sub-graph neighbors.

---

## ⚡ Classification Endpoint
### `POST /api/classify`
- **Request Body**:
```json
{
  "src_ip": "192.168.1.105",
  "src_port": 44332,
  "dst_ip": "142.250.190.46",
  "dst_port": 443,
  "protocol": "TCP",
  "duration": 12.5,
  "total_packets": 145,
  "total_bytes": 98230,
  "is_vpn": false
}
```
- **Response**:
```json
{
  "flow_id": "flow_849201",
  "predicted_category": "Streaming",
  "confidence": 0.965,
  "uncertainty_score": 0.012,
  "is_unknown": false,
  "class_probabilities": {
    "Streaming": 0.965,
    "Browsing": 0.02,
    "Chat": 0.005
  },
  "explanation": {
    "top_features": [
      { "feature_name": "fwd_byte_count", "importance_score": 38.5 },
      { "feature_name": "flow_duration", "importance_score": 24.1 }
    ]
  }
}
```

---

## 📡 Live WebSocket Feed
### `WS /api/ws`
- Connection broadcasts `NEW_FLOW` messages continuously as live flows are captured and processed by the GNN model.
