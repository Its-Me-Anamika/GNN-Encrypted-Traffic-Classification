# Fine-Grained Encrypted Traffic Classification using Graph Neural Networks (GraphSAGE & GAT)

[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-v2.3+-ee4c2c.svg)](https://pyg.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-v18.2+-61dafb.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end engineering system for classifying encrypted network traffic (TLS, VPN, Tor) using **PyTorch Geometric Graph Neural Networks (GraphSAGE & GAT)**. Includes Monte Carlo Dropout uncertainty estimation, GNNExplainer feature attributions, a real-time FastAPI backend, and an interactive React.js dashboard.

---

## 🌟 Key Features

1. **Graph Representation Learning**:
   - **Nodes**: Network flow 5-tuples `(src_ip, src_port, dst_ip, dst_port, protocol)`.
   - **Edges**: Shared IP relationships and temporal proximity edge weights $w_{ij} = \exp(-\Delta t / \tau)$.
   - **28 Statistical Features**: Packet rates, byte counts, IAT metrics, TCP flags, and flow duration.
2. **GNN Architectures & Uncertainty**:
   - **GraphSAGE** and **GAT (Graph Attention Network)** layers with skip connections.
   - **Monte Carlo Dropout ($N=20$)**: Measures predictive variance to isolate out-of-distribution / unknown traffic.
3. **Explainable AI (GNNExplainer)**:
   - Integrated feature attribution masks showing top factors driving flow classification.
4. **Real-Time React Dashboard**:
   - Interactive network topology canvas rendered with `vis-network`.
   - Live flow updates via WebSocket stream (`/api/ws`).
   - Dark theme glassmorphism UI with Chart.js analytics.
5. **Baselines & Evaluation**:
   - Built-in comparison against Random Forest and Multi-Layer Perceptron (MLP).

---

## 🚀 Quick Start Guide

### Option 1: Using Docker Compose (Recommended)

```bash
# Clone or navigate to the project directory
cd D:\Users\ANAMIKA\Documents\gnn-encrypted-traffic

# Build and start all services (Backend + Frontend)
docker-compose up --build
```
- **React Dashboard**: [http://localhost:3000](http://localhost:3000)
- **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option 2: Local Python & Node Setup

#### 1. Backend Setup (FastAPI & PyTorch Geometric)
```bash
cd backend

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Bootstrap synthetic dataset and train initial GNN weights
python synthetic_data.py

# Start FastAPI dev server
uvicorn app.main:app --reload --port 8000
```

#### 2. Frontend Setup (React.js)
```bash
cd frontend

# Install Node dependencies
npm install

# Start React dev server
npm start
```

---

## 📁 Project Architecture

```
gnn-encrypted-traffic/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app & WebSocket entry
│   │   ├── config.py            # Settings & Hyperparameters
│   │   ├── models/              # SQLAlchemy & Pydantic models
│   │   ├── api/                 # REST routes & WebSocket manager
│   │   ├── services/            # Classifier, GraphBuilder, GNNExplainer
│   │   ├── ml/                  # GraphSAGE & GAT models, Trainer
│   │   └── utils/               # Preprocessing & Evaluation metrics
│   ├── synthetic_data.py        # Bootstrapper script
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js               # Main React dashboard layout
│   │   ├── components/          # Network graph, charts, tables & modal
│   │   ├── services/            # Axios API & WebSocket client
│   │   └── styles/              # Dark mode design system
│   └── package.json
├── ml/
│   ├── train.py                 # CLI GNN training script
│   ├── preprocess.py            # ISCX dataset pipeline
│   ├── evaluate.py              # Baseline benchmark runner
│   └── notebooks/               # Jupyter exploration notebooks
└── docker-compose.yml
```

---

## 📊 Application Classes
The system classifies encrypted traffic into 9 fine-grained categories:
`Streaming`, `Chat`, `VoIP`, `File Transfer`, `Browsing`, `P2P`, `Email`, `VPN`, `Unknown`

---

## 📖 Further Documentation
- [INSTALLATION.md](INSTALLATION.md): Detailed installation instructions.
- [API_DOCS.md](API_DOCS.md): Full REST & WebSocket API specification.
- [PROJECT_REPORT.md](PROJECT_REPORT.md): Engineering project synopsis and research methodology.
