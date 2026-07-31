<div align="center">

# 🔒 Fine-Grained Encrypted Traffic Classification Using Graph Neural Networks

### *Deep Learning Architecture for TLS/VPN/Tor Traffic Identification, Uncertainty Estimation, and Explainability*

[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-v2.3+-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pyg.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.104+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-v18.2+-61dafb.svg?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

</div>

---

## 📌 Executive Summary

With the near-universal adoption of modern transport layer encryption protocols (**TLS 1.3, HTTPS, QUIC, WireGuard, OpenVPN, Tor**), legacy network monitoring techniques reliant on **Deep Packet Inspection (DPI)** or unencrypted SNI/DNS inspection have become ineffective. 

This repository implements an end-to-end, production-grade engineering solution for **Fine-Grained Encrypted Traffic Classification** using **Graph Neural Networks (PyTorch Geometric GraphSAGE & Graph Attention Networks)**. 

By representing network communication sessions as graph nodes and constructing dynamic temporal edges based on shared IP subnets and inter-flow timing proximity, the model captures complex topological relationship patterns among encrypted flows. The system features:

- **Monte Carlo Dropout ($N=20$)** for predictive uncertainty measurement and out-of-distribution / unknown traffic isolation.
- **GNNExplainer** for transparent, feature-level decision attribution per classified flow.
- **FastAPI REST & WebSocket Backend** broadcasting real-time flow telemetry.
- **React.js Dark Theme Dashboard** featuring interactive `vis-network` graph topology visualization, Chart.js metrics, and flow explainability modals.

---

## 📚 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Machine Learning & Graph Technical Specification](#-machine-learning--graph-technical-specification)
- [Application Category Mapping](#-application-category-mapping)
- [Directory Structure](#-directory-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Method 1: Docker Compose (Recommended)](#method-1-docker-compose-recommended)
  - [Method 2: Manual Local Installation](#method-2-manual-local-installation)
- [CLI Training & Evaluation Scripts](#-cli-training--evaluation-scripts)
- [API Overview](#-api-overview)
- [Dashboard Features](#-dashboard-features)
- [Project Report & Documentation](#-project-report--documentation)


---

## ✨ Key Features

- 🧠 **Graph Neural Network Classification**: Hybrid **GraphSAGE** and **GAT** architectures with residual skip connections, multi-head attention, and batch normalization.
- 📊 **28 Statistical Flow Features**: Extracts packet rates, byte counts, IAT (Inter-Arrival Time) statistics, TCP flag distributions, and flow duration without decrypting payload data.
- 🌐 **Dynamic Graph Topology**: Models flows as 5-tuple nodes `(src_ip, src_port, dst_ip, dst_port, protocol)` linked by weighted edges representing temporal proximity $w_{ij} = \exp(-|\Delta t| / \tau)$ and shared IP subnets.
- 🎲 **Uncertainty & Unknown Detection**: Implements Monte Carlo Dropout forward sampling to compute predictive variance $\sigma^2$; flags flows with low confidence ($<0.65$) or high variance as `Unknown / Suspicious`.
- 🔍 **GNNExplainer Attributions**: Generates feature-level attribution scores per flow, highlighting the top factors driving each classification decision.
- ⚡ **Real-Time Live Streaming**: FastAPI WebSocket service (`/api/ws`) pushes live flow captures directly to the web dashboard.
- 🖥️ **Interactive Dark-Theme Dashboard**: Built with React.js, Lucide icons, Chart.js, and `vis-network` for topological graph exploration.
- 🐳 **Docker-Ready**: Orchestrated container deployment using Docker Compose for seamless single-command deployment.

---

## 🏗️ System Architecture

```
                               ┌─────────────────────────────────────────┐
                               │  Encrypted Network Traffic / PCAPs      │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │ 5-Tuple Extraction & Feature Engine     │
                               │ (28 Statistical Packet & Flow Metrics)   │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │ PyTorch Geometric Graph Builder         │
                               │  Nodes = Flows | Edges = Shared IP/Time │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │ EncryptedTrafficGNN Architecture        │
                               │ (GraphSAGE / GAT + Monte Carlo Dropout) │
                               └──────────┬───────────────────┬──────────┘
                                          │                   │
                                          ▼                   ▼
                     ┌──────────────────────────┐   ┌──────────────────────────┐
                     │ Confidence & Uncertainty │   │ GNNExplainer Module      │
                     │ Thresholding (<0.65)     │   │ (Feature Importance)     │
                     └────────────┬─────────────┘   └────────────┬─────────────┘
                                  │                              │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                               ┌─────────────────────────────────────────┐
                               │ FastAPI REST & WebSocket Backend Server │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │ React.js Interactive Dashboard          │
                               │ (Graph Canvas, Charts, Flow Modal)      │
                               └─────────────────────────────────────────┘
```

---

## 🔬 Machine Learning & Graph Technical Specification

### 1. Flow 5-Tuple & Feature Extraction
Every network conversation is identified by its unique 5-tuple:
$$\text{Flow} = (\text{src\_ip}, \text{src\_port}, \text{dst\_ip}, \text{dst\_port}, \text{protocol})$$

From each flow, **28 statistical features** are extracted:
1. `fwd_pkt_count`, `bwd_pkt_count`, `total_pkt_count`
2. `fwd_byte_count`, `bwd_byte_count`, `total_byte_count`
3. `pkt_len_min`, `pkt_len_max`, `pkt_len_mean`, `pkt_len_std`, `pkt_len_median`
4. `fwd_pkt_len_mean`, `bwd_pkt_len_mean`
5. `flow_duration`, `flow_bytes_per_sec`, `flow_pkts_per_sec`
6. `iat_mean`, `iat_std`, `iat_min`, `iat_max`
7. `tcp_flag_syn`, `tcp_flag_ack`, `tcp_flag_fin`, `tcp_flag_rst`, `tcp_flag_psh`, `tcp_flag_urg`
8. `payload_bytes_ratio`, `is_vpn_flag`

### 2. Graph Construction Formulation
- **Nodes ($V$)**: Individual flow feature vectors $x_i \in \mathbb{R}^{28}$.
- **Edges ($E$)**: An edge $e_{ij}$ is formed if flow $i$ and flow $j$ share a source IP, destination IP, or cross-talking relationship.
- **Edge Weight ($w_{ij}$)**: Calculated via exponential decay over flow duration difference $\Delta t$:
$$w_{ij} = \exp\left(-\frac{|\text{duration}_i - \text{duration}_j|}{\tau}\right) \quad (\text{where } \tau = 10.0)$$

### 3. Uncertainty Estimation via Monte Carlo Dropout
During evaluation, dropout layers remain active across $N=20$ stochastic forward passes:
$$\mu = \frac{1}{N}\sum_{s=1}^{N} P(y \mid X, W_s), \quad \sigma^2 = \frac{1}{N}\sum_{s=1}^{N} \left(P(y \mid X, W_s) - \mu\right)^2$$
If $\max(\mu) < 0.65$ or $\text{mean}(\sigma^2) > 0.08$, the flow is categorized as **`Unknown`**.

---

## 🏷️ Application Category Mapping

| Category | Description | Primary Ports | Protocol Characteristics |
| :--- | :--- | :--- | :--- |
| 🟣 **Streaming** | Netflix, YouTube, Spotify, Twitch | 443 | High byte count, sustained flow duration |
| 🔵 **Chat** | WhatsApp, Telegram, Signal | 443 | Low packet count, periodic short burst IAT |
| 🔷 **VoIP** | Zoom, Skype, Google Meet | 5060 / UDP | High packet rate, low payload variance |
| 🟢 **File Transfer** | FTP, SFTP, Mega, Google Drive | 21, 443 | High forward byte ratio, max packet sizes |
| 🟠 **Browsing** | HTTPS, General Web Navigation | 443 | Medium packet size, short flow duration |
| 🔴 **P2P** | BitTorrent, Peer-to-peer sharing | 6881-6889 | High connection concurrency, dual UDP/TCP |
| 💗 **Email** | IMAPS, SMTPS, POP3S | 993, 465 | Low packet rate, strict handshake sequence |
| 🔮 **VPN** | OpenVPN, WireGuard, IPSec | 1194, 500 | High byte throughput, encapsulated headers |
| ⚪ **Unknown** | Out-of-distribution / Anomaly | Various | High uncertainty score ($\sigma^2 > 0.08$) |

---

## 📂 Directory Structure

```
gnn-encrypted-traffic/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entrypoint & WS lifespan
│   │   ├── config.py            # Pydantic settings & threshold configs
│   │   ├── models/
│   │   │   ├── database.py      # SQLAlchemy schemas (TrafficFlow, Metrics)
│   │   │   └── schemas.py       # Pydantic validation models
│   │   ├── api/
│   │   │   ├── routes.py        # REST API endpoints (/health, /metrics, /flows, etc.)
│   │   │   └── websocket.py     # ConnectionManager for WebSocket streaming
│   │   ├── services/
│   │   │   ├── classifier.py    # GNN inference, MC Dropout & fallback service
│   │   │   ├── feature_extractor.py # Scapy PCAP reader & feature builder
│   │   │   ├── graph_builder.py # PyTorch Geometric Data object builder
│   │   │   └── explainer.py     # GNNExplainer feature attribution
│   │   ├── ml/
│   │   │   ├── models.py        # GraphSAGE & GAT PyG network architectures
│   │   │   ├── baselines.py     # Random Forest & MLP baseline implementations
│   │   │   ├── dataset.py       # PyG dataset loader & graph split generator
│   │   │   └── trainer.py       # Training loop with metrics computation
│   │   └── utils/
│   │       ├── preprocessing.py # Standard scaling & synthetic flow generator
│   │       └── evaluation.py    # Accuracy, F1-score & confusion matrix computation
│   ├── synthetic_data.py        # Dataset & pre-trained weight bootstrapper
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.js               # Main React application layout & tab state
│   │   ├── index.js             # React DOM root entry point
│   │   ├── styles/
│   │   │   └── global.css       # Sleek Dark Theme, glassmorphism, responsive grid
│   │   ├── components/
│   │   │   ├── Layout/          # Header, Sidebar, Footer
│   │   │   ├── Dashboard/       # GraphVisualization, ClassificationChart, Tables, Cards
│   │   │   ├── FlowDetail/      # FlowDetailModal, ExplainabilityPanel
│   │   │   └── Common/          # LoadingSpinner, ErrorBoundary
│   │   ├── services/            # Axios API client & WebSocket listener
│   │   ├── hooks/               # useWebSocket & useFlows React hooks
│   │   └── utils/               # Formatters & category badge utilities
│   ├── package.json
│   ├── Dockerfile
│   └── .env.example
├── ml/
│   ├── train.py                 # CLI model training entry point
│   ├── preprocess.py            # ISCX VPN-nonVPN raw PCAP preprocessor
│   ├── evaluate.py              # Baseline benchmark comparison runner
│   ├── inference.py             # Standalone CLI prediction tool
│   ├── config.yaml              # Hyperparameters & dataset configuration
│   └── notebooks/               # Jupyter research notebooks
├── docker/
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
│   └── nginx.conf
├── docker-compose.yml
├── README.md
├── INSTALLATION.md
├── API_DOCS.md
└── PROJECT_REPORT.md
```

---

## ⚡ Getting Started

### Prerequisites

- **Python**: `3.10` or higher
- **Node.js**: `18.x` or higher
- **Docker & Docker Compose**: (Optional, for containerized run)

---

### Method 1: Docker Compose (Recommended)

Run the complete stack (Backend + Frontend + Nginx) with a single command:

```bash
# 1. Navigate to project root directory
cd D:\Users\ANAMIKA\Documents\gnn-encrypted-traffic

# 2. Build and launch containers
docker-compose up --build
```

- 🌐 **React Dashboard**: [http://localhost:3000](http://localhost:3000)
- ⚡ **FastAPI REST API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 📡 **WebSocket Endpoint**: `ws://localhost:8000/api/ws`

---

### Method 2: Manual Local Installation

#### 1. Start the FastAPI Backend

```bash
# Navigate to backend folder
cd D:\Users\ANAMIKA\Documents\gnn-encrypted-traffic\backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Bootstrap synthetic dataset and train initial GNN model weights
python synthetic_data.py

# Launch FastAPI development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Start the React.js Dashboard

In a new terminal window:

```bash
# Navigate to frontend folder
cd D:\Users\ANAMIKA\Documents\gnn-encrypted-traffic\frontend

# Install Node modules
npm install

# Start React development server
npm start
```

Visit [http://localhost:3000](http://localhost:3000) in your web browser.

---

## 🧪 CLI Training & Evaluation Scripts

The `ml/` folder contains CLI commands for training models, preprocessing data, and evaluating performance:

```bash
# Train GraphSAGE GNN Model (50 Epochs)
python ml/train.py --model GraphSAGE --epochs 50 --lr 0.001

# Train Graph Attention Network (GAT) Model
python ml/train.py --model GAT --epochs 50 --lr 0.001

# Run Baseline Comparison (GraphSAGE vs GAT vs Random Forest)
python ml/evaluate.py

# Run Standalone Flow Prediction CLI
python ml/inference.py --category Streaming --is-vpn

# Preprocess Raw PCAP Directory or Generate Synthetic Benchmark
python ml/preprocess.py --samples 1000
```

---

## 📡 API Overview

### REST Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System health check and model loading status |
| `GET` | `/api/metrics` | Real-time KPIs, category distribution & protocol split |
| `GET` | `/api/flows` | List recent traffic flows with search, category filtering, & pagination |
| `GET` | `/api/flows/{id}` | Single flow metadata with GNNExplainer attributions |
| `POST` | `/api/classify` | Classify raw 5-tuple flow dictionary or PCAP snippet |
| `GET` | `/api/analytics` | Time-series traffic trends and bandwidth volume |
| `GET` | `/api/model/performance` | Model accuracy, F1-scores, per-class metrics & confusion matrix |

### WebSocket Endpoint
- **URL**: `ws://localhost:8000/api/ws`
- **Behavior**: Continuously streams `NEW_FLOW` events with live classification results as network traffic is captured.

---

## 🖥️ Dashboard Features

1. **Live Metrics Cards**: Total flows, VPN encapsulation ratio, average GNN confidence score, and unknown traffic count.
2. **Interactive Network Graph**: `vis-network` graph canvas illustrating flow node relationships, color-coded by application category.
3. **Category Breakdown Chart**: Chart.js doughnut chart rendering percentage distribution across application types.
4. **Traffic Volume Timeline**: Real-time traffic ingestion line graph.
5. **Flow Table & Filter**: Searchable traffic table supporting instant category filter pills and deep-dive inspection.
6. **Flow Detail Modal**: Displays complete 5-tuple metrics alongside **GNNExplainer feature attributions**, explaining *why* the neural network made a specific prediction.

---

## 📄 Project Report & Documentation

- [INSTALLATION.md](INSTALLATION.md): Detailed installation and troubleshooting guide.
- [API_DOCS.md](API_DOCS.md): Complete REST & WebSocket API specification.
- [PROJECT_REPORT.md](PROJECT_REPORT.md): Academic engineering report synopsis with research background, equations, and benchmark comparison tables.

