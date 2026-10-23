# Fine-Grained Encrypted Traffic Classification using Graph Neural Networks (GraphSAGE/GAT)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20Scapy%20%7C%20NetworkX-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-Operational%20Research%20Pipeline-brightgreen.svg)]()

A research project that classifies encrypted network traffic into fine-grained application categories (video streaming, VoIP, file transfer, web browsing, messaging, etc.) and analyzes VPN vs Non-VPN encapsulation **without decrypting network payloads**.

The system extracts statistical bidirectional flow metadata using **Scapy**, models communication relationships as a **Flow Graph** (via host-interaction and temporal correlation), and trains an authentic **GraphSAGE** (Hamilton et al., NeurIPS 2017) inductive model and **GAT** (Veličković et al., ICLR 2018) alongside conventional machine learning baselines (Random Forest, MLP).

---

## 1. Research Objectives & Core Questions

1. **Zero Payload Decryption**: Maintain end-user privacy and compatibility with TLS 1.3 / QUIC by inspecting only packet-header timing and sizing metadata.
2. **Beyond Binary Classification**: VPN vs Non-VPN is evaluated as Experiment 1; the primary research goal is fine-grained multi-class application categorization.
3. **Core Research Question**: *Does contextual graph structure (modeling concurrent sessions, shared endpoints, and temporal bursts) provide statistically significant accuracy and macro-F1 improvements over isolated flow tabular classifiers?*
4. **Generalization & Leakage Prevention**: Enforce strict train/test separation and evaluate host-disjoint splits to ensure the model learns generalized behavioral dynamics rather than memorizing IP addresses.

---

## 2. Pipeline Architecture

```
[ Real Labeled PCAPs / PCAPNG ]
            │
            ▼  (src/flow_extractor.py)
[ Scapy Streaming Flow Extractor ] ──► Bidirectional 5-Tuple Tracking
            │                          (Packets, Bytes, IATs, Sizes, TCP Flags)
            ▼
[ 26 Statistical Features ]
            │
            ├───► [ Tabular Baselines ] ──► Random Forest / MLP Classifier
            │
            ▼  (src/graph_builder.py)
[ Flow-Level Graph Construction ] ──► Nodes: Flows | Edges: Endpoint Sharing / Temporal
            │                          Zero Label Leakage Enforced
            ▼
[ Graph Neural Networks ] ─────────► 1. GraphSAGE (Mean Aggregator + L2-Norm)
                                       2. GAT (Multi-Head Attention)
            │
            ▼  (src/evaluator.py)
[ Comprehensive Evaluation ] ───────► Accuracy, Precision, Recall, Macro-F1,
                                       Confusion Matrix, Comparative Summary Table
```

---

## 3. Mathematical Formulation

### 3.1 GraphSAGE (Hamilton et al., 2017)
For node $v \in \mathcal{V}$ at layer $k$:
1. **Neighborhood Mean Aggregation**:
   $$h_{\mathcal{N}(v)}^{(k)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}$$
2. **Concatenation & Linear Transformation**:
   $$z_v^{(k)} = \sigma\left(W^{(k)} \cdot [h_v^{(k-1)} \,\|\, h_{\mathcal{N}(v)}^{(k)}] + b^{(k)}\right)$$
3. **L2 Normalization**:
   $$h_v^{(k)} = \frac{z_v^{(k)}}{\|z_v^{(k)}\|_2}$$

### 3.2 Graph Attention Network (Veličković et al., 2018)
Self-attention over neighbor interactions:
$$\alpha_{uv} = \frac{\exp\left(\text{LeakyReLU}\left(\vec{a}^T [W h_u \,\|\, W h_v]\right)\right)}{\sum_{k \in \mathcal{N}(v)} \exp\left(\text{LeakyReLU}\left(\vec{a}^T [W h_k \,\|\, W h_v]\right)\right)}$$
$$h_v' = \sigma\left(\frac{1}{K} \sum_{k=1}^K \sum_{u \in \mathcal{N}(v)} \alpha_{uv}^k W^k h_u\right)$$

---

## 4. Extracted Flow Metadata Features (26 Dimensions)

| Feature Group | Features Extracted |
|---|---|
| **Packet Counts** | `total_packets`, `fwd_packets`, `bwd_packets` |
| **Byte Counts** | `total_bytes`, `fwd_bytes`, `bwd_bytes` |
| **Duration & Rates** | `duration_sec`, `packet_rate_per_sec`, `byte_rate_per_sec` |
| **Packet Size Distribution** | `pkt_size_min`, `pkt_size_max`, `pkt_size_mean`, `pkt_size_std` |
| **Inter-Arrival Times (IAT)** | `iat_min`, `iat_max`, `iat_mean`, `iat_std` |
| **Directional Ratios** | `fwd_bwd_packet_ratio`, `fwd_bwd_byte_ratio` |
| **TCP Control Flags** | `flag_syn_count`, `flag_ack_count`, `flag_fin_count`, `flag_rst_count`, `flag_psh_count`, `flag_urg_count` |
| **Transport Protocol** | One-hot encoded `TCP`, `UDP`, `ICMP`, `OTHER` |

---

## 5. Public Benchmark Datasets

The system directly supports the two canonical research benchmark datasets:

### 1. ISCXVPN2016 (UNB / Canadian Institute for Cybersecurity)
- **URL**: [https://www.unb.ca/cic/datasets/vpn.html](https://www.unb.ca/cic/datasets/vpn.html)
- **Size**: ~28 GB PCAP files.
- **Classes**:
  - Binary: `vpn` vs `non_vpn`
  - Multi-Class Categories: `browsing`, `streaming`, `chat`, `voip`, `file_transfer`, `mail`, `vpn_streaming`, `vpn_chat`, `vpn_voip`, `vpn_file_transfer`.

### 2. USTC-TFC2016
- **URL**: [https://github.com/yungshenglu/USTC-TFC2016](https://github.com/yungshenglu/USTC-TFC2016)
- **Size**: ~3.7 GB PCAP files.
- **Classes**: 10 benign encrypted applications (`Facetime`, `FTP`, `Gmail`, `MySQL`, `Outlook`, `Skype`, `SMB`, `Weibo`, `WorldOfWarcraft`, `YouTube`).

### How to Add Data to This Project
* Place PCAPs into `data/raw_pcaps/<application_name>/` (e.g. `data/raw_pcaps/youtube/*.pcap`).
* Or for binary experiments, place into `data/raw_pcaps/vpn/` and `data/raw_pcaps/non_vpn/`.

---

## 6. Project Structure

```
encrypted_traffic_gnn_demo/
├── data/
│   ├── raw_pcaps/            # Place real PCAPs here by class or VPN folder
│   ├── processed/            # Extracted flows_dataset.csv
│   └── test_samples/         # Explicitly marked SYNTHETIC test PCAPs for testing
├── outputs/
│   ├── saved_models/         # PyTorch checkpoints (.pt)
│   ├── cm_*.png              # Confusion matrix heatmaps
│   ├── metrics_*.json        # Detailed quantitative metrics
│   └── model_comparison_*.csv # Tabular vs GNN comparison
├── src/
│   ├── config.py             # Feature definitions, paths, hyperparameters
│   ├── flow_extractor.py     # Scapy bidirectional flow parser
│   ├── graph_builder.py      # Flow graph construction & leakage-free splits
│   ├── dataset_manager.py    # PCAP discovery, label mapping, dataset catalog
│   ├── evaluator.py          # Metrics, confusion matrices, comparison tables
│   ├── pipeline.py           # Unified experiment orchestrator
│   └── models/
│       ├── baseline.py       # Random Forest and MLP classifiers
│       ├── graphsage.py      # Authentic GraphSAGE (Hamilton et al. 2017)
│       └── gat.py            # Graph Attention Network (Veličković et al. 2018)
├── tests/
│   ├── test_flow_extractor.py # Tests Scapy parsing & flow statistics
│   ├── test_graph_builder.py  # Tests graph topology & zero leakage
│   └── test_models.py         # Tests GraphSAGE, GAT, and baselines
├── ui/                       # UI assets
├── app.py                    # Demonstration interface (Native Windows GUI & CLI)
├── run_project.bat           # 1-Click Windows Batch Launcher
├── run_project.ps1           # 1-Click Windows PowerShell Launcher
├── requirements.txt          # Python dependencies
└── README.md                 # Complete documentation
```

---

## 7. Installation & Quick Start

### Step 1: Install Dependencies
Open PowerShell or Command Prompt in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: Run Using the Windows Script
Double-click `run_project.bat` or run in PowerShell:
```powershell
.\run_project.ps1
```

### Step 3: Command-Line Execution
* **Experiment 1 (Fine-Grained Application Classification)**:
  ```bash
  python -m src.pipeline --experiment fine_grained_app
  ```
* **Experiment 2 (Binary VPN vs Non-VPN Classification)**:
  ```bash
  python -m src.pipeline --experiment binary_vpn
  ```
* **Run Unit Tests**:
  ```bash
  python -m unittest discover tests -v
  ```
* **Launch GUI Demonstration**:
  ```bash
  python app.py
  ```
* **Launch CLI Demonstration on a PCAP**:
  ```bash
  python app.py --cli --pcap "path/to/traffic.pcap"
  ```

---

## 8. Scientific Honesty & Validation Protocols

1. **No Synthetic Misrepresentation**: If real PCAPs are not yet downloaded into `data/raw_pcaps/`, the pipeline automatically executes in self-test mode with explicitly labeled `SYNTHETIC_PIPELINE_VERIFICATION.pcap` so all code and dimensions can be verified. Such runs are explicitly flagged `DATA MODE: SYNTHETIC` in outputs and console logs.
2. **Zero Label Leakage**: Graph edges are computed strictly using host-interaction endpoints, timestamps, or unsupervised feature similarity. Labels $y$ are never visible during graph topology generation.
3. **Exact Same Train/Test Splits**: Random Forest, MLP, GraphSAGE, and GAT are evaluated on the identical test split to guarantee fair scientific comparison.
