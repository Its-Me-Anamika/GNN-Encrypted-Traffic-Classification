"""
Fine-Grained Encrypted Traffic Classification using Graph Neural Networks (GraphSAGE/GAT).
Configuration module: paths, feature definitions, dataset catalogs, graph topology,
and model hyperparameters for both Binary (VPN vs Non-VPN) and Fine-Grained Application tasks.
"""
from pathlib import Path

# Base Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PCAP_DIR = DATA_DIR / "raw_pcaps"
VPN_PCAP_DIR = RAW_PCAP_DIR / "vpn"
NON_VPN_PCAP_DIR = RAW_PCAP_DIR / "non_vpn"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TEST_SAMPLES_DIR = DATA_DIR / "test_samples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "saved_models"

# Ensure essential output and data folders exist
for p in [RAW_PCAP_DIR, VPN_PCAP_DIR, NON_VPN_PCAP_DIR, PROCESSED_DATA_DIR, TEST_SAMPLES_DIR, OUTPUTS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Feature Names extracted from bidirectional flow metadata (No payload decryption)
NUMERICAL_FEATURES = [
    # Packet counts
    "total_packets",
    "fwd_packets",
    "bwd_packets",
    # Byte counts
    "total_bytes",
    "fwd_bytes",
    "bwd_bytes",
    # Duration and rates
    "duration_sec",
    "packet_rate_per_sec",
    "byte_rate_per_sec",
    # Packet size statistics (in bytes)
    "pkt_size_min",
    "pkt_size_max",
    "pkt_size_mean",
    "pkt_size_std",
    # Inter-arrival time statistics (in seconds)
    "iat_min",
    "iat_max",
    "iat_mean",
    "iat_std",
    # Directional ratios
    "fwd_bwd_packet_ratio",
    "fwd_bwd_byte_ratio",
    # TCP flag counts
    "flag_syn_count",
    "flag_ack_count",
    "flag_fin_count",
    "flag_rst_count",
    "flag_psh_count",
    "flag_urg_count",
]

CATEGORICAL_FEATURES = [
    "protocol",  # TCP, UDP, ICMP, OTHER
]

ALL_FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Classification Tasks
# 1. Binary Task (Initial Experiment)
BINARY_CLASSES = ["non_vpn", "vpn"]

# 2. Canonical Fine-Grained Application Categories (Primary Research Objective)
# Aligned with ISCXVPN2016 / USTC-TFC2016 research benchmarks
CANONICAL_APP_CATEGORIES = [
    "web_browsing",
    "video_streaming",
    "messaging",
    "voip",
    "file_transfer",
    "audio_streaming",
    "vpn_browsing",
    "vpn_streaming",
    "vpn_messaging",
    "vpn_voip",
    "vpn_file_transfer",
]

# Flow Inactivity Timeout (RFC 5103 / IPFIX standard: 120 seconds)
FLOW_INACTIVITY_TIMEOUT = 120.0

# Graph Construction Topologies
# Options:
# - 'endpoint_sharing': Connects flows that share source or destination IP addresses / host endpoints
# - 'temporal': Connects flows whose active time intervals overlap or are within delta_t seconds
# - 'knn': Feature similarity k-NN in normalized flow metadata space
# - 'hybrid': Union of endpoint sharing and temporal co-occurrence
DEFAULT_GRAPH_TOPOLOGY = "endpoint_sharing"
KNN_K_NEIGHBORS = 5
TEMPORAL_WINDOW_SEC = 5.0

# GraphSAGE Model Hyperparameters (Hamilton et al., NeurIPS 2017)
GRAPHSAGE_CONFIG = {
    "hidden_dim": 64,
    "embed_dim": 32,
    "dropout": 0.2,
    "aggregator_type": "mean",  # Mean Aggregator
    "learning_rate": 0.005,
    "weight_decay": 5e-4,
    "epochs": 60,
    "patience": 10,
    "random_seed": 42,
}

# GAT (Graph Attention Network) Hyperparameters (Veličković et al., ICLR 2018)
GAT_CONFIG = {
    "hidden_dim": 64,
    "embed_dim": 32,
    "num_heads": 4,
    "dropout": 0.2,
    "alpha": 0.2,  # LeakyReLU slope for attention coefficients
    "learning_rate": 0.005,
    "weight_decay": 5e-4,
    "epochs": 60,
    "patience": 10,
    "random_seed": 42,
}

# Baseline Random Forest & MLP Hyperparameters
BASELINE_RF_CONFIG = {
    "n_estimators": 100,
    "max_depth": 12,
    "random_state": 42,
}

BASELINE_MLP_CONFIG = {
    "hidden_layer_sizes": (64, 32),
    "max_iter": 200,
    "random_state": 42,
}

# Supported Benchmark Datasets Catalog
DATASET_CATALOG = {
    "ISCXVPN2016": {
        "full_name": "ISCX VPN-nonVPN Traffic Dataset (UNB / CIC)",
        "citation": "Gerard Drapier et al., 'Characterization of Encrypted and VPN Traffic using Time-related Features', ICISSP 2016.",
        "url": "https://www.unb.ca/cic/datasets/vpn.html",
        "total_size": "~28 GB uncompressed PCAPs",
        "supports_binary_vpn": True,
        "supports_fine_grained_apps": True,
        "classes": [
            "browsing (Firefox, Chrome)",
            "streaming (YouTube, Vimeo, Netflix)",
            "chat (Facebook, Hangouts, Skype)",
            "voip (Skype audio, Hangouts audio)",
            "file_transfer (FTPS, SFTP, Skype transfer)",
            "mail (Thunderbird SMTPS, POP3S, IMAPS)",
        ],
        "notes": "Gold standard for VPN vs Non-VPN and application category classification.",
    },
    "USTC-TFC2016": {
        "full_name": "USTC Encrypted Traffic / Malware Dataset (2016)",
        "citation": "Wang et al., 'Malware Traffic Classification Using Convolutional Neural Network for Representation Learning', IPCCC 2017.",
        "url": "https://github.com/yungshenglu/USTC-TFC2016",
        "total_size": "~3.7 GB PCAPs",
        "supports_binary_vpn": False,
        "supports_fine_grained_apps": True,
        "classes": [
            "Facetime", "FTP", "Gmail", "MySQL", "Outlook",
            "Skype", "SMB", "Weibo", "WorldOfWarcraft", "YouTube"
        ],
        "notes": "Excellent for 10-class fine-grained benign application classification.",
    },
}
