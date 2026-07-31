import math
import random
import numpy as np

FEATURE_NAMES = [
    "fwd_pkt_count", "bwd_pkt_count", "total_pkt_count",
    "fwd_byte_count", "bwd_byte_count", "total_byte_count",
    "pkt_len_min", "pkt_len_max", "pkt_len_mean", "pkt_len_std", "pkt_len_median",
    "fwd_pkt_len_mean", "bwd_pkt_len_mean",
    "flow_duration", "flow_bytes_per_sec", "flow_pkts_per_sec",
    "iat_mean", "iat_std", "iat_min", "iat_max",
    "tcp_flag_syn", "tcp_flag_ack", "tcp_flag_fin", "tcp_flag_rst", "tcp_flag_psh", "tcp_flag_urg",
    "payload_bytes_ratio", "is_vpn_flag"
]

class FeatureScaler:
    """StandardScaler implementation for feature normalization."""
    def __init__(self, num_features=28):
        self.num_features = num_features
        self.mean = np.zeros(num_features, dtype=np.float32)
        self.std = np.ones(num_features, dtype=np.float32)
        self.is_fitted = False

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0  # Avoid division by zero
        self.is_fitted = True
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if not self.is_fitted:
            return X
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def extract_statistical_features(packet_lengths=None, packet_iats=None, flags=None, is_vpn=False, duration=1.0):
    """
    Extracts 28 statistical features from network flow packet metadata.
    """
    if packet_lengths is None or len(packet_lengths) == 0:
        packet_lengths = [64, 128, 512, 1024, 64, 256, 1280]
    if packet_iats is None or len(packet_iats) == 0:
        packet_iats = [0.01, 0.05, 0.02, 0.12, 0.03]
    if flags is None:
        flags = {"SYN": 1, "ACK": 12, "FIN": 1, "RST": 0, "PSH": 5, "URG": 0}

    total_pkts = len(packet_lengths)
    fwd_pkts = max(1, total_pkts // 2 + random.randint(-1, 1))
    bwd_pkts = max(1, total_pkts - fwd_pkts)

    total_bytes = int(sum(packet_lengths))
    fwd_bytes = int(total_bytes * 0.6)
    bwd_bytes = int(total_bytes - fwd_bytes)

    pkt_min = float(min(packet_lengths))
    pkt_max = float(max(packet_lengths))
    pkt_mean = float(np.mean(packet_lengths))
    pkt_std = float(np.std(packet_lengths))
    pkt_median = float(np.median(packet_lengths))

    fwd_len_mean = fwd_bytes / fwd_pkts
    bwd_len_mean = bwd_bytes / bwd_pkts

    duration = max(0.001, duration)
    bytes_per_sec = total_bytes / duration
    pkts_per_sec = total_pkts / duration

    iat_mean = float(np.mean(packet_iats))
    iat_std = float(np.std(packet_iats))
    iat_min = float(min(packet_iats))
    iat_max = float(max(packet_iats))

    syn = float(flags.get("SYN", 0))
    ack = float(flags.get("ACK", 0))
    fin = float(flags.get("FIN", 0))
    rst = float(flags.get("RST", 0))
    psh = float(flags.get("PSH", 0))
    urg = float(flags.get("URG", 0))

    payload_ratio = float(fwd_bytes / max(1, total_bytes))
    vpn_flag = 1.0 if is_vpn else 0.0

    features = [
        float(fwd_pkts), float(bwd_pkts), float(total_pkts),
        float(fwd_bytes), float(bwd_bytes), float(total_bytes),
        pkt_min, pkt_max, pkt_mean, pkt_std, pkt_median,
        float(fwd_len_mean), float(bwd_len_mean),
        float(duration), float(bytes_per_sec), float(pkts_per_sec),
        iat_mean, iat_std, iat_min, iat_max,
        syn, ack, fin, rst, psh, urg,
        payload_ratio, vpn_flag
    ]
    return features

class SyntheticDataGenerator:
    """Generates realistic synthetic network flow samples for demo & testing."""
    
    CATEGORIES = [
        "Streaming", "Chat", "VoIP", "File Transfer", "Browsing", "P2P", "Email", "VPN", "Unknown"
    ]

    CATEGORY_PROFILES = {
        "Streaming": {"pkts": (500, 3000), "bytes": (400000, 5000000), "dur": (30, 300), "port": 443},
        "Chat": {"pkts": (10, 80), "bytes": (1000, 15000), "dur": (2, 40), "port": 443},
        "VoIP": {"pkts": (200, 1500), "bytes": (20000, 200000), "dur": (10, 180), "port": 5060},
        "File Transfer": {"pkts": (1000, 8000), "bytes": (1000000, 20000000), "dur": (15, 200), "port": 21},
        "Browsing": {"pkts": (30, 300), "bytes": (5000, 150000), "dur": (1, 15), "port": 443},
        "P2P": {"pkts": (400, 4000), "bytes": (300000, 8000000), "dur": (20, 300), "port": 6881},
        "Email": {"pkts": (15, 120), "bytes": (3000, 50000), "dur": (2, 20), "port": 993},
        "VPN": {"pkts": (800, 5000), "bytes": (500000, 10000000), "dur": (40, 400), "port": 1194},
        "Unknown": {"pkts": (5, 50), "bytes": (500, 5000), "dur": (0.5, 5), "port": 8888}
    }

    SUBNETS = ["192.168.1.", "10.0.0.", "172.16.0."]
    DEST_IPS = ["142.250.190.46", "157.240.22.35", "104.16.132.229", "13.107.42.14", "52.84.18.23"]

    @classmethod
    def generate_flow(cls, category=None):
        if category is None or category not in cls.CATEGORY_PROFILES:
            category = random.choice(cls.CATEGORIES)

        profile = cls.CATEGORY_PROFILES[category]
        duration = float(random.uniform(*profile["dur"]))
        total_pkts = random.randint(*profile["pkts"])
        total_bytes = random.randint(*profile["bytes"])

        pkt_lengths = [random.randint(64, 1500) for _ in range(min(total_pkts, 50))]
        pkt_iats = [random.uniform(0.001, duration / max(1, total_pkts)) for _ in range(min(total_pkts, 50))]

        is_vpn = (category == "VPN") or (random.random() < 0.15)
        flags = {"SYN": 1, "ACK": random.randint(10, 50), "FIN": 1, "RST": 0, "PSH": random.randint(2, 20), "URG": 0}

        features = extract_statistical_features(
            packet_lengths=pkt_lengths,
            packet_iats=pkt_iats,
            flags=flags,
            is_vpn=is_vpn,
            duration=duration
        )

        src_ip = random.choice(cls.SUBNETS) + str(random.randint(2, 254))
        dst_ip = random.choice(cls.DEST_IPS)
        src_port = random.randint(1024, 65535)
        dst_port = profile["port"] if random.random() < 0.8 else random.randint(80, 8443)
        protocol = "TCP" if dst_port not in [5060, 6881, 1194] or random.random() < 0.7 else "UDP"

        return {
            "src_ip": src_ip,
            "src_port": src_port,
            "dst_ip": dst_ip,
            "dst_port": dst_port,
            "protocol": protocol,
            "duration": round(duration, 2),
            "total_packets": total_pkts,
            "total_bytes": total_bytes,
            "is_vpn": is_vpn,
            "true_category": category,
            "features": features
        }
