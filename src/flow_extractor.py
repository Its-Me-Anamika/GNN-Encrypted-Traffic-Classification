"""
Bidirectional Network Flow Extractor using Scapy.
Processes PCAP / PCAPNG files, extracts 5-tuple bidirectional flows,
and computes packet, byte, timing (IAT), duration, and TCP flag metrics.
"""
import os
import math
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from scapy.all import PcapReader, IP, IPv6, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLS,
    FLOW_INACTIVITY_TIMEOUT,
)

logger = logging.getLogger(__name__)


class FlowRecord:
    """Represents a single bidirectional flow tracked across time."""

    def __init__(self, fwd_key: Tuple[str, str, int, int, str], start_time: float):
        self.src_ip = fwd_key[0]
        self.dst_ip = fwd_key[1]
        self.src_port = fwd_key[2]
        self.dst_port = fwd_key[3]
        self.proto_name = fwd_key[4]

        self.start_time: float = start_time
        self.last_time: float = start_time

        # Packet and byte lists
        self.packet_sizes: List[int] = []
        self.timestamps: List[float] = []

        # Directional statistics
        self.fwd_packets: int = 0
        self.bwd_packets: int = 0
        self.fwd_bytes: int = 0
        self.bwd_bytes: int = 0

        # TCP Flag counters
        self.flag_syn: int = 0
        self.flag_ack: int = 0
        self.flag_fin: int = 0
        self.flag_rst: int = 0
        self.flag_psh: int = 0
        self.flag_urg: int = 0

    def add_packet(self, pkt_len: int, timestamp: float, is_fwd: bool, tcp_flags: Optional[str] = None):
        """Add a packet to this flow."""
        self.packet_sizes.append(pkt_len)
        self.timestamps.append(timestamp)
        self.last_time = max(self.last_time, timestamp)

        if is_fwd:
            self.fwd_packets += 1
            self.fwd_bytes += pkt_len
        else:
            self.bwd_packets += 1
            self.bwd_bytes += pkt_len

        if tcp_flags:
            flags_upper = str(tcp_flags).upper()
            if "S" in flags_upper:
                self.flag_syn += 1
            if "A" in flags_upper:
                self.flag_ack += 1
            if "F" in flags_upper:
                self.flag_fin += 1
            if "R" in flags_upper:
                self.flag_rst += 1
            if "P" in flags_upper:
                self.flag_psh += 1
            if "U" in flags_upper:
                self.flag_urg += 1

    def to_dict(self, label: Optional[str] = None, source_file: str = "") -> Dict[str, Any]:
        """Compute statistical features and return as a feature dictionary."""
        total_packets = len(self.packet_sizes)
        total_bytes = sum(self.packet_sizes)
        duration_sec = max(self.last_time - self.start_time, 0.0)

        # Packet size statistics
        if total_packets > 0:
            pkt_size_min = float(min(self.packet_sizes))
            pkt_size_max = float(max(self.packet_sizes))
            pkt_size_mean = float(np.mean(self.packet_sizes))
            pkt_size_std = float(np.std(self.packet_sizes)) if total_packets > 1 else 0.0
        else:
            pkt_size_min = pkt_size_max = pkt_size_mean = pkt_size_std = 0.0

        # Inter-arrival time (IAT) statistics
        if total_packets > 1:
            sorted_times = sorted(self.timestamps)
            iats = [sorted_times[i] - sorted_times[i - 1] for i in range(1, len(sorted_times))]
            # Clamp negative IATs (clock skew in packet capture)
            iats = [max(0.0, iat) for iat in iats]
            iat_min = float(min(iats))
            iat_max = float(max(iats))
            iat_mean = float(np.mean(iats))
            iat_std = float(np.std(iats)) if len(iats) > 1 else 0.0
        else:
            iat_min = iat_max = iat_mean = iat_std = 0.0

        # Rates
        dur_div = max(duration_sec, 1e-6)
        packet_rate = float(total_packets / dur_div)
        byte_rate = float(total_bytes / dur_div)

        # Ratios
        fwd_bwd_pkt_ratio = float(self.fwd_packets / max(self.bwd_packets, 1))
        fwd_bwd_byte_ratio = float(self.fwd_bytes / max(self.bwd_bytes, 1))

        flow_id = f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}_{self.proto_name}_{self.start_time:.3f}"

        return {
            "flow_id": flow_id,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.proto_name,
            "start_time": self.start_time,
            "end_time": self.last_time,
            "source_file": source_file,
            "label": label if label is not None else "unknown",
            # Numerical features
            "total_packets": total_packets,
            "fwd_packets": self.fwd_packets,
            "bwd_packets": self.bwd_packets,
            "total_bytes": total_bytes,
            "fwd_bytes": self.fwd_bytes,
            "bwd_bytes": self.bwd_bytes,
            "duration_sec": duration_sec,
            "packet_rate_per_sec": packet_rate,
            "byte_rate_per_sec": byte_rate,
            "pkt_size_min": pkt_size_min,
            "pkt_size_max": pkt_size_max,
            "pkt_size_mean": pkt_size_mean,
            "pkt_size_std": pkt_size_std,
            "iat_min": iat_min,
            "iat_max": iat_max,
            "iat_mean": iat_mean,
            "iat_std": iat_std,
            "fwd_bwd_packet_ratio": fwd_bwd_pkt_ratio,
            "fwd_bwd_byte_ratio": fwd_bwd_byte_ratio,
            "flag_syn_count": self.flag_syn,
            "flag_ack_count": self.flag_ack,
            "flag_fin_count": self.flag_fin,
            "flag_rst_count": self.flag_rst,
            "flag_psh_count": self.flag_psh,
            "flag_urg_count": self.flag_urg,
        }


class PCAPFlowExtractor:
    """Extracts bidirectional flows from PCAP/PCAPNG files with streaming memory efficiency."""

    def __init__(self, inactivity_timeout: float = FLOW_INACTIVITY_TIMEOUT):
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required for PCAP parsing. Run 'pip install scapy'.")
        self.inactivity_timeout = inactivity_timeout

    def extract_from_pcap(
        self,
        pcap_path: str,
        label: Optional[str] = None,
        max_packets: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Reads a PCAP file packet-by-packet using PcapReader, tracks bidirectional flows,
        and outputs a pandas DataFrame with flow-level statistics.
        """
        pcap_path = str(pcap_path)
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        # Active flows map: canonical_key -> FlowRecord
        # Key: (min_endpoint, max_endpoint, proto) to group both directions
        flows: Dict[Tuple, FlowRecord] = {}
        completed_flows: List[Dict[str, Any]] = []

        filename = os.path.basename(pcap_path)
        packet_count = 0

        try:
            reader = PcapReader(pcap_path)
            for pkt in reader:
                packet_count += 1
                if max_packets and packet_count > max_packets:
                    break

                # Extract IP layer
                if IP in pkt:
                    ip_layer = pkt[IP]
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst
                    proto_num = ip_layer.proto
                elif IPv6 in pkt:
                    ip_layer = pkt[IPv6]
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst
                    proto_num = ip_layer.nh
                else:
                    # Non-IP packet (ARP, etc.), skip
                    continue

                timestamp = float(pkt.time)
                pkt_len = len(pkt)

                # Extract transport layer ports & flags
                tcp_flags = None
                if TCP in pkt:
                    proto_name = "TCP"
                    src_port = int(pkt[TCP].sport)
                    dst_port = int(pkt[TCP].dport)
                    tcp_flags = str(pkt[TCP].flags)
                elif UDP in pkt:
                    proto_name = "UDP"
                    src_port = int(pkt[UDP].sport)
                    dst_port = int(pkt[UDP].dport)
                elif ICMP in pkt:
                    proto_name = "ICMP"
                    src_port = 0
                    dst_port = 0
                else:
                    proto_name = f"IP_{proto_num}"
                    src_port = 0
                    dst_port = 0

                ep1 = (src_ip, src_port)
                ep2 = (dst_ip, dst_port)

                # Canonical flow conversation key
                if ep1 <= ep2:
                    canon_key = (ep1, ep2, proto_name)
                else:
                    canon_key = (ep2, ep1, proto_name)

                # Check if flow already exists
                if canon_key in flows:
                    flow = flows[canon_key]
                    # Check inactivity timeout
                    if (timestamp - flow.last_time) > self.inactivity_timeout:
                        # Flow expired; close it and start a new session
                        completed_flows.append(flow.to_dict(label=label, source_file=filename))
                        flow = FlowRecord((src_ip, dst_ip, src_port, dst_port, proto_name), timestamp)
                        flows[canon_key] = flow
                        is_fwd = True
                    else:
                        is_fwd = (src_ip == flow.src_ip and src_port == flow.src_port)
                else:
                    # First packet defines forward orientation
                    flow = FlowRecord((src_ip, dst_ip, src_port, dst_port, proto_name), timestamp)
                    flows[canon_key] = flow
                    is_fwd = True

                flow.add_packet(pkt_len, timestamp, is_fwd=is_fwd, tcp_flags=tcp_flags)

            reader.close()
        except Exception as e:
            logger.error(f"Error while reading {pcap_path}: {e}")
            raise

        # Flush remaining active flows
        for flow in flows.values():
            completed_flows.append(flow.to_dict(label=label, source_file=filename))

        if not completed_flows:
            # Return empty DataFrame with expected columns
            cols = ["flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "label"] + ALL_FEATURE_COLS
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(completed_flows)
        return df


def extract_flows_from_directory(
    directory_path: str,
    label: Optional[str] = None,
    output_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extracts flows from all .pcap and .pcapng files in a directory.
    Labels are attached to ground-truth sets.
    """
    extractor = PCAPFlowExtractor()
    pcap_files = []
    dir_path = Path(directory_path)

    if dir_path.is_dir():
        for ext in ("*.pcap", "*.pcapng", "*.cap"):
            pcap_files.extend(list(dir_path.glob(ext)))

    if not pcap_files:
        logger.warning(f"No PCAP files found in directory {directory_path}")
        return pd.DataFrame()

    dfs = []
    for pcap_file in sorted(pcap_files):
        try:
            df = extractor.extract_from_pcap(str(pcap_file), label=label)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.error(f"Skipping failed PCAP {pcap_file}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        combined_df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved {len(combined_df)} flows to {output_csv_path}")

    return combined_df
