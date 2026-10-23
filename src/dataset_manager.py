"""
Dataset Manager for Encrypted Network Traffic.
Handles dataset cataloging, PCAP discovery, label mapping (binary & fine-grained application),
and pipeline self-test generation with strict scientific integrity safeguards.
"""
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from scapy.all import IP, TCP, UDP, Raw, wrpcap

from src.config import (
    RAW_PCAP_DIR,
    PROCESSED_DATA_DIR,
    TEST_SAMPLES_DIR,
    DATASET_CATALOG,
    ALL_FEATURE_COLS,
)
from src.flow_extractor import PCAPFlowExtractor, extract_flows_from_directory

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages raw PCAP ingestion, application label mapping, and dataset health checks."""

    def __init__(self, raw_dir: Path = RAW_PCAP_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def print_dataset_catalog(self) -> str:
        """Returns a formatted academic summary of benchmark datasets."""
        lines = [
            "=" * 78,
            " PUBLIC LABELED ENCRYPTED TRAFFIC DATASET CATALOG",
            "=" * 78,
        ]
        for name, meta in DATASET_CATALOG.items():
            lines.append(f"\n[Dataset] {name}: {meta['full_name']}")
            lines.append(f"  - Citation: {meta['citation']}")
            lines.append(f"  - Download URL: {meta['url']}")
            lines.append(f"  - Total Size: {meta['total_size']}")
            lines.append(f"  - Binary VPN vs Non-VPN: {'Supported' if meta['supports_binary_vpn'] else 'No'}")
            lines.append(f"  - Fine-Grained Application Labels: {'Supported' if meta['supports_fine_grained_apps'] else 'No'}")
            lines.append(f"  - Classes: {', '.join(meta['classes'])}")
            lines.append(f"  - Notes: {meta['notes']}")
        lines.append("\n" + "=" * 78)
        lines.append("HOW TO INGEST DATA INTO THIS PROJECT:")
        lines.append("  Option 1 (By Category / Application):")
        lines.append("    Create subfolders in 'data/raw_pcaps/<app_name>/' (e.g. 'data/raw_pcaps/youtube/')")
        lines.append("    and drop your downloaded .pcap / .pcapng files inside.")
        lines.append("  Option 2 (Binary VPN vs Non-VPN):")
        lines.append("    Drop VPN pcaps in 'data/raw_pcaps/vpn/' and non-VPN in 'data/raw_pcaps/non_vpn/'.")
        lines.append("=" * 78)
        return "\n".join(lines)

    def scan_for_raw_pcaps(self) -> List[Tuple[Path, str, str]]:
        """
        Scans data/raw_pcaps recursively for .pcap and .pcapng files.
        Returns a list of tuples: (file_path, binary_label, application_label).
        """
        results = []
        extensions = ("*.pcap", "*.pcapng", "*.cap")
        all_pcaps = []
        for ext in extensions:
            all_pcaps.extend(list(self.raw_dir.rglob(ext)))

        for pcap_file in sorted(all_pcaps):
            rel_path = pcap_file.relative_to(self.raw_dir)
            parts = rel_path.parts
            fname_lower = pcap_file.name.lower()

            # Determine binary label (vpn vs non_vpn)
            if any("vpn" in part.lower() for part in parts) or "vpn" in fname_lower:
                binary_label = "vpn"
            else:
                binary_label = "non_vpn"

            # Determine fine-grained application label
            if len(parts) > 1 and parts[0].lower() not in ("vpn", "non_vpn"):
                app_label = parts[0].lower()
            elif len(parts) > 2 and parts[0].lower() in ("vpn", "non_vpn"):
                app_label = parts[1].lower()
            else:
                # Infer application from filename conventions (e.g. skype, youtube, vimeo, etc.)
                app_label = self._infer_app_from_filename(fname_lower)

            results.append((pcap_file, binary_label, app_label))

        return results

    @staticmethod
    def _infer_app_from_filename(filename: str) -> str:
        """Infers fine-grained application name from standard benchmark PCAP filenames."""
        known_keywords = [
            "youtube", "vimeo", "netflix", "spotify", "skype", "facebook",
            "hangouts", "ftps", "sftp", "torrent", "bittorrent", "gmail",
            "email", "browsing", "firefox", "chrome", "voip", "chat"
        ]
        for kw in known_keywords:
            if kw in filename:
                return kw
        # If no known keyword, use sanitized stem
        stem = Path(filename).stem
        sanitized = re.sub(r"[_\d]+$", "", stem).lower().replace("vpn", "").strip("-_")
        return sanitized if sanitized else "unknown_application"

    def process_all_raw_pcaps(self, max_packets_per_pcap: Optional[int] = None) -> pd.DataFrame:
        """
        Processes all discovered raw PCAPs into flow-level features.
        Saves output to data/processed/flows_dataset.csv.
        """
        found = self.scan_for_raw_pcaps()
        if not found:
            logger.warning("No PCAP files discovered in data/raw_pcaps/.")
            return pd.DataFrame()

        extractor = PCAPFlowExtractor()
        all_dfs = []

        print(f"[*] Found {len(found)} PCAP file(s) across raw_pcaps directories.")
        for pcap_file, binary_label, app_label in found:
            print(f"  -> Processing: {pcap_file.name} | Binary: {binary_label} | App: {app_label}")
            try:
                df = extractor.extract_from_pcap(str(pcap_file), max_packets=max_packets_per_pcap)
                if not df.empty:
                    df["binary_label"] = binary_label
                    df["app_label"] = app_label
                    all_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to process {pcap_file}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        csv_path = self.processed_dir / "flows_dataset.csv"
        combined_df.to_csv(csv_path, index=False)
        print(f"[+] Flow extraction complete: {len(combined_df)} bidirectional flows saved to {csv_path}")
        return combined_df

    def load_processed_flows(self) -> Optional[pd.DataFrame]:
        """Loads processed flows CSV if available."""
        csv_path = self.processed_dir / "flows_dataset.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None

    @staticmethod
    def generate_pipeline_verification_pcap(output_path: Optional[Path] = None) -> Path:
        """
        Generates an explicit, strictly marked SYNTHETIC_PIPELINE_VERIFICATION PCAP file.
        Used strictly for automated self-tests, verifying Scapy parsing, flow grouping,
        graph construction, and training loops on machines before external PCAPs are downloaded.
        """
        if output_path is None:
            output_path = TEST_SAMPLES_DIR / "SYNTHETIC_PIPELINE_VERIFICATION.pcap"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        packets = []
        base_time = 1700000000.0

        # Define 4 simulated applications with distinctive conversational patterns:
        # App 1: 'video_streaming' (high downstream packet sizes, bursty)
        # App 2: 'web_browsing' (short bursts, request-response)
        # App 3: 'voip' (regular small packets, low IAT)
        # App 4: 'file_transfer' (continuous maximum segment size packets)
        flow_configs = [
            # (client_ip, srv_ip, client_port, srv_port, proto, num_pkts, pkt_len_fwd, pkt_len_bwd, iat, app)
            ("192.168.1.10", "142.250.190.46", 50001, 443, "TCP", 12, 100, 1400, 0.02, "video_streaming"),
            ("192.168.1.10", "142.250.190.46", 50002, 443, "TCP", 10, 120, 1350, 0.03, "video_streaming"),
            ("192.168.1.20", "104.244.42.1", 50003, 443, "TCP", 6, 250, 600, 0.15, "web_browsing"),
            ("192.168.1.20", "151.101.1.140", 50004, 443, "TCP", 8, 300, 850, 0.20, "web_browsing"),
            ("192.168.1.30", "20.198.118.22", 50005, 3478, "UDP", 16, 160, 160, 0.02, "voip"),
            ("192.168.1.30", "20.198.118.22", 50006, 3478, "UDP", 14, 160, 160, 0.02, "voip"),
            ("192.168.1.40", "185.199.108.133", 50007, 22, "TCP", 20, 80, 1460, 0.005, "file_transfer"),
            ("192.168.1.40", "185.199.108.133", 50008, 22, "TCP", 18, 80, 1460, 0.006, "file_transfer"),
        ]

        for src_ip, dst_ip, sport, dport, proto, n_pkts, len_fwd, len_bwd, iat, _ in flow_configs:
            t = base_time
            for i in range(n_pkts):
                is_fwd = (i % 2 == 0)
                cur_len = len_fwd if is_fwd else len_bwd
                cur_src = src_ip if is_fwd else dst_ip
                cur_dst = dst_ip if is_fwd else src_ip
                cur_sport = sport if is_fwd else dport
                cur_dport = dport if is_fwd else sport

                payload = b"Z" * max(cur_len - 40, 10)
                if proto == "TCP":
                    pkt = IP(src=cur_src, dst=cur_dst) / TCP(sport=cur_sport, dport=cur_dport, flags="PA") / Raw(load=payload)
                else:
                    pkt = IP(src=cur_src, dst=cur_dst) / UDP(sport=cur_sport, dport=cur_dport) / Raw(load=payload)
                pkt.time = t
                packets.append(pkt)
                t += iat
            base_time += 1.0

        wrpcap(str(output_path), packets)
        logger.info(f"Generated synthetic verification PCAP with {len(packets)} packets at {output_path}")
        return output_path
