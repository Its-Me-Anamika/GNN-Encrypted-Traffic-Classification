import base64
import io
import numpy as np

from app.utils.preprocessing import extract_statistical_features, FeatureScaler

try:
    from scapy.all import rdpcap, IP, TCP, UDP
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

class FeatureExtractor:
    """Service to parse PCAP files or raw flow dicts into 28 statistical features."""
    
    def __init__(self):
        self.scaler = FeatureScaler()

    def extract_from_pcap_bytes(self, pcap_bytes, is_vpn=False):
        """Parses raw PCAP bytes into flow statistical features using Scapy."""
        if not HAS_SCAPY:
            # Fallback to simulated extraction if Scapy not present
            return extract_statistical_features(is_vpn=is_vpn)

        try:
            packets = rdpcap(io.BytesIO(pcap_bytes))
            if not packets:
                return extract_statistical_features(is_vpn=is_vpn)

            packet_lengths = []
            packet_timestamps = []
            flags = {"SYN": 0, "ACK": 0, "FIN": 0, "RST": 0, "PSH": 0, "URG": 0}

            for pkt in packets:
                packet_lengths.append(len(pkt))
                packet_timestamps.append(float(pkt.time))

                if pkt.haslayer(TCP):
                    tcp = pkt[TCP]
                    if tcp.flags.S: flags["SYN"] += 1
                    if tcp.flags.A: flags["ACK"] += 1
                    if tcp.flags.F: flags["FIN"] += 1
                    if tcp.flags.R: flags["RST"] += 1
                    if tcp.flags.P: flags["PSH"] += 1
                    if tcp.flags.U: flags["URG"] += 1

            duration = max(0.001, packet_timestamps[-1] - packet_timestamps[0]) if len(packet_timestamps) > 1 else 1.0
            
            iats = []
            for i in range(1, len(packet_timestamps)):
                iats.append(max(0.0001, packet_timestamps[i] - packet_timestamps[i-1]))
            if not iats:
                iats = [0.01]

            return extract_statistical_features(
                packet_lengths=packet_lengths,
                packet_iats=iats,
                flags=flags,
                is_vpn=is_vpn,
                duration=duration
            )

        except Exception as e:
            print(f"[!] Error parsing PCAP bytes with Scapy: {e}")
            return extract_statistical_features(is_vpn=is_vpn)

    def extract_from_base64_pcap(self, b64_str, is_vpn=False):
        try:
            raw_bytes = base64.b64decode(b64_str)
            return self.extract_from_pcap_bytes(raw_bytes, is_vpn=is_vpn)
        except Exception:
            return extract_statistical_features(is_vpn=is_vpn)

    def extract_from_dict(self, flow_dict):
        """Converts flow parameter dictionary into normalized feature vector."""
        if "custom_features" in flow_dict and flow_dict["custom_features"] is not None and len(flow_dict["custom_features"]) == 28:
            return flow_dict["custom_features"]

        duration = flow_dict.get("duration", 1.0)
        total_packets = flow_dict.get("total_packets", 10)
        total_bytes = flow_dict.get("total_bytes", 1500)
        is_vpn = flow_dict.get("is_vpn", False)

        avg_pkt_len = max(64, total_bytes // max(1, total_packets))
        packet_lengths = [int(np.random.normal(avg_pkt_len, avg_pkt_len * 0.2)) for _ in range(min(total_packets, 40))]
        packet_iats = [random_iat for random_iat in np.random.exponential(scale=duration / max(1, total_packets), size=min(total_packets, 40))]

        return extract_statistical_features(
            packet_lengths=packet_lengths,
            packet_iats=packet_iats,
            is_vpn=is_vpn,
            duration=duration
        )
