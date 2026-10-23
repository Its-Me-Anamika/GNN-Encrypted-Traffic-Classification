"""
Test suite for PCAP reading and bidirectional flow extraction.
Generates an explicit SYNTHETIC_TEST_ONLY PCAP with known ground-truth packet structures
to verify Scapy parsing, flow grouping, timing, and feature computation.
"""
import os
import unittest
import numpy as np
from pathlib import Path
from scapy.all import IP, TCP, UDP, Raw, wrpcap

from src.flow_extractor import PCAPFlowExtractor
from src.config import TEST_SAMPLES_DIR, ALL_FEATURE_COLS, NUMERICAL_FEATURES


class TestFlowExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_SAMPLES_DIR, exist_ok=True)
        cls.test_pcap_path = str(TEST_SAMPLES_DIR / "SYNTHETIC_TEST_ONLY.pcap")

        # Synthesize known test packets:
        # Flow 1: TCP conversation between 192.168.1.100:54321 and 10.0.0.1:443
        # Flow 2: UDP conversation between 192.168.1.100:60000 and 8.8.8.8:53
        packets = []

        # Flow 1 packets:
        # Packet 1 (SYN) from client to server at t = 100.0
        p1 = IP(src="192.168.1.100", dst="10.0.0.1") / TCP(sport=54321, dport=443, flags="S")
        p1.time = 100.0
        packets.append(p1)

        # Packet 2 (SYN+ACK) from server to client at t = 100.05
        p2 = IP(src="10.0.0.1", dst="192.168.1.100") / TCP(sport=443, dport=54321, flags="SA")
        p2.time = 100.05
        packets.append(p2)

        # Packet 3 (ACK + Data) from client to server at t = 100.10 with 500 bytes payload
        p3 = IP(src="192.168.1.100", dst="10.0.0.1") / TCP(sport=54321, dport=443, flags="PA") / Raw(load=b"X" * 500)
        p3.time = 100.10
        packets.append(p3)

        # Packet 4 (ACK + Data) from server to client at t = 100.15 with 1200 bytes payload
        p4 = IP(src="10.0.0.1", dst="192.168.1.100") / TCP(sport=443, dport=54321, flags="PA") / Raw(load=b"Y" * 1200)
        p4.time = 100.15
        packets.append(p4)

        # Flow 2 packets (UDP DNS query/response):
        # Packet 5 from client to 8.8.8.8 at t = 100.02
        p5 = IP(src="192.168.1.100", dst="8.8.8.8") / UDP(sport=60000, dport=53) / Raw(load=b"DNS_QUERY" * 4)
        p5.time = 100.02
        packets.append(p5)

        # Packet 6 from 8.8.8.8 to client at t = 100.08
        p6 = IP(src="8.8.8.8", dst="192.168.1.100") / UDP(sport=53, dport=60000) / Raw(load=b"DNS_RESP" * 8)
        p6.time = 100.08
        packets.append(p6)

        wrpcap(cls.test_pcap_path, packets)

    def test_pcap_extraction(self):
        extractor = PCAPFlowExtractor()
        df = extractor.extract_from_pcap(self.test_pcap_path, label="vpn")

        # 1. Verify two flows extracted
        self.assertEqual(len(df), 2, f"Expected 2 flows, found {len(df)}")

        # 2. Check TCP Flow
        tcp_flows = df[df["protocol"] == "TCP"]
        self.assertEqual(len(tcp_flows), 1)
        tcp_flow = tcp_flows.iloc[0]

        self.assertEqual(tcp_flow["total_packets"], 4)
        self.assertEqual(tcp_flow["fwd_packets"], 2)
        self.assertEqual(tcp_flow["bwd_packets"], 2)
        self.assertAlmostEqual(tcp_flow["duration_sec"], 0.15, places=3)
        self.assertEqual(tcp_flow["flag_syn_count"], 2)  # S and SA
        self.assertGreaterEqual(tcp_flow["flag_ack_count"], 3)  # SA, PA, PA

        # 3. Check UDP Flow
        udp_flows = df[df["protocol"] == "UDP"]
        self.assertEqual(len(udp_flows), 1)
        udp_flow = udp_flows.iloc[0]
        self.assertEqual(udp_flow["total_packets"], 2)
        self.assertAlmostEqual(udp_flow["duration_sec"], 0.06, places=3)

        # 4. Check all expected feature columns are present and non-null
        for col in ALL_FEATURE_COLS:
            self.assertIn(col, df.columns, f"Missing expected feature {col}")
            self.assertFalse(df[col].isnull().any(), f"Column {col} contains NaN values")

        # 5. Check label assignment
        self.assertTrue((df["label"] == "vpn").all())

    def test_missing_pcap_raises(self):
        extractor = PCAPFlowExtractor()
        with self.assertRaises(FileNotFoundError):
            extractor.extract_from_pcap("non_existent_file.pcap")


if __name__ == "__main__":
    unittest.main()
