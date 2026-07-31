import os
import sys
import argparse
import numpy as np

# Ensure root import works
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.utils.preprocessing import SyntheticDataGenerator, extract_statistical_features

def download_iscx_dataset_instructions():
    print("=" * 70)
    print(" ISCX VPN-nonVPN DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("1. Download dataset PCAPs from UNB CIC official portal:")
    print("   https://www.unb.ca/cic/datasets/vpn.html")
    print("2. Place downloaded PCAP files into directory: ml/raw_pcaps/")
    print("3. Run this script with --pcap-dir ml/raw_pcaps/")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Preprocess ISCX PCAPs or Generate Synthetic Dataset")
    parser.add_argument("--pcap-dir", type=str, default=None, help="Directory containing raw PCAP files")
    parser.add_argument("--out-dir", type=str, default="ml/processed_data", help="Output directory for processed tensors")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples if no PCAPs provided")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pcap-dir and os.path.exists(args.pcap-dir):
        print(f"[*] Processing PCAP files from {args.pcap-dir}...")
        # Scapy parsing workflow
        pcap_files = [os.path.join(args.pcap-dir, f) for f in os.listdir(args.pcap-dir) if f.endswith(".pcap")]
        print(f"[+] Found {len(pcap_files)} PCAP files.")
    else:
        print("[*] No PCAP directory provided or found. Generating synthetic dataset benchmark...")
        download_iscx_dataset_instructions()

        features_list = []
        labels_list = []
        cat_map = {cat: idx for idx, cat in enumerate(SyntheticDataGenerator.CATEGORIES)}

        for _ in range(args.samples):
            flow = SyntheticDataGenerator.generate_flow()
            features_list.append(flow["features"])
            labels_list.append(cat_map[flow["true_category"]])

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)

        np.save(os.path.join(args.out_dir, "features.npy"), X)
        np.save(os.path.join(args.out_dir, "labels.npy"), y)
        print(f"[+] Processed dataset saved to: {args.out_dir} (Shapes: X={X.shape}, y={y.shape})")

if __name__ == "__main__":
    main()
