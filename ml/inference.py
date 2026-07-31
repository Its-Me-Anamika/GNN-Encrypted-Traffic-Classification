import os
import sys
import argparse
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.classifier import classifier_service
from app.utils.preprocessing import SyntheticDataGenerator

def main():
    parser = argparse.ArgumentParser(description="Standalone Encrypted Traffic GNN Inference CLI")
    parser.add_argument("--category", type=str, default=None, help="Force synthetic sample from category")
    parser.add_argument("--is-vpn", action="store_true", help="Flag flow as VPN encapsulated")
    args = parser.parse_args()

    flow_sample = SyntheticDataGenerator.generate_flow(category=args.category)
    if args.is_vpn:
        flow_sample["is_vpn"] = True

    print("[*] Running GNN Inference on Input Flow 5-Tuple...")
    print(f"    - Source:      {flow_sample['src_ip']}:{flow_sample['src_port']}")
    print(f"    - Destination: {flow_sample['dst_ip']}:{flow_sample['dst_port']}")
    print(f"    - Protocol:    {flow_sample['protocol']}")
    print(f"    - VPN Mode:    {flow_sample['is_vpn']}")

    result = classifier_service.classify_flow(flow_sample)

    print("\n[+] CLASSIFICATION RESULT:")
    print(f"    - Predicted Category: {result['predicted_category']}")
    print(f"    - Confidence Score:   {result['confidence'] * 100:.2f}%")
    print(f"    - Uncertainty Score:  {result['uncertainty_score']:.4f}")
    print(f"    - Unknown Flag:       {result['is_unknown']}")
    print(f"\n[+] GNNEXPLAINER TOP ATTRIBUTES:")
    for feat in result['explanation']['top_features'][:4]:
        print(f"    * {feat['feature_name']:<22}: {feat['importance_score']}% contribution")

if __name__ == "__main__":
    main()
