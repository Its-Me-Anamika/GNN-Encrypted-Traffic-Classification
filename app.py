"""
Demonstration Interface for Encrypted Traffic Classification using Graph Neural Networks.
Provides:
1. Native Windows GUI (Tkinter) to browse/load any PCAP, extract flows, inspect graph stats, and classify.
2. CLI mode for headless presentation or quick command-line demonstrations.
"""
import os
import sys
import argparse
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from src.config import (
    RAW_PCAP_DIR,
    TEST_SAMPLES_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    DEFAULT_GRAPH_TOPOLOGY,
    GRAPHSAGE_CONFIG,
)
from src.flow_extractor import PCAPFlowExtractor
from src.graph_builder import FlowGraphBuilder
from src.models.graphsage import GraphSAGEClassifier, GraphSAGETrainer
from src.models.baseline import TabularBaseline
from src.dataset_manager import DatasetManager


class PCAPDemoEngine:
    """Core processing engine used by both GUI and CLI demonstration interfaces."""

    def __init__(self):
        self.extractor = PCAPFlowExtractor()
        self.builder = FlowGraphBuilder(topology_mode=DEFAULT_GRAPH_TOPOLOGY)
        self.dataset_manager = DatasetManager()
        self.loaded_df: Optional[pd.DataFrame] = None
        self.last_pcap_path: Optional[str] = None
        self.graph_data = None

    def process_pcap_file(self, pcap_path: str, max_packets: Optional[int] = None) -> pd.DataFrame:
        """Extracts bidirectional flows from user-selected PCAP."""
        pcap_path = str(pcap_path)
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        self.last_pcap_path = pcap_path
        # Infer labels from filename if available
        fname_lower = os.path.basename(pcap_path).lower()
        binary_label = "vpn" if "vpn" in fname_lower else "non_vpn"
        app_label = self.dataset_manager._infer_app_from_filename(fname_lower)

        df = self.extractor.extract_from_pcap(pcap_path, label=app_label, max_packets=max_packets)
        if not df.empty:
            df["binary_label"] = binary_label
            df["app_label"] = app_label
        self.loaded_df = df
        return df

    def analyze_and_predict(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Constructs flow graph and runs GraphSAGE and Baseline prediction."""
        if df is None:
            df = self.loaded_df
        if df is None or df.empty:
            raise ValueError("No flows available. Process a PCAP file first.")

        # Construct Graph
        graph_data = self.builder.create_graph_data(
            df=df,
            label_col="app_label" if "app_label" in df.columns else "label",
            test_size=0.3,
            val_size=0.1,
        )
        self.graph_data = graph_data

        # Fit Tabular Baseline
        X = graph_data.x.numpy()
        y = graph_data.y.numpy()
        train_mask = graph_data.train_mask.numpy()

        rf = TabularBaseline(model_type="random_forest")
        rf.fit(X[train_mask], y[train_mask])
        rf_preds = rf.predict(X)

        # Train GraphSAGE
        sage_model = GraphSAGEClassifier(
            in_features=graph_data.num_features,
            hidden_dim=GRAPHSAGE_CONFIG["hidden_dim"],
            embed_dim=GRAPHSAGE_CONFIG["embed_dim"],
            num_classes=graph_data.num_classes,
        )
        trainer = GraphSAGETrainer(sage_model, config=GRAPHSAGE_CONFIG)
        trainer.train_graph(
            x=graph_data.x,
            edge_index=graph_data.edge_index,
            y=graph_data.y,
            train_mask=graph_data.train_mask,
            val_mask=graph_data.val_mask,
            epochs=30,
        )
        sage_preds, sage_probs = trainer.predict(graph_data.x, graph_data.edge_index)

        class_names = graph_data.label_names
        flow_predictions = []
        for i in range(len(df)):
            pred_idx = sage_preds[i]
            pred_class = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"
            conf = float(sage_probs[i, pred_idx])
            flow_predictions.append({
                "flow_id": graph_data.flow_ids[i],
                "src_ip": df.iloc[i].get("src_ip", ""),
                "dst_ip": df.iloc[i].get("dst_ip", ""),
                "protocol": df.iloc[i].get("protocol", ""),
                "packets": int(df.iloc[i].get("total_packets", 0)),
                "bytes": int(df.iloc[i].get("total_bytes", 0)),
                "duration_sec": round(float(df.iloc[i].get("duration_sec", 0.0)), 3),
                "predicted_app": pred_class,
                "confidence": round(conf, 4),
            })

        return {
            "graph_stats": graph_data.graph_stats,
            "predictions": flow_predictions,
            "class_names": class_names,
            "total_flows": len(df),
        }


def run_cli_demo(pcap_path: Optional[str] = None):
    """Runs demonstration in CLI mode."""
    engine = PCAPDemoEngine()
    print("\n" + "=" * 78)
    print(" ENCRYPTED TRAFFIC GNN CLASSIFIER - DEMONSTRATION")
    print("=" * 78)

    if not pcap_path:
        print("[*] No PCAP specified. Checking data/test_samples for sample verification PCAP...")
        sample_path = TEST_SAMPLES_DIR / "SYNTHETIC_PIPELINE_VERIFICATION.pcap"
        if not sample_path.exists():
            print("[*] Generating synthetic verification PCAP...")
            sample_path = engine.dataset_manager.generate_pipeline_verification_pcap()
        pcap_path = str(sample_path)
        print(f"[!] Using Verification Sample: {pcap_path}")
        print("    (Marked SYNTHETIC_TEST_ONLY; real PCAPs can be passed via: python app.py --pcap <file>)")

    print(f"\n[1] Reading and extracting flows from: {pcap_path} ...")
    df = engine.process_pcap_file(pcap_path)
    print(f"[+] Successfully extracted {len(df)} bidirectional flows.")

    print("\n[2] Building Flow Graph & Performing Classification...")
    result = engine.analyze_and_predict(df)
    stats = result["graph_stats"]

    print("\n" + "-" * 40)
    print(" GRAPH TOPOLOGY STATISTICS:")
    print("-" * 40)
    print(f"  Nodes (Flows):             {stats['num_nodes']}")
    print(f"  Edges (Interactions):      {stats['num_edges']}")
    print(f"  Average Degree:            {stats['avg_degree']:.2f}")
    print(f"  Graph Density:             {stats['density']:.4f}")
    print(f"  Connected Components:      {stats['num_connected_components']}")

    print("\n" + "-" * 78)
    print(" PREDICTED FLOW CATEGORIES (GraphSAGE):")
    print("-" * 78)
    pred_df = pd.DataFrame(result["predictions"])
    cols_to_show = ["flow_id", "protocol", "packets", "bytes", "duration_sec", "predicted_app", "confidence"]
    print(pred_df[cols_to_show].head(15).to_string(index=False))
    print("-" * 78)


def run_gui_demo():
    """Runs demonstration in Native Windows Tkinter GUI mode."""
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    engine = PCAPDemoEngine()
    root = tk.Tk()
    root.title("Fine-Grained Encrypted Traffic Classification (GraphSAGE/GAT)")
    root.geometry("980x680")
    root.minsize(800, 500)

    # Style
    style = ttk.Style()
    style.theme_use("clam")

    # Header Frame
    header_frame = ttk.Frame(root, padding=10)
    header_frame.pack(fill=tk.X)
    title_label = ttk.Label(
        header_frame,
        text="Fine-Grained Encrypted Traffic Classification using GNNs",
        font=("Segoe UI", 14, "bold"),
    )
    title_label.pack(anchor=tk.W)

    subtitle_label = ttk.Label(
        header_frame,
        text="GraphSAGE / GAT Flow-Level Architecture | Bidirectional Metadata (No Payload Decryption)",
        font=("Segoe UI", 9),
    )
    subtitle_label.pack(anchor=tk.W)

    # Control Frame
    ctrl_frame = ttk.LabelFrame(root, text=" 1. Input PCAP / Dataset Selection ", padding=10)
    ctrl_frame.pack(fill=tk.X, padx=10, pady=5)

    pcap_entry_var = tk.StringVar()
    pcap_entry = ttk.Entry(ctrl_frame, textvariable=pcap_entry_var, width=70)
    pcap_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def browse_file():
        file_path = filedialog.askopenfilename(
            title="Select PCAP / PCAPNG File",
            filetypes=[("PCAP files", "*.pcap *.pcapng *.cap"), ("All files", "*.*")],
        )
        if file_path:
            pcap_entry_var.set(file_path)

    def use_sample():
        sample_path = TEST_SAMPLES_DIR / "SYNTHETIC_PIPELINE_VERIFICATION.pcap"
        if not sample_path.exists():
            sample_path = engine.dataset_manager.generate_pipeline_verification_pcap()
        pcap_entry_var.set(str(sample_path))
        messagebox.showinfo("Sample Selected", f"Selected synthetic verification PCAP:\n{sample_path}\n(For code integrity testing)")

    browse_btn = ttk.Button(ctrl_frame, text="Browse PCAP...", command=browse_file)
    browse_btn.pack(side=tk.LEFT, padx=5)

    sample_btn = ttk.Button(ctrl_frame, text="Use Verification Sample", command=use_sample)
    sample_btn.pack(side=tk.LEFT, padx=5)

    # Action Buttons
    action_frame = ttk.Frame(root, padding=5)
    action_frame.pack(fill=tk.X, padx=10)

    # Status & Output Notebook
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    tab_flows = ttk.Frame(notebook)
    tab_graph = ttk.Frame(notebook)
    tab_catalog = ttk.Frame(notebook)

    notebook.add(tab_flows, text=" Extracted Flows & Predictions ")
    notebook.add(tab_graph, text=" Graph Topology & Metrics ")
    notebook.add(tab_catalog, text=" Dataset Catalog & Instructions ")

    # Table for Flows
    tree_scroll = ttk.Scrollbar(tab_flows)
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    tree_cols = ("flow_id", "protocol", "packets", "bytes", "duration", "prediction", "confidence")
    flow_tree = ttk.Treeview(tab_flows, columns=tree_cols, show="headings", yscrollcommand=tree_scroll.set)
    tree_scroll.config(command=flow_tree.yview)

    for col in tree_cols:
        flow_tree.heading(col, text=col.replace("_", " ").title())
        flow_tree.column(col, width=120)
    flow_tree.pack(fill=tk.BOTH, expand=True)

    # Graph stats text
    graph_text = tk.Text(tab_graph, wrap=tk.WORD, font=("Consolas", 10), padx=10, pady=10)
    graph_text.pack(fill=tk.BOTH, expand=True)

    # Catalog text
    cat_text = tk.Text(tab_catalog, wrap=tk.WORD, font=("Consolas", 10), padx=10, pady=10)
    cat_text.pack(fill=tk.BOTH, expand=True)
    cat_text.insert(tk.END, engine.dataset_manager.print_dataset_catalog())
    cat_text.config(state=tk.DISABLED)

    # Status Bar
    status_var = tk.StringVar(value="Ready. Select a PCAP file or use the verification sample to begin.")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W, padding=4)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def process_and_run():
        path = pcap_entry_var.get().strip()
        if not path:
            messagebox.showwarning("No File", "Please select a PCAP file first.")
            return

        status_var.set(f"Extracting flows from {os.path.basename(path)} with Scapy...")
        root.update_idletasks()

        try:
            df = engine.process_pcap_file(path)
            if df.empty:
                messagebox.showerror("Error", "No IP packets or valid flows found in PCAP.")
                status_var.set("PCAP extraction failed: No flows detected.")
                return

            status_var.set(f"Extracted {len(df)} flows. Constructing graph and training GraphSAGE...")
            root.update_idletasks()

            res = engine.analyze_and_predict(df)
            stats = res["graph_stats"]

            # Populate table
            for row in flow_tree.get_children():
                flow_tree.delete(row)

            for item in res["predictions"]:
                flow_tree.insert("", tk.END, values=(
                    item["flow_id"][:30] + "...",
                    item["protocol"],
                    item["packets"],
                    item["bytes"],
                    f"{item['duration_sec']}s",
                    item["predicted_app"],
                    f"{item['confidence'] * 100:.1f}%",
                ))

            # Populate Graph stats
            graph_text.delete("1.0", tk.END)
            graph_text.insert(tk.END, "=" * 60 + "\n")
            graph_text.insert(tk.END, " FLOW GRAPH TOPOLOGY METRICS\n")
            graph_text.insert(tk.END, "=" * 60 + "\n\n")
            graph_text.insert(tk.END, f"  Topology Mode:         {stats.get('topology_mode', 'N/A')}\n")
            graph_text.insert(tk.END, f"  Total Graph Nodes:     {stats.get('num_nodes', 0)}\n")
            graph_text.insert(tk.END, f"  Total Graph Edges:     {stats.get('num_edges', 0)}\n")
            graph_text.insert(tk.END, f"  Average Node Degree:   {stats.get('avg_degree', 0.0):.2f}\n")
            graph_text.insert(tk.END, f"  Graph Density:         {stats.get('density', 0.0):.4f}\n")
            graph_text.insert(tk.END, f"  Connected Components:  {stats.get('num_connected_components', 0)}\n\n")
            graph_text.insert(tk.END, "  Detected Classes:      " + ", ".join(res["class_names"]) + "\n")

            status_var.set(f"Done! Processed {len(df)} flows, constructed graph ({stats['num_edges']} edges), and ran GraphSAGE.")
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
            status_var.set(f"Error: {e}")

    run_btn = ttk.Button(action_frame, text="2. Process PCAP & Run GraphSAGE", command=process_and_run)
    run_btn.pack(side=tk.LEFT, padx=5)

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstration Interface for Encrypted Traffic GNN")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of GUI")
    parser.add_argument("--pcap", type=str, default=None, help="Path to PCAP file for CLI mode")
    args = parser.parse_args()

    if args.cli or os.environ.get("HEADLESS") == "1":
        run_cli_demo(args.pcap)
    else:
        # Default: attempt GUI, fallback to CLI if DISPLAY or Tk is unavailable
        try:
            run_gui_demo()
        except Exception as e:
            print(f"[*] Note: GUI mode could not initialize ({e}). Falling back to CLI mode...")
            run_cli_demo(args.pcap)
