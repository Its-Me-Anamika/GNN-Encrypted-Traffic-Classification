"""
End-to-End Experiment Pipeline for Encrypted Traffic Classification.
Coordinates PCAP extraction, flow graph construction, baseline ML, GraphSAGE, GAT,
and comparative evaluation answering the core research questions.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import torch

from src.config import (
    RAW_PCAP_DIR,
    PROCESSED_DATA_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    GRAPHSAGE_CONFIG,
    GAT_CONFIG,
    DEFAULT_GRAPH_TOPOLOGY,
)
from src.dataset_manager import DatasetManager
from src.graph_builder import FlowGraphBuilder, FlowGraphData
from src.models.baseline import TabularBaseline
from src.models.graphsage import GraphSAGEClassifier, GraphSAGETrainer
from src.models.gat import GATClassifier, GATTrainer
from src.evaluator import TrafficEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ExperimentPipeline:
    """Executes research experiments: Binary VPN and Fine-Grained Application Classification."""

    def __init__(self, output_dir: Path = OUTPUTS_DIR):
        self.output_dir = Path(output_dir)
        self.dataset_manager = DatasetManager()
        self.evaluator = TrafficEvaluator(self.output_dir)

    def load_or_extract_data(self, force_reextract: bool = False) -> Tuple[pd.DataFrame, bool]:
        """
        Loads processed flows from CSV if available, or extracts from raw PCAPs.
        If no real PCAPs exist, generates a synthetic pipeline verification dataset
        and explicitly flags is_synthetic=True.
        """
        # 1. Check if processed flows CSV already exists
        csv_path = PROCESSED_DATA_DIR / "flows_dataset.csv"
        if csv_path.exists() and not force_reextract:
            df = pd.read_csv(csv_path)
            if not df.empty:
                logger.info(f"Loaded existing processed flows dataset with {len(df)} flows from {csv_path}")
                is_synthetic = "SYNTHETIC" in str(df.get("source_file", "").values)
                return df, is_synthetic

        # 2. Check for real PCAPs in raw_pcaps
        raw_pcaps = self.dataset_manager.scan_for_raw_pcaps()
        if raw_pcaps:
            logger.info(f"Discovered {len(raw_pcaps)} real PCAPs. Extracting bidirectional flows...")
            df = self.dataset_manager.process_all_raw_pcaps()
            return df, False

        # 3. No real PCAPs found: Generate explicit synthetic verification dataset
        logger.warning(
            "\n" + "!" * 78 + "\n"
            " NOTICE: No real PCAP files found in 'data/raw_pcaps/'.\n"
            " Generating SYNTHETIC_PIPELINE_VERIFICATION.pcap to verify code execution.\n"
            " THIS DATA IS FOR TESTING CODE INTEGRITY ONLY - NOT FOR RESEARCH CLAIMS.\n"
            + "!" * 78
        )
        test_pcap = self.dataset_manager.generate_pipeline_verification_pcap()
        from src.flow_extractor import PCAPFlowExtractor
        extractor = PCAPFlowExtractor()
        df = extractor.extract_from_pcap(str(test_pcap))

        # Add explicit synthetic labels for 4 application categories
        # Flows are mapped to synthetic applications based on conversational profile
        apps = ["video_streaming", "web_browsing", "voip", "file_transfer"]
        df["app_label"] = [apps[i % len(apps)] for i in range(len(df))]
        df["binary_label"] = ["vpn" if "192.168.1.10" in str(ip) or "192.168.1.40" in str(ip) else "non_vpn" for ip in df["src_ip"]]
        df["label"] = df["app_label"]

        # Duplicate slightly to build multi-node graph verification topology
        df_expanded = pd.concat([df] * 5, ignore_index=True)
        for i in range(len(df_expanded)):
            df_expanded.loc[i, "flow_id"] = f"synthetic_flow_{i}"
            df_expanded.loc[i, "start_time"] += (i * 0.5)

        # Save to processed directory with clear note
        df_expanded.to_csv(csv_path, index=False)
        return df_expanded, True

    def run_experiment(
        self,
        experiment_type: str = "fine_grained_app",  # 'fine_grained_app' or 'binary_vpn'
        topology: str = DEFAULT_GRAPH_TOPOLOGY,
        split_strategy: str = "stratified",  # 'stratified' or 'unseen_host'
        include_gat: bool = True,
        force_reextract: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs the full research pipeline comparing Baselines against GraphSAGE (and GAT).
        """
        print("\n" + "=" * 78)
        print(f" STARTING EXPERIMENT: {experiment_type.upper()}")
        print(f" Graph Topology: {topology} | Split: {split_strategy}")
        print("=" * 78)

        df, is_synthetic = self.load_or_extract_data(force_reextract=force_reextract)
        if df.empty:
            raise RuntimeError("Flow extraction produced an empty DataFrame. Check your PCAP files.")

        # Determine target label column
        if experiment_type == "binary_vpn":
            label_col = "binary_label" if "binary_label" in df.columns else "label"
            exp_display_name = "Binary_VPN_vs_NonVPN"
        else:
            label_col = "app_label" if "app_label" in df.columns else "label"
            exp_display_name = "Fine_Grained_Application"

        print(f"[*] Total Flows: {len(df)}")
        print(f"[*] Target Classes: {sorted(df[label_col].unique().tolist())}")
        if is_synthetic:
            print("[!] DATA MODE: SYNTHETIC (Self-test only; real PCAPs not yet ingested)")
        else:
            print("[+] DATA MODE: REAL NETWORK TRAFFIC")

        # 1. Build Flow Graph
        print("\n[Stage 1/5] Constructing Flow Graph...")
        builder = FlowGraphBuilder(topology_mode=topology)
        graph_data: FlowGraphData = builder.create_graph_data(
            df=df,
            label_col=label_col,
            test_size=0.25,
            val_size=0.15,
            split_strategy=split_strategy,
        )

        stats = graph_data.graph_stats
        print(f"  - Graph Nodes: {stats['num_nodes']}")
        print(f"  - Graph Edges: {stats['num_edges']}")
        print(f"  - Avg Degree:  {stats['avg_degree']:.2f}")
        print(f"  - Density:     {stats['density']:.4f}")
        print(f"  - Train Nodes: {stats['train_nodes']} | Val: {stats['val_nodes']} | Test: {stats['test_nodes']}")

        # Prepare Tabular Data for Conventional Baselines (same split, same features)
        X_all = graph_data.x.numpy()
        y_all = graph_data.y.numpy()
        train_idx = graph_data.train_mask.numpy()
        test_idx = graph_data.test_mask.numpy()

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        class_names = graph_data.label_names

        results = []

        # 2. Train Tabular Baseline: Random Forest
        print("\n[Stage 2/5] Training Conventional Baseline: Random Forest...")
        rf = TabularBaseline(model_type="random_forest")
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_metrics = self.evaluator.evaluate(
            y_true=y_test,
            y_pred=rf_preds,
            class_names=class_names,
            model_name="Random Forest (Baseline)",
            experiment_name=exp_display_name,
        )
        results.append(rf_metrics)
        print(f"  -> RF Test Accuracy: {rf_metrics['accuracy']:.4f} | Macro-F1: {rf_metrics['macro_f1']:.4f}")

        # 3. Train Tabular Baseline: MLP
        print("\n[Stage 3/5] Training Conventional Baseline: Multi-Layer Perceptron (MLP)...")
        mlp = TabularBaseline(model_type="mlp")
        mlp.fit(X_train, y_train)
        mlp_preds = mlp.predict(X_test)
        mlp_metrics = self.evaluator.evaluate(
            y_true=y_test,
            y_pred=mlp_preds,
            class_names=class_names,
            model_name="MLP (Baseline)",
            experiment_name=exp_display_name,
        )
        results.append(mlp_metrics)
        print(f"  -> MLP Test Accuracy: {mlp_metrics['accuracy']:.4f} | Macro-F1: {mlp_metrics['macro_f1']:.4f}")

        # 4. Train GraphSAGE (Authentic GNN with Neighborhood Aggregation)
        print("\n[Stage 4/5] Training GraphSAGE Model (Hamilton et al., 2017)...")
        sage_model = GraphSAGEClassifier(
            in_features=graph_data.num_features,
            hidden_dim=GRAPHSAGE_CONFIG["hidden_dim"],
            embed_dim=GRAPHSAGE_CONFIG["embed_dim"],
            num_classes=graph_data.num_classes,
            dropout=GRAPHSAGE_CONFIG["dropout"],
        )
        sage_trainer = GraphSAGETrainer(sage_model, config=GRAPHSAGE_CONFIG)
        sage_history = sage_trainer.train_graph(
            x=graph_data.x,
            edge_index=graph_data.edge_index,
            y=graph_data.y,
            train_mask=graph_data.train_mask,
            val_mask=graph_data.val_mask,
        )
        sage_preds_all, sage_probs_all = sage_trainer.predict(graph_data.x, graph_data.edge_index)
        sage_test_preds = sage_preds_all[test_idx]
        sage_metrics = self.evaluator.evaluate(
            y_true=y_test,
            y_pred=sage_test_preds,
            class_names=class_names,
            model_name="GraphSAGE (GNN)",
            experiment_name=exp_display_name,
        )
        results.append(sage_metrics)
        print(f"  -> GraphSAGE Test Accuracy: {sage_metrics['accuracy']:.4f} | Macro-F1: {sage_metrics['macro_f1']:.4f}")

        # Save GraphSAGE checkpoint
        torch.save(sage_model.state_dict(), MODELS_DIR / f"graphsage_{exp_display_name}.pt")

        # 5. Optionally Train GAT (Graph Attention Network)
        if include_gat:
            print("\n[Stage 5/5] Training Graph Attention Network (GAT) (Velickovic et al., 2018)...")
            gat_model = GATClassifier(
                in_features=graph_data.num_features,
                hidden_dim=GAT_CONFIG["hidden_dim"],
                embed_dim=GAT_CONFIG["embed_dim"],
                num_classes=graph_data.num_classes,
                num_heads=GAT_CONFIG["num_heads"],
                dropout=GAT_CONFIG["dropout"],
            )
            gat_trainer = GATTrainer(gat_model, config=GAT_CONFIG)
            gat_history = gat_trainer.train_graph(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                y=graph_data.y,
                train_mask=graph_data.train_mask,
                val_mask=graph_data.val_mask,
            )
            gat_preds_all, gat_probs_all = gat_trainer.predict(graph_data.x, graph_data.edge_index)
            gat_test_preds = gat_preds_all[test_idx]
            gat_metrics = self.evaluator.evaluate(
                y_true=y_test,
                y_pred=gat_test_preds,
                class_names=class_names,
                model_name="GAT (GNN)",
                experiment_name=exp_display_name,
            )
            results.append(gat_metrics)
            print(f"  -> GAT Test Accuracy: {gat_metrics['accuracy']:.4f} | Macro-F1: {gat_metrics['macro_f1']:.4f}")
            torch.save(gat_model.state_dict(), MODELS_DIR / f"gat_{exp_display_name}.pt")

        # Generate comparative summary
        comp_df = self.evaluator.compare_models(results, experiment_name=exp_display_name)
        print("\n" + "=" * 78)
        print(f" FINAL MODEL COMPARISON ({exp_display_name})")
        print("=" * 78)
        print(comp_df.to_string(index=False))
        print("=" * 78)

        # Save individual flow test predictions
        test_flow_ids = [graph_data.flow_ids[i] for i in np.where(test_idx)[0]]
        self.evaluator.save_predictions(
            flow_ids=test_flow_ids,
            y_true=y_test,
            y_pred=sage_test_preds,
            probs=sage_probs_all[test_idx],
            class_names=class_names,
            filename=f"predictions_{exp_display_name}_graphsage.csv",
        )

        return {
            "comparison_table": comp_df,
            "results": results,
            "is_synthetic": is_synthetic,
            "graph_stats": stats,
            "class_names": class_names,
        }


def main():
    """Command-line entry point for running experiments."""
    import argparse
    parser = argparse.ArgumentParser(description="Encrypted Traffic GNN Research Pipeline")
    parser.add_argument("--experiment", type=str, default="fine_grained_app", choices=["fine_grained_app", "binary_vpn"], help="Experiment type")
    parser.add_argument("--topology", type=str, default="endpoint_sharing", choices=["endpoint_sharing", "temporal", "knn", "hybrid"], help="Graph topology")
    parser.add_argument("--split", type=str, default="stratified", choices=["stratified", "unseen_host"], help="Split strategy")
    parser.add_argument("--force_reextract", action="store_true", help="Force re-extraction of PCAPs")
    parser.add_argument("--show_catalog", action="store_true", help="Print benchmark dataset catalog")
    args = parser.parse_args()

    if args.show_catalog:
        dm = DatasetManager()
        print(dm.print_dataset_catalog())
        return

    pipeline = ExperimentPipeline()
    pipeline.run_experiment(
        experiment_type=args.experiment,
        topology=args.topology,
        split_strategy=args.split,
        force_reextract=args.force_reextract,
    )


if __name__ == "__main__":
    main()
