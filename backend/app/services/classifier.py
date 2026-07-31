import os
import torch
import torch.nn.functional as F
import numpy as np

from app.ml.models import EncryptedTrafficGNN
from app.services.feature_extractor import FeatureExtractor
from app.services.graph_builder import GraphBuilder
from app.services.explainer import GNNExplainabilityService
from app.config import settings

class TrafficClassifierService:
    """
    Main inference service handling model loading, feature scaling,
    Monte Carlo Dropout uncertainty estimation, Unknown class thresholding, and caching.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncryptedTrafficGNN(
            architecture=settings.MODEL_TYPE,
            in_channels=settings.NUM_FEATURES,
            out_channels=len(settings.CLASSES),
            dropout_rate=0.3
        ).to(self.device)

        self.feature_extractor = FeatureExtractor()
        self.explainer = GNNExplainabilityService(self.model)
        self.cache = {}
        self.is_loaded = False

        # Attempt to load model checkpoint if exists
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(settings.MODEL_PATH):
                checkpoint = torch.load(settings.MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                self.is_loaded = True
                print(f"[+] Loaded GNN model from {settings.MODEL_PATH}")
            else:
                print(f"[*] Pre-trained model not found at {settings.MODEL_PATH}. Operating with initialized GNN weights.")
                self.model.eval()
                self.is_loaded = True
        except Exception as e:
            print(f"[!] Error loading model: {e}. Falling back to default mode.")
            self.model.eval()
            self.is_loaded = True

    def classify_flow(self, flow_input):
        """
        Classifies a single flow or PCAP snippet, applying Monte Carlo Dropout uncertainty
        and GNNExplainer attribution.
        """
        # Step 1: Feature Extraction
        if isinstance(flow_input, dict):
            features = self.feature_extractor.extract_from_dict(flow_input)
            flow_meta = flow_input
        else:
            features = self.feature_extractor.extract_from_dict({"is_vpn": False})
            flow_meta = {}

        x = torch.tensor(np.array([features], dtype=np.float32), dtype=torch.float32).to(self.device)
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(self.device)

        # Step 2: Monte Carlo Dropout Inference for Uncertainty
        predictions, confidences, uncertainty_scores, mean_probs = self.model.predict_with_mc_dropout(
            x, edge_index, num_samples=settings.MC_DROPOUT_SAMPLES
        )

        pred_idx = int(predictions[0].item())
        confidence = float(confidences[0].item())
        uncertainty = float(uncertainty_scores[0].item())

        prob_dict = {}
        probs_array = mean_probs[0].cpu().numpy()
        for i, cls_name in enumerate(settings.CLASSES):
            prob_dict[cls_name] = round(float(probs_array[i]), 4)

        # Step 3: Unknown Traffic Detection Thresholding
        if confidence < settings.CONFIDENCE_THRESHOLD or uncertainty > 0.08:
            predicted_category = "Unknown"
            is_unknown = True
        else:
            predicted_category = settings.CLASSES[pred_idx] if pred_idx < len(settings.CLASSES) else "Unknown"
            is_unknown = False

        # Step 4: GNN Attribution
        explanation = self.explainer.explain_flow(x, edge_index, target_node_idx=0)

        flow_id = f"flow_{np.random.randint(100000, 999999)}"

        return {
            "flow_id": flow_id,
            "predicted_category": predicted_category,
            "confidence": round(confidence, 4),
            "uncertainty_score": round(uncertainty, 4),
            "is_unknown": is_unknown,
            "class_probabilities": prob_dict,
            "explanation": explanation,
            "features": features
        }

    def classify_graph_batch(self, flows_list):
        """
        Classifies a full graph batch of network flows concurrently.
        """
        graph_data = GraphBuilder.build_flow_graph(flows_list)
        x = graph_data.x.to(self.device) if hasattr(graph_data, "x") else graph_data["x"].to(self.device)
        edge_index = graph_data.edge_index.to(self.device) if hasattr(graph_data, "edge_index") else graph_data["edge_index"].to(self.device)

        predictions, confidences, uncertainty_scores, mean_probs = self.model.predict_with_mc_dropout(
            x, edge_index, num_samples=settings.MC_DROPOUT_SAMPLES
        )

        results = []
        num_nodes = x.size(0)

        for i in range(num_nodes):
            pred_idx = int(predictions[i].item())
            conf = float(confidences[i].item())
            unc = float(uncertainty_scores[i].item())

            prob_dict = {
                cls_name: round(float(mean_probs[i, idx].item()), 4)
                for idx, cls_name in enumerate(settings.CLASSES)
            }

            if conf < settings.CONFIDENCE_THRESHOLD or unc > 0.08:
                cat = "Unknown"
                is_unk = True
            else:
                cat = settings.CLASSES[pred_idx] if pred_idx < len(settings.CLASSES) else "Unknown"
                is_unk = False

            exp = self.explainer.explain_flow(x, edge_index, target_node_idx=i)

            results.append({
                "predicted_category": cat,
                "confidence": round(conf, 4),
                "uncertainty_score": round(unc, 4),
                "is_unknown": is_unk,
                "class_probabilities": prob_dict,
                "explanation": exp
            })

        return results

# Singleton Classifier Service Instance
classifier_service = TrafficClassifierService()
