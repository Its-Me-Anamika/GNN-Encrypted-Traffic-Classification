import numpy as np

def compute_classification_metrics(y_true, y_pred, classes):
    """
    Computes comprehensive multi-class classification evaluation metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num_classes = len(classes)

    # Initialize Confusion Matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf_matrix[t, p] += 1

    total_samples = len(y_true)
    correct = np.trace(conf_matrix)
    accuracy = float(correct / total_samples) if total_samples > 0 else 0.0

    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    support_weights = []

    for i, cls_name in enumerate(classes):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        support = np.sum(conf_matrix[i, :])

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        support_weights.append(support)

        per_class[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": int(support)
        }

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0

    total_support = sum(support_weights)
    weighted_f1 = float(sum(f * w for f, w in zip(f1s, support_weights)) / total_support) if total_support > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "precision": round(macro_precision, 4),
        "recall": round(macro_recall, 4),
        "per_class_metrics": per_class,
        "confusion_matrix": conf_matrix.tolist(),
        "classes": classes
    }

def plot_confusion_matrix_data(conf_matrix, classes):
    """Formats confusion matrix for JSON transmission and charting."""
    return {
        "labels": classes,
        "matrix": conf_matrix
    }
