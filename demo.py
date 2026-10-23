"""
Encrypted Traffic Classification — emergency runnable prototype
NOTE: Uses SYNTHETIC flow metadata, not real PCAP captures.
It demonstrates feature processing, graph construction, neighbor aggregation,
classification, and evaluation. It is not evidence of real-world performance.
Run: python demo.py
"""
import csv, math, random
from collections import Counter, defaultdict

SEED = 42
CLASSES = ["video_streaming", "web_browsing", "messaging", "file_transfer"]
FEATURES = [
    "mean_packet_size", "packet_size_std", "mean_iat_ms",
    "packets_per_flow", "out_in_ratio", "duration_ms"
]

# Synthetic class profiles (feature units are illustrative).
PROFILES = {
    "video_streaming": [1250, 260, 18, 95, 3.4, 24000],
    "web_browsing":   [720, 420, 85, 28, 1.2, 8500],
    "messaging":       [180, 110, 260, 12, 0.8, 6000],
    "file_transfer":   [1380, 180, 9, 150, 0.55, 42000],
}

def make_dataset(n_per_class=100, seed=SEED):
    rng = random.Random(seed)
    rows = []
    for label, center in PROFILES.items():
        for i in range(n_per_class):
            # Positive, noisy metadata; no payload bytes are generated.
            vals = []
            for j, mu in enumerate(center):
                sd = max(abs(mu) * 0.22, 1.0)
                value = rng.gauss(mu, sd)
                if j in (0, 1, 2, 3, 5):
                    value = max(1.0, value)
                else:
                    value = max(0.05, value)
                vals.append(value)
            rows.append({"id": f"{label}_{i}", "label": label,
                         "features": vals})
    rng.shuffle(rows)
    return rows

def distance(a, b):
    # Scale features by profile ranges to prevent duration dominating.
    scales = [1500, 500, 300, 180, 4, 45000]
    return math.sqrt(sum(((x-y)/s)**2 for x,y,s in zip(a,b,scales)))

def build_graph(rows, k=5):
    """Undirected k-nearest-neighbor graph over flow metadata."""
    n = len(rows)
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        ds = [(distance(rows[i]["features"], rows[j]["features"]), j)
              for j in range(n) if j != i]
        ds.sort()
        for _, j in ds[:k]:
            neighbors[i].append(j)
            neighbors[j].append(i)
    # Deduplicate symmetric additions.
    return [sorted(set(x)) for x in neighbors]

def aggregate_features(rows, graph):
    result = []
    for i, row in enumerate(rows):
        ns = graph[i]
        if not ns:
            result.append(row["features"][:])
            continue
        means = [sum(rows[j]["features"][f] for j in ns)/len(ns)
                 for f in range(len(FEATURES))]
        # Graph representation = original node features + neighbor mean.
        result.append(row["features"] + means)
    return result

def split(rows, seed=SEED):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    train, test = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        cut = int(len(group)*0.75)
        train.extend(group[:cut]); test.extend(group[cut:])
    rng.shuffle(train); rng.shuffle(test)
    return train, test

def fit_centroids(train_rows, vectors):
    centers = {}
    for label in CLASSES:
        group = [vectors[i] for i,r in enumerate(train_rows) if r["label"] == label]
        centers[label] = [sum(v[j] for v in group)/len(group)
                          for j in range(len(group[0]))]
    return centers

def predict(vector, centers):
    return min(centers, key=lambda label: distance(vector, centers[label]))

def evaluate(test_rows, test_vectors, centers):
    correct = 0
    confusion = {c: Counter() for c in CLASSES}
    for row, vec in zip(test_rows, test_vectors):
        pred = predict(vec, centers)
        confusion[row["label"]][pred] += 1
        correct += pred == row["label"]
    accuracy = correct/len(test_rows)
    per_class = {}
    for c in CLASSES:
        tp = confusion[c][c]
        fp = sum(confusion[other][c] for other in CLASSES if other != c)
        fn = sum(confusion[c][other] for other in CLASSES if other != c)
        precision = tp/(tp+fp) if tp+fp else 0
        recall = tp/(tp+fn) if tp+fn else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
        per_class[c] = (precision, recall, f1)
    macro_f1 = sum(x[2] for x in per_class.values())/len(CLASSES)
    return accuracy, macro_f1, confusion, per_class

def main():
    rows = make_dataset()
    train, test = split(rows)
    # Graph is constructed over all rows for a compact demo. For rigorous evaluation,
    # graph construction must avoid test-to-train leakage and use a defined protocol.
    graph = build_graph(rows, k=5)
    vectors = aggregate_features(rows, graph)
    index = {id(row): i for i,row in enumerate(rows)}
    train_vectors = [vectors[index[id(r)]] for r in train]
    test_vectors = [vectors[index[id(r)]] for r in test]
    centers = fit_centroids(train, train_vectors)
    acc, mf1, cm, pc = evaluate(test, test_vectors, centers)

    print("="*62)
    print(" ENCRYPTED TRAFFIC CLASSIFICATION — GRAPH PROTOTYPE")
    print("="*62)
    print("Dataset: synthetic metadata (NOT real PCAP traffic)")
    print(f"Flows: {len(rows)} | Train: {len(train)} | Test: {len(test)}")
    print(f"Graph: k-NN | k=5 | nodes={len(graph)} | edges={sum(map(len,graph))//2}")
    print("Classifier: nearest class centroid over node + neighbor-mean features")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro-F1: {mf1:.3f}")
    print("\nPer-class precision / recall / F1:")
    for c,(p,r,f) in pc.items():
        print(f"  {c:18s} P={p:.3f} R={r:.3f} F1={f:.3f}")
    print("\nConfusion matrix (actual rows, predicted columns):")
    print("actual\\pred      " + " ".join(f"{c[:7]:>8s}" for c in CLASSES))
    for c in CLASSES:
        print(f"{c[:14]:14s} " + " ".join(f"{cm[c][p]:8d}" for p in CLASSES))
    print("\nDemo prediction for one synthetic flow:")
    sample = {"features": [1100, 230, 22, 90, 2.8, 21000]}
    # Include one-node representation; graph aggregation is unavailable for a new
    # isolated sample, so classify with its raw features + itself as neighbor mean.
    vector = sample["features"] + sample["features"]
    print("  predicted class:", predict(vector, centers))
    print("\nLIMITATION: synthetic-data proof-of-pipeline only. Do not report these")
    print("metrics as real-world encrypted traffic results.")
    print("="*62)

if __name__ == "__main__":
    main()
