# Installation & Setup Guide

This guide provides step-by-step instructions for installing and running the **Fine-Grained Encrypted Traffic Classification System**.

---

## 🛠️ Prerequisites

- **Python**: Version 3.10 or higher
- **Node.js**: Version 18.x or higher
- **Git**: Installed
- **Docker & Docker Compose**: (Optional, for containerized deployment)

---

## 📦 Step-by-Step Installation

### Step 1: Clone or Navigate to Project
```bash
cd D:\Users\ANAMIKA\Documents\gnn-encrypted-traffic
```

### Step 2: Environment Setup
Copy the environment example files:
```bash
# Copy Backend env
cp backend/.env.example backend/.env

# Copy Frontend env
cp frontend/.env.example frontend/.env
```

---

### Step 3: Install & Run Backend
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Bootstrap database and model weights
python synthetic_data.py

# Launch FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### Step 4: Install & Run Frontend
In a new terminal window:
```bash
cd frontend

# Install packages
npm install

# Start React app
npm start
```

Open your web browser at [http://localhost:3000](http://localhost:3000).

---

## 🧪 Running Machine Learning Benchmarks & CLI Scripts

```bash
# Train GraphSAGE GNN Model
python ml/train.py --model GraphSAGE --epochs 50

# Train GAT GNN Model
python ml/train.py --model GAT --epochs 50

# Run Model Evaluation Benchmark against Random Forest
python ml/evaluate.py

# Test Standalone CLI Flow Inference
python ml/inference.py --category Streaming
```
