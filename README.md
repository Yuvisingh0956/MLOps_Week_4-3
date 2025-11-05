# 🌐 IRIS MLOps Pipeline — CI/CD + Scaling & Load Testing on Kubernetes

## 🎯 Objective

This project demonstrates a complete **MLOps pipeline** for the IRIS classification model — covering **Continuous Integration (CI)**, **Continuous Deployment (CD)**, and **Scaling under Load** using **GitHub Actions**, **Docker**, **Google Cloud (Artifact Registry + GKE)**, and **Kubernetes Autoscaling**.

---

## 🧩 Pipeline Overview

  ┌────────────────────────────────────────────┐
  │                GitHub Repo                 │
  │  (dev & main branches + CI/CD workflows)   │
  └────────────────────────────┬───────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────┐
  │          GitHub Actions Workflows          │
  │  • CI - Test model with pytest + DVC       │
  │  • CD - Build & Deploy API on GKE          │
  │  • LoadTest - wrk-based scaling analysis   │
  └────────────────────────────┬───────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────┐
  │           Google Cloud Platform (GCP)      │
  │  • Artifact Registry (Docker images)       │
  │  • GKE (Iris API Deployment)               │
  │  • GCS (DVC remote storage)                │
  │  • HPA (Autoscaling test)                  │
  └────────────────────────────────────────────┘

---

## 🧱 Folder Structure

iris-mlops/
├── src/
│ └── train.py # Model training and saving
├── app.py # Flask API serving predictions
├── Dockerfile # Container configuration
├── requirements.txt # Dependencies
├── models/model.pkl # Tracked model (via DVC)
├── data/data.csv # Dataset (via DVC)
├── k8s/
│ ├── deployment.yaml # API Deployment manifest
│ ├── service.yaml # LoadBalancer Service
│ ├── hpa-max3.yaml # Autoscaling (max=3)
│ ├── hpa-max1.yaml # Restricted scaling (max=1)
│ ├── wrk-configmap.yaml # Lua script for wrk
│ ├── job-wrk-1000.yaml # Stress test: 1000 concurrent reqs
│ └── job-wrk-2000.yaml # Stress test: 2000 concurrent reqs
└── .github/workflows/
├── ci.yml # Continuous Integration
├── cd.yml # Continuous Deployment
└── cd_loadtest.yml # Load Test + Autoscaling


---

## 🧠 Features

| Component | Description |
|------------|-------------|
| **CI (ci.yml)** | Validates model, runs pytest, and checks DVC data integrity |
| **CD (cd.yml)** | Builds Docker image, pushes to Artifact Registry, deploys to GKE |
| **Sanity (sanity.yml)** | Verifies GCS bucket access, DVC remote, and environment setup |
| **Load Test (cd_loadtest.yml)** | Runs wrk-based stress tests and demonstrates autoscaling |
| **HPA (hpa-max3 / hpa-max1)** | Scales pods dynamically or restricts to show bottlenecks |

---

## ⚙️ Setup Instructions

### 1️⃣ Google Cloud Setup

```bash
# Set project
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export REPO="iris-artifact-repo"
export CLUSTER_NAME="iris-gke-cluster"
export CLUSTER_ZONE="us-central1-a"

gcloud config set project $PROJECT_ID

# Create GCS bucket for DVC:
gsutil mb -p $PROJECT_ID -l $REGION gs://iris-dvc-bucket/

# Artifact Registry:
gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION

# GKE Cluster
gcloud container clusters create $CLUSTER_NAME \
  --zone $CLUSTER_ZONE \
  --num-nodes=2

# Service Account for GitHub Actions:
gcloud iam service-accounts create gha-deployer
SA="gha-deployer@$PROJECT_ID.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/container.admin"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/storage.admin"

gcloud iam service-accounts keys create gha-deployer-key.json \
  --iam-account=$SA

# Add the following secrets in GitHub → Settings → Secrets → Actions:
GCP_SA_KEY
GCP_PROJECT
GCP_REGION
GKE_CLUSTER_NAME
GKE_CLUSTER_ZONE
ARTIFACT_REGISTRY_REPOSITORY
GCPKEY
BUCKET_NAME
```

## 🚀 Continuous Integration (ci.yml)
- Trigger: On push or PR to dev/main
- Purpose: Run DVC pull, unit tests (pytest), and generate markdown report via CML.
- Output: Commented test results on Pull Request.

## 🧰 Continuous Deployment (cd.yml)

- Trigger: Push to main
- Purpose: Build Docker image → Push to Artifact Registry → Deploy to GKE
- Tools: google-github-actions/auth, get-gke-credentials, kubectl apply.

## ⚡ Load Testing & Autoscaling (cd_loadtest.yml)

- Trigger: Push to main or manual dispatch
Steps:

    -- DVC Pull (get model)

    -- Build and push Docker image

    -- Deploy to GKE

    -- Apply HPA (max=3) → Run wrk job (1000 concurrency)

    -- Apply restricted HPA (max=1) → Run wrk job (2000 concurrency)

    -- Logs and metrics printed via kubectl

## 🧩 Example Results
| Scenario        | Concurrency | Pods | Latency | Throughput | Observation        |
| --------------- | ----------- | ---- | ------- | ---------- | ------------------ |
| Autoscaling ON  | 1000        | 3    | Stable  | High       | Scales smoothly    |
| Autoscaling OFF | 2000        | 1    | High    | Low        | Bottleneck appears |

## wrk Output Example
wrk -t4 -c100 -d30s --latency -s post.lua http://35.193.14.40:80/predict
Running 30s test @ http://35.193.14.40:80/predict
  4 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   199.16ms   28.40ms 438.98ms   70.08%
    Req/Sec    70.50     25.55   177.00     72.03%
  Latency Distribution
     50%  196.18ms
     75%  212.01ms
     90%  231.37ms
     99%  300.66ms
  8410 requests in 30.08s, 1.58MB read
Requests/sec:    279.55
Transfer/sec:     53.78KB

## 🧩 Key Learnings

- ✅ CI ensures code, model, and data integrity before deployment

- ✅ CD automates delivery using GitHub → GCP → GKE

- ✅ Autoscaling maintains stability under heavy load

- ✅ Restricting pods shows clear CPU and latency bottlenecks

- ✅ End-to-end automation builds a resilient MLOps pipeline

## 🧾 Summary
| Component           | Status         |
| ------------------- | -------------- |
| CI (pytest + DVC)   | ✅ Implemented  |
| CD (Docker + GKE)   | ✅ Implemented  |
| Load Testing (wrk)  | ✅ Implemented  |
| Autoscaling (HPA)   | ✅ Demonstrated |
| Bottleneck Analysis | ✅ Observed     |
| Sanity Validation   | ✅ Done         |

