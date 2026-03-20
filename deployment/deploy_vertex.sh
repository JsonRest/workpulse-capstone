#!/bin/bash
# ============================================================
# WorkPulse — Deploy to GCP Vertex AI (Option B: Custom Container)
# ============================================================
# Usage:
#   chmod +x deploy_vertex.sh
#   ./deploy_vertex.sh
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Docker installed
#   - models/best_model.pkl exists (run Step 4 notebook first)
# ============================================================

set -e  # Exit on error

# ── CONFIGURATION (edit these) ────────────────────────────────
PROJECT_ID="your-project-id"          # <-- CHANGE THIS
REGION="us-central1"
REPO_NAME="workpulse-repo"
IMAGE_NAME="workpulse-api"
IMAGE_TAG="v1"
ENDPOINT_NAME="workpulse-endpoint"
MODEL_NAME="workpulse-custom-v1"
MACHINE_TYPE="n1-standard-2"

# Derived variables
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "============================================"
echo "WorkPulse Vertex AI Deployment"
echo "============================================"
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Image:    ${IMAGE_URI}"
echo "============================================"

# ── STEP 1: Enable APIs ──────────────────────────────────────
echo ""
echo "[1/7] Enabling APIs..."
gcloud config set project $PROJECT_ID
gcloud services enable aiplatform.googleapis.com
gcloud services enable artifactregistry.googleapis.com
echo "  Done ✅"

# ── STEP 2: Create Artifact Registry ─────────────────────────
echo ""
echo "[2/7] Creating Artifact Registry repository..."
gcloud artifacts repositories create $REPO_NAME \
  --repository-format=docker \
  --location=$REGION \
  --description="WorkPulse Docker images" \
  2>/dev/null || echo "  Repository already exists, skipping."
echo "  Done ✅"

# ── STEP 3: Build Docker image ───────────────────────────────
echo ""
echo "[3/7] Building Docker image..."
docker build -t $IMAGE_URI .
echo "  Done ✅"

# ── STEP 4: Push to Artifact Registry ────────────────────────
echo ""
echo "[4/7] Pushing image to Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
docker push $IMAGE_URI
echo "  Done ✅"

# ── STEP 5: Upload model to Vertex AI ────────────────────────
echo ""
echo "[5/7] Uploading model to Vertex AI Model Registry..."
MODEL_ID=$(gcloud ai models upload \
  --region=$REGION \
  --display-name=$MODEL_NAME \
  --container-image-uri=$IMAGE_URI \
  --container-health-route=/health \
  --container-predict-route=/predict \
  --container-ports=8080 \
  --format="value(model)" 2>&1 | grep -oP 'models/\K[0-9]+' | head -1)

if [ -z "$MODEL_ID" ]; then
  echo "  Uploading... (check Vertex AI console for model ID)"
  echo "  Run: gcloud ai models list --region=$REGION"
else
  echo "  Model ID: $MODEL_ID ✅"
fi

# ── STEP 6: Create endpoint ──────────────────────────────────
echo ""
echo "[6/7] Creating endpoint..."
ENDPOINT_ID=$(gcloud ai endpoints create \
  --region=$REGION \
  --display-name=$ENDPOINT_NAME \
  --format="value(name)" 2>&1 | grep -oP 'endpoints/\K[0-9]+' | head -1)

if [ -z "$ENDPOINT_ID" ]; then
  echo "  Creating... (check Vertex AI console for endpoint ID)"
  echo "  Run: gcloud ai endpoints list --region=$REGION"
else
  echo "  Endpoint ID: $ENDPOINT_ID ✅"
fi

# ── STEP 7: Deploy model to endpoint ─────────────────────────
echo ""
echo "[7/7] Deploying model to endpoint..."
echo "  This takes 5-10 minutes..."

if [ -n "$MODEL_ID" ] && [ -n "$ENDPOINT_ID" ]; then
  gcloud ai endpoints deploy-model $ENDPOINT_ID \
    --region=$REGION \
    --model=$MODEL_ID \
    --display-name=workpulse-deployment \
    --machine-type=$MACHINE_TYPE \
    --min-replica-count=1 \
    --max-replica-count=3 \
    --traffic-split=0=100
  echo "  Done ✅"
else
  echo "  Could not auto-detect IDs. Run these manually:"
  echo ""
  echo "  # Get your IDs from:"
  echo "  gcloud ai models list --region=$REGION"
  echo "  gcloud ai endpoints list --region=$REGION"
  echo ""
  echo "  # Then deploy:"
  echo "  gcloud ai endpoints deploy-model ENDPOINT_ID \\"
  echo "    --region=$REGION \\"
  echo "    --model=MODEL_ID \\"
  echo "    --display-name=workpulse-deployment \\"
  echo "    --machine-type=$MACHINE_TYPE \\"
  echo "    --min-replica-count=1 \\"
  echo "    --max-replica-count=3 \\"
  echo "    --traffic-split=0=100"
fi

echo ""
echo "============================================"
echo "Deployment complete!"
echo "============================================"
echo ""
echo "Test with:"
echo "  python test_vertex_endpoint.py"
echo ""
echo "IMPORTANT: Undeploy when done to avoid charges:"
echo "  gcloud ai endpoints undeploy-model ENDPOINT_ID --region=$REGION --deployed-model-id=DEPLOYED_MODEL_ID"
echo "  gcloud ai endpoints delete ENDPOINT_ID --region=$REGION"
echo "  gcloud ai models delete MODEL_ID --region=$REGION"
