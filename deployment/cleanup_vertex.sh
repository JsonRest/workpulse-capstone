#!/bin/bash
# ============================================================
# WorkPulse — Clean Up Vertex AI Resources
# ============================================================
# IMPORTANT: Run this when done testing to avoid ongoing charges!
#
# Usage:
#   chmod +x cleanup_vertex.sh
#   ./cleanup_vertex.sh
# ============================================================

set -e

# ── CONFIGURATION (must match deploy_vertex.sh) ──────────────
PROJECT_ID="your-project-id"          # <-- CHANGE THIS
REGION="us-central1"
REPO_NAME="workpulse-repo"

gcloud config set project $PROJECT_ID

echo "============================================"
echo "WorkPulse Vertex AI Cleanup"
echo "============================================"

# List current resources
echo ""
echo "Current endpoints:"
gcloud ai endpoints list --region=$REGION --format="table(name, displayName)" 2>/dev/null || echo "  None found"

echo ""
echo "Current models:"
gcloud ai models list --region=$REGION --format="table(name, displayName)" 2>/dev/null || echo "  None found"

echo ""
read -p "Enter ENDPOINT_ID to clean up (or 'skip'): " ENDPOINT_ID
read -p "Enter MODEL_ID to clean up (or 'skip'): " MODEL_ID

# Undeploy all models from endpoint
if [ "$ENDPOINT_ID" != "skip" ] && [ -n "$ENDPOINT_ID" ]; then
    echo ""
    echo "Undeploying models from endpoint ${ENDPOINT_ID}..."
    
    # Get deployed model IDs
    DEPLOYED_MODELS=$(gcloud ai endpoints describe $ENDPOINT_ID \
        --region=$REGION \
        --format="value(deployedModels.id)" 2>/dev/null)
    
    for DM_ID in $DEPLOYED_MODELS; do
        echo "  Undeploying deployed model: $DM_ID"
        gcloud ai endpoints undeploy-model $ENDPOINT_ID \
            --region=$REGION \
            --deployed-model-id=$DM_ID \
            --quiet
    done
    
    echo "  Deleting endpoint..."
    gcloud ai endpoints delete $ENDPOINT_ID --region=$REGION --quiet
    echo "  Endpoint deleted ✅"
fi

# Delete model
if [ "$MODEL_ID" != "skip" ] && [ -n "$MODEL_ID" ]; then
    echo ""
    echo "Deleting model ${MODEL_ID}..."
    gcloud ai models delete $MODEL_ID --region=$REGION --quiet
    echo "  Model deleted ✅"
fi

# Optionally delete Artifact Registry
echo ""
read -p "Delete Artifact Registry repo '${REPO_NAME}'? (y/n): " DELETE_REPO
if [ "$DELETE_REPO" = "y" ]; then
    gcloud artifacts repositories delete $REPO_NAME \
        --location=$REGION --quiet
    echo "  Repository deleted ✅"
fi

echo ""
echo "============================================"
echo "Cleanup complete! No more charges."
echo "============================================"
