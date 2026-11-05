#!/usr/bin/env bash
# Deploy to Cloud Run from source (Cloud Build builds the Dockerfile).
# Usage: PROJECT_ID=my-proj ./deploy.sh
set -euo pipefail
: "${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-market-risk-assistant}"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  aiplatform.googleapis.com logging.googleapis.com

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION,MODEL=${MODEL:-gemini-3.1-flash-lite}" \
  --memory 512Mi --cpu 1 --max-instances 3 --timeout 120 \
  --no-allow-unauthenticated   # LLM endpoints cost money: require IAM auth

# The Cloud Run service account needs Vertex AI access:
SA="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(spec.template.spec.serviceAccountName)')"
if [ -z "$SA" ]; then
  SA="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
fi
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:$SA" --role roles/aiplatform.user --condition=None >/dev/null

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo "Deployed: $URL"
echo "Try: curl -s -X POST $URL/chat -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" \\"
echo "       -H 'Content-Type: application/json' -d '{\"message\":\"Compute 99% VaR for a \$5M portfolio\"}'"
