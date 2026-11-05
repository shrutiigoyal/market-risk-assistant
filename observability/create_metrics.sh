#!/usr/bin/env bash
# Log-based metrics for the dashboard. Run once after the first deploy.
set -euo pipefail
: "${PROJECT_ID:?Set PROJECT_ID}"
SVC='resource.type="cloud_run_revision" AND resource.labels.service_name="market-risk-assistant"'

gcloud logging metrics create agent_errors --project "$PROJECT_ID" \
  --description="Agent invocation failures" \
  --log-filter="$SVC AND jsonPayload.event=\"agent_error\""

gcloud logging metrics create guardrail_blocks --project "$PROJECT_ID" \
  --description="Requests/responses blocked by guardrails" \
  --log-filter="$SVC AND jsonPayload.event=\"guardrail_blocked\""

# Distribution metric on latency (ms), taken from the agent_response log field
cat > /tmp/latency_metric.json <<JSON
{
  "name": "agent_latency_ms",
  "description": "End-to-end agent latency",
  "filter": "$(echo "$SVC" | sed 's/"/\\"/g') AND jsonPayload.event=\"agent_response\"",
  "valueExtractor": "EXTRACT(jsonPayload.latency_ms)",
  "metricDescriptor": {"metricKind": "DELTA", "valueType": "DISTRIBUTION", "unit": "ms"},
  "bucketOptions": {"exponentialBuckets": {"numFiniteBuckets": 30, "growthFactor": 1.4, "scale": 100}}
}
JSON
gcloud logging metrics create agent_latency_ms --project "$PROJECT_ID" --config-from-file=/tmp/latency_metric.json
echo "Created metrics. Build a Cloud Monitoring dashboard from logging.googleapis.com/user/* metrics."
