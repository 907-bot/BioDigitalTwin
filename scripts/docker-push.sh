#!/usr/bin/env bash
set -euo pipefail
NS="${1:-abhishekadari}"

echo "==> Tagging images as $NS/bdt-*:latest"
docker tag bio-digital-twin-api:latest         "$NS/bdt-api:latest"
docker tag bio-digital-twin-frontend:latest     "$NS/bdt-frontend:latest"
docker tag bio-digital-twin-ontology-svc:latest "$NS/bdt-ontology-svc:latest"
docker tag bio-digital-twin-knowledge-svc:latest "$NS/bdt-knowledge-svc:latest"
docker tag bio-digital-twin-patient-svc:latest  "$NS/bdt-patient-svc:latest"
docker tag bio-digital-twin-twin-svc:latest     "$NS/bdt-twin-svc:latest"
docker tag bio-digital-twin-agent-svc:latest    "$NS/bdt-agent-svc:latest"
docker tag bio-digital-twin-narrative-svc:latest "$NS/bdt-narrative-svc:latest"

echo "==> Pushing..."
docker push "$NS/bdt-api:latest"
docker push "$NS/bdt-frontend:latest"
docker push "$NS/bdt-ontology-svc:latest"
docker push "$NS/bdt-knowledge-svc:latest"
docker push "$NS/bdt-patient-svc:latest"
docker push "$NS/bdt-twin-svc:latest"
docker push "$NS/bdt-agent-svc:latest"
docker push "$NS/bdt-narrative-svc:latest"

echo "==> Done! Images pushed to $NS/*"
echo "==> Recipient runs: docker compose -f docker-compose.registry.yml up -d"
