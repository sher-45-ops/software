#!/usr/bin/env bash
# Drive Recon3D over HTTP. Start the server first:  recon3d serve --port 8760
set -euo pipefail
BASE=${BASE:-http://127.0.0.1:8760}
REFS=${REFS:-./demo/refs}

curl -s "$BASE/health" | python -m json.tool

curl -s -X POST "$BASE/v1/projects" -H 'Content-Type: application/json' \
  -d '{"name":"api-example","subject_type":"robot"}' | python -m json.tool

for f in "$REFS"/*.png; do
  curl -s -X POST "$BASE/v1/projects/api-example/images" -F "files=@$f" > /dev/null
done
echo "uploaded $(ls "$REFS"/*.png | wc -l) image(s)"

JOB=$(curl -s -X POST "$BASE/v1/projects/api-example/reconstruct" \
  -H 'Content-Type: application/json' \
  -d '{"preset":"draft","texture_resolution":1024,"export_formats":["glb","obj"]}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["job"]["id"])')
echo "job: $JOB"

while true; do
  STATE=$(curl -s "$BASE/v1/jobs/$JOB" | python -c 'import json,sys; print(json.load(sys.stdin)["state"])')
  echo "  state: $STATE"
  [ "$STATE" = "completed" ] || [ "$STATE" = "failed" ] || [ "$STATE" = "cancelled" ] || { sleep 5; continue; }
  break
done

curl -s "$BASE/v1/jobs/$JOB/artifacts" | python -c '
import json,sys
d = json.load(sys.stdin)
for f in d["files"]:
    if f["path"].startswith(("final/", "textures/")):
        print(f"{f[\"bytes\"]:>10}  {f[\"path\"]}")'
