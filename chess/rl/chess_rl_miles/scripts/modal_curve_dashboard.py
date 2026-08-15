"""Read-only metrics and web UI for the chess RL comparison dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import modal


APP_NAME = "chess-rl-pretrain-sft-rl-comparison"
VOLUME_NAME = "chess-rl-dashboard-data"
DATA_ROOT = Path("/dashboard-data")
DATA_FILE = DATA_ROOT / "data.json"
_THIS_FILE = Path(__file__).resolve()
DASHBOARD_ROOT = (
    _THIS_FILE.parents[2] / "dashboard"
    if len(_THIS_FILE.parents) > 2
    else Path("/site")
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi==0.116.1")
dashboard_image = (
    modal.Image.from_registry("node:22-slim", add_python="3.12")
    .add_local_dir(
        str(DASHBOARD_ROOT),
        remote_path="/site",
        copy=True,
        ignore=[
            ".git/**",
            ".wrangler/**",
            "dist/**",
            "node_modules/**",
            "public/data.json",
            "public/data.json.tmp",
        ],
    )
    .run_commands("cd /site && npm ci", "cd /site && npm run build")
)


@app.function(
    image=image,
    volumes={str(DATA_ROOT): data_volume},
    min_containers=1,
    timeout=60,
)
@modal.fastapi_endpoint(method="GET")
def metrics():
    from fastapi.responses import JSONResponse

    data_volume.reload()
    if not DATA_FILE.is_file():
        return JSONResponse(
            {"error": "Dashboard telemetry has not been published yet."},
            status_code=503,
            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"},
        )
    payload = json.loads(DATA_FILE.read_text())
    return JSONResponse(
        payload,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-store, max-age=0",
        },
    )


@app.function(
    image=dashboard_image,
    min_containers=1,
    timeout=24 * 60 * 60,
)
@modal.web_server(3000, startup_timeout=180)
def dashboard():
    subprocess.Popen(
        [
            "npm",
            "run",
            "start",
            "--",
            "--hostname",
            "0.0.0.0",
            "--port",
            "3000",
        ],
        cwd="/site",
    )
