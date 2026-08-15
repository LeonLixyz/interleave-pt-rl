"""Deploy the interleaved pretrain<->RL dashboard as a Modal web endpoint.

Serves dashboard/interleaved_dashboard.html at a public URL.
The HTML carries an embedded data snapshot; redeploy to refresh it
(regenerate the HTML from INTERLEAVED_CORE_REGISTRY.json first).

    modal deploy dashboard/deploy_interleave_pt_rl.py
"""
from pathlib import Path

import modal

LOCAL_HTML = Path(__file__).parent / "interleaved_dashboard.html"
REMOTE_HTML = "/assets/interleaved_dashboard.html"

app = modal.App("interleave-pt-rl")
image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]")
    .add_local_file(LOCAL_HTML, REMOTE_HTML, copy=True)
)


@app.function(image=image, min_containers=1)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    html = Path(REMOTE_HTML).read_text()
    api = FastAPI(title="interleave-pt-rl")

    @api.get("/", response_class=HTMLResponse)
    def index():
        return html

    @api.get("/healthz")
    def healthz():
        return {"ok": True}

    return api
