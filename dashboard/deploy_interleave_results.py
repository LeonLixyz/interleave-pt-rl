"""Deploy the static results-summary page as a Modal web endpoint.

    modal deploy dashboard/deploy_interleave_results.py
"""
from pathlib import Path

import modal

LOCAL_HTML = Path(__file__).parent / "results_summary.html"
REMOTE_HTML = "/assets/results_summary.html"

app = modal.App("interleave-results")
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
    api = FastAPI(title="interleave-results")

    @api.get("/", response_class=HTMLResponse)
    def index():
        return html

    @api.get("/healthz")
    def healthz():
        return {"ok": True}

    return api
