"""Serve the training dashboard as a public Modal web endpoint.

Deploy:
    modal deploy dashboard_serve.py

Refresh HTML (after editing the local file):
    modal deploy dashboard_serve.py   # re-uploads and re-caches

The URL is printed on deploy; it's stable across redeploys.
"""

from __future__ import annotations

import modal

LOCAL_HTML = "/private/tmp/claude-501/-Users-leonli66-Desktop-Research-RL-Chess-RL/2dd38b26-cd2d-45b2-b9b8-ebd33a360e65/scratchpad/dashboard.html"
REMOTE_HTML = "/root/dashboard.html"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]==0.115.0")
    .add_local_file(LOCAL_HTML, remote_path=REMOTE_HTML, copy=True)
)

app = modal.App("math-1b-dashboard", image=image)


@app.function(min_containers=1, timeout=60)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    api = FastAPI()

    HTML_HEAD = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<style>* { box-sizing: border-box; } html, body { margin: 0; padding: 0; background: #0e0f13; }</style>"
        "</head><body>"
    )
    HTML_TAIL = "</body></html>"

    with open(REMOTE_HTML) as f:
        body = f.read()
    full = HTML_HEAD + body + HTML_TAIL

    @api.get("/", response_class=HTMLResponse)
    def index():
        return full

    return api
