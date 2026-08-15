"""Splice f_panels.html into results_summary.html at the F_PANELS marker."""
import re
from pathlib import Path

S = Path(__file__).parent
PAGE = Path("/Users/leonli66/Desktop/Research/RL/Chess RL/dashboard/results_summary.html")

panels = (S / "f_panels.html").read_text()
block = f"<!--F_PANELS_START-->\n{panels}<!--F_PANELS_END-->"
html = PAGE.read_text()
if "<!--F_PANELS_START-->" in html:
    html = re.sub(r"<!--F_PANELS_START-->.*?<!--F_PANELS_END-->", lambda _: block,
                  html, flags=re.S)
elif "<!--F_PANELS-->" in html:
    html = html.replace("<!--F_PANELS-->", block)
else:
    raise SystemExit("no panel marker found")
PAGE.write_text(html)
print("spliced", len(panels), "bytes of panels")
