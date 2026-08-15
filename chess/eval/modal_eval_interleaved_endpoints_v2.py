"""Collision-safe v2 entrypoint for interleaved endpoint evaluation.

This wrapper must be invoked in a fresh Python process (as ``modal run`` and
``modal deploy`` do).  It selects the audited v2 profile before importing the
shared evaluator.  The shared evaluator defaults to v1, so existing v1
commands and the already-running v1 Modal app are unaffected.

Safe preparation commands (do not run until v2 endpoints should be watched):

    modal run modal_eval_interleaved_endpoints_v2.py --mode dry-run
    modal run --detach modal_eval_interleaved_endpoints_v2.py --mode launch
"""

from __future__ import annotations

import os
import sys


_PROFILE_ENV = "CHESS_INTERLEAVE_ENDPOINT_EVAL_PROFILE"
_existing_profile = os.environ.get(_PROFILE_ENV)
if _existing_profile is not None and _existing_profile.strip().lower() != "v2":
    raise RuntimeError(
        f"{_PROFILE_ENV} is already {_existing_profile!r}; refusing to "
        "silently select a different endpoint evaluator profile"
    )
os.environ[_PROFILE_ENV] = "v2"

_loaded_base_names = {
    "Eval.modal_eval_interleaved_endpoints",
    "modal_eval_interleaved_endpoints",
}.intersection(sys.modules)
if _loaded_base_names:
    raise RuntimeError(
        "the shared endpoint evaluator was imported before the v2 profile "
        f"was selected: {sorted(_loaded_base_names)}"
    )

if __package__:
    from .modal_eval_interleaved_endpoints import *  # noqa: F401,F403
else:
    from modal_eval_interleaved_endpoints import *  # type: ignore  # noqa: F401,F403
