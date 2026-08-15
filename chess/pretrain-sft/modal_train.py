# Moved to modal_scripts/train.py — this re-exports for backwards compat.
# Usage: modal run modal_scripts/train.py
#        modal run modal_scripts/sweep_C9e18_beta0.04.py
from modal_scripts.train import app, train, download_data, download_test_data, main
