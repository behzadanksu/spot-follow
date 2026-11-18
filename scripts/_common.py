"""Shared helpers for the scripts: put the repo on sys.path and set up logging."""
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)  # so relative paths in the config (models/, data/) resolve from the repo root


def setup_logging(verbose=False):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
                        datefmt='%H:%M:%S')
    for noisy in ('bosdyn', 'urllib3', 'PIL', 'matplotlib'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
