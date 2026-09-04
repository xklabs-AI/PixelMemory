"""Allow `python -m backend.ingest /path/to/photos`."""
# This file exists so `python -m backend` works as an entry point.
# The actual CLI is in backend/ingest.py — run:
#   python -m backend.ingest /path/to/photos

from backend.ingest import main
main()
