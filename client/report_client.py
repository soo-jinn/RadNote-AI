"""Portable report-upload CLI for the Ubuntu VM demonstration."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from client.pi_client import main, upload

if __name__=="__main__":
    main()
