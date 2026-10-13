"""Portable browser proxy, including the Docker client container."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from client.pi_web import main

if __name__=="__main__":
    main()
