"""Capture CPU and RSS of the lightweight client during an upload on the actual device."""
import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from client.pi_client import upload

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server",required=True);parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("results/device/client_resources.json"))
    args=parser.parse_args();process=psutil.Process();values=[];done=threading.Event()
    process.cpu_percent(None)
    def sample():
        while not done.wait(.1):
            values.append((process.cpu_percent(None),process.memory_info().rss/1024**2))
    monitor=threading.Thread(target=sample);monitor.start();start=time.perf_counter()
    try:
        result=upload(args.server,args.image)
    finally:
        done.set();monitor.join()
    metrics={"host":platform.platform(),"architecture":platform.machine(),"duration_seconds":time.perf_counter()-start,"peak_rss_mb":max([process.memory_info().rss/1024**2]+[v[1] for v in values]),"cpu_percent_samples":[v[0] for v in values],"response":result,"note":"Client process measurement only. Values describe this actual device; server inference is remote."}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(metrics,indent=2),encoding="utf-8");print(json.dumps(metrics,indent=2))
