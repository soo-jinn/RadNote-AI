"""Measure sustained warm API load for a timed window, not a short extrapolated burst."""
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import httpx

def percentile(values,q):
    values=sorted(values);k=(len(values)-1)*q;i=int(k)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(k-i) if values else None

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--server",default="http://127.0.0.1:8000");parser.add_argument("--seconds",type=float,default=60);parser.add_argument("--output",type=Path,default=Path("results/local/load_sustained.json"));args=parser.parse_args()
    if args.seconds<=0:raise ValueError("Duration must be positive")
    image=Path("samples/ocr/images/case_001.png").read_bytes();results={}
    for kind,concurrency in [("text",4),("image",2)]:
        started=time.perf_counter();deadline=started+args.seconds
        def worker():
            rows=[]
            with httpx.Client(timeout=180) as client:
                while time.perf_counter()<deadline:
                    begin=time.perf_counter()
                    try:
                        response=client.post(args.server+"/predict",json={"findings":"The lungs are clear.","impression":"No acute disease."}) if kind=="text" else client.post(args.server+"/predict-image",files={"file":("case_001.png",image,"image/png")})
                        status=response.status_code
                    except httpx.HTTPError:status=0
                    rows.append((status,(time.perf_counter()-begin)*1000))
            return rows
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            rows=[row for batch in executor.map(lambda _:worker(),range(concurrency)) for row in batch]
        elapsed=time.perf_counter()-started;successes=sum(status==200 for status,_ in rows)
        results[kind]={"target_window_seconds":args.seconds,"actual_elapsed_seconds":elapsed,"concurrency":concurrency,"requests":len(rows),"successes":successes,"failures":len(rows)-successes,"observed_successes_per_minute":successes*60/elapsed,"p50_round_trip_ms":percentile([ms for _,ms in rows],.5),"p95_round_trip_ms":percentile([ms for _,ms in rows],.95)}
        print(kind+": "+json.dumps(results[kind]),flush=True)
    output={"server":args.server,"load":results,"notes":["Warm single server process on development laptop; repeat on Ubuntu.","Persistent HTTP clients; repeated short normal text/page mix. Does not represent all report lengths or users.","In-flight requests finish after the timed window; throughput uses total actual elapsed time."]};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2),encoding="utf-8")
    if any(r['failures'] for r in results.values()):raise SystemExit("Sustained load failures recorded")

if __name__=="__main__":main()
