"""Measure synthetic OCR accuracy, client latency, and HTTP throughput. No clinical validation."""
import argparse
import csv
import json
import platform
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
import psutil

def distance(a, b):
    previous = list(range(len(b)+1))
    for i, value in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(value!=other)))
        previous = current
    return previous[-1]

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    k = (len(ordered)-1)*q
    i = int(k)
    return ordered[i]+(ordered[min(i+1,len(ordered)-1)]-ordered[i])*(k-i)

def benchmark(server, manifest_path, output, server_pid=None):
    manifest = json.loads(manifest_path.read_text()); output.mkdir(parents=True, exist_ok=True)
    rows = []; samples = []
    process = psutil.Process(server_pid) if server_pid else None
    with httpx.Client(timeout=180) as client:
        health = client.get(server+"/health").json()
        for i, case in enumerate(manifest):
            image = manifest_path.parent/case["image"]
            started = time.perf_counter()
            with image.open("rb") as stream:
                response = client.post(server+"/predict-image", files={"file":(image.name, stream, "image/png")})
            latency = (time.perf_counter()-started)*1000
            row = {"id":case["id"],"variant":case["variant"],"status":response.status_code,"round_trip_ms":round(latency,2),"scenario_label":case["scenario_label"]}
            if response.status_code == 200:
                result = response.json(); reference = normalize(case["reference_text"]); observed = normalize(result["extracted_text"])
                row.update({"predicted_label":result["urgency"],"cer_errors":distance(reference,observed),"cer_length":len(reference),"wer_errors":distance(reference.split(),observed.split()),"wer_length":len(reference.split()),"server_ms":result["latency_ms"]})
                row["cer"] = row["cer_errors"]/row["cer_length"]
                row["wer"] = row["wer_errors"]/row["wer_length"]
                samples.append({"id":case["id"],"response":result})
            else:
                row["error"] = response.text[:500]
            if process:
                row["server_rss_mb"] = process.memory_info().rss/1024**2
            rows.append(row)
            print(f"{i+1}/{len(manifest)} {case['id']} HTTP {response.status_code} {latency:.0f} ms", flush=True)
    def load_request(kind):
        started = time.perf_counter()
        with httpx.Client(timeout=180) as client:
            if kind == "text":
                response = client.post(server+"/predict",json={"findings":"The lungs are clear.","impression":"No acute disease."})
            else:
                image = manifest_path.parent/manifest[0]["image"]
                response = client.post(server+"/predict-image", files={"file":(image.name,image.read_bytes(),"image/png")})
        return response.status_code, (time.perf_counter()-started)*1000
    throughput = {}
    for kind, count, concurrency in [("text",40,4),("image",8,2)]:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            data = list(executor.map(load_request, [kind]*count))
        elapsed = time.perf_counter()-start; successful = sum(code==200 for code,_ in data)
        throughput[kind] = {"requests":count,"concurrency":concurrency,"successes":successful,"elapsed_seconds":elapsed,"successful_requests_per_minute":successful*60/elapsed,"p95_round_trip_ms":percentile([ms for _,ms in data],.95)}
    successes = [r for r in rows if r["status"] == 200]
    summary = {"measurement_host":platform.platform(),"client_machine":platform.machine(),"logical_cpus":psutil.cpu_count(),"server_health":health,"dataset":"60 original synthetic pages (12 templates, 5 visual variants each); no Open-I report text", "sample_count":len(rows),"successes":len(successes),"failures":len(rows)-len(successes),"ocr_normalization":"lowercase and whitespace collapse; punctuation retained", "cer":sum(r["cer_errors"] for r in successes)/sum(r["cer_length"] for r in successes) if successes else None,"wer":sum(r["wer_errors"] for r in successes)/sum(r["wer_length"] for r in successes) if successes else None,"p50_round_trip_ms":percentile([r["round_trip_ms"] for r in successes],.5),"p95_round_trip_ms":percentile([r["round_trip_ms"] for r in successes],.95),"max_observed_server_rss_mb":max((r.get("server_rss_mb",0) for r in rows),default=0) if process else None,"synthetic_scenario_label_agreement":sum(r["scenario_label"]==r["predicted_label"] for r in successes)/len(successes) if successes else None,"throughput":throughput,"notes":["Windows/laptop measurements must be repeated on Ubuntu; Raspberry Pi CPU/RAM not measured.","Synthetic scenario labels and repeated templates are engineering test cases, not clinical ground truth or independent classifier validation.","OCR requests are serialized per server process to bound CPU/memory use. Load results are specific to this host, sample mix, concurrency, and process count."]}
    (output/"benchmark_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (output/"synthetic_responses.json").write_text(json.dumps(samples,indent=2),encoding="utf-8")
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (output/"benchmark_rows.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2))
    if len(successes) != len(rows):
        raise SystemExit("Some synthetic image requests failed; see benchmark_rows.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server",default="http://127.0.0.1:8000")
    parser.add_argument("--manifest",type=Path,default=Path("samples/ocr/manifest.json"))
    parser.add_argument("--output",type=Path,default=Path("results/local"))
    parser.add_argument("--server-pid",type=int)
    args=parser.parse_args();benchmark(args.server.rstrip("/"),args.manifest,args.output,args.server_pid)
