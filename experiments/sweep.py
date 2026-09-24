"""Run a list of experiment specs in parallel subprocesses."""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

def launch(spec, out, steps, threads):
    name, patches, overrides, extra = spec
    cmd = [PY, "-m", "experiments.harness", "--name", name, "--steps", str(steps),
           "--threads", str(threads), "--patches", ",".join(patches),
           "--overrides", json.dumps(overrides), "--out", out, *extra]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        print(f"FAILED {name} rc={r.returncode} stderr={r.stderr[-1500:]!r}", flush=True)
    else:
        print(r.stdout.strip(), flush=True)

def sweep(specs, out, *, steps=1500, threads=2, workers=11):
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda s: launch(s, out, steps, threads), specs))
