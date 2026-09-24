"""Print a table of results grouped by run name (mean ± std over seeds)."""
import json, statistics, sys
from collections import defaultdict

rows = defaultdict(list)
for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); rows[r["name"]].append(r)
base = statistics.mean(r["nll"] for r in rows["baseline"]) if "baseline" in rows else None
print(f"{'name':24s} {'n':>2s} {'val NLL':>15s} {'dNLL':>7s} {'ppl':>7s} {'acc':>6s} {'params':>8s} {'active':>6s} {'sec':>5s}")
for name, rs in sorted(rows.items(), key=lambda kv: statistics.mean(r["nll"] for r in kv[1])):
    nll = [r["nll"] for r in rs]
    sd = statistics.stdev(nll) if len(nll) > 1 else 0.0
    m = statistics.mean(nll)
    extra = f" zero-mem {statistics.mean(r['zero_memory_nll'] for r in rs):.3f}" if "zero_memory_nll" in rs[0] else ""
    print(f"{name:24s} {len(rs):2d} {m:8.4f}±{sd:.3f} {(m-base) if base else 0:+7.3f} {statistics.mean(r['ppl'] for r in rs):7.1f} "
          f"{statistics.mean(r['dense_acc'] for r in rs):6.3f} {rs[0]['params']:8d} {rs[0]['active_fraction']:6.3f} {statistics.mean(r['train_seconds'] for r in rs):5.0f}{extra}")
