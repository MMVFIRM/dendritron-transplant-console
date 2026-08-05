#!/usr/bin/env python3
"""Run the strongest packaged diagnostic without starting the web server."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_server.engine import DemoEngine  # noqa: E402


def main() -> None:
    engine = DemoEngine(ROOT, threads=1)
    status = engine.status()
    analysis = engine.analyze(
        {"example_id": "true_if_the", "mode": "correct", "max_new_tokens": 1}
    )
    comparison = engine.compare({"example_id": "true_if_the"})
    step = analysis["steps"][0]
    summary = {
        "donor_loaded": status["donor_loaded"],
        "recipient_parameters": status["recipient"]["stored_parameters"],
        "conditional_active_fraction": status["recipient"]["conditional_active_fraction"],
        "prompt": analysis["input"]["decoded"],
        "generated": analysis["generated_text"],
        "correct_target": step["target"],
        "phrase_hit": step["phrase_hits"][-1] if step["phrase_hits"] else None,
        "recurrent_visits": len(step["routing"]),
        "controls": [
            {
                "mode": item["mode"],
                "target_nll": item["target"]["nll"],
                "target_rank": item["target"]["rank"],
                "next_token": item["next_token"],
            }
            for item in comparison["modes"]
        ],
    }
    print(json.dumps(summary, indent=2))
    correct = comparison["modes"][0]
    alternatives = comparison["modes"][1:]
    assert correct["target"]["rank"] == 1
    assert all(correct["target"]["nll"] < item["target"]["nll"] for item in alternatives)
    print("\nSMOKE PASS: correct memory is rank 1 and beats every packaged corruption.")


if __name__ == "__main__":
    main()
