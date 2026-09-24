"""Train a Dendritron recipient on the CPython-docstring benchmark.

Examples:

    # Recommended compact recipient (706K parameters)
    python scripts/train_recipient.py --out build/recipient.pt

    # Larger, slightly better recipient (885K parameters)
    python scripts/train_recipient.py --preset quality --out build/recipient_quality.pt

    # The original packaged recipe for comparison
    python scripts/train_recipient.py --preset packaged --out build/packaged.pt

Configuration starts from the v1.0.0 recipient config; the preset and any
``--set key=value`` overrides are applied on top.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dendritron_transplant.config import DendritronRecipientConfig  # noqa: E402
from dendritron_transplant.model import DendritronRecipientLM  # noqa: E402
from dendritron_transplant.training import (  # noqa: E402
    V1_RECIPIENT_CONFIG,
    TrainingSettings,
    build_data,
    evaluate,
    train,
)

_RECIPE = {"residual_mode": "prenorm", "residual_dropout": 0.2, "training_loss": "full"}
PRESETS = {
    # Findings from experiments/ (see docs/EXPERIMENTS.md). Validation NLL on
    # the fixed 1,000-record validation set is given for each preset.
    "compact": {  # 706K parameters, 3.588 NLL (3 seeds)
        "corpus": "stdlib_full",
        "config": {**_RECIPE, "model_width": 64},
        "settings": {"learning_rate": 3e-2, "weight_decay": 0.1, "steps": 6000, "batch_size": 64},
    },
    "quality": {  # 885K parameters, 3.565 NLL (3 seeds)
        "corpus": "stdlib_full",
        "config": {**_RECIPE, "model_width": 80},
        "settings": {"learning_rate": 3e-2, "weight_decay": 0.1, "steps": 6000, "batch_size": 64},
    },
    "benchmark-corpus": {  # packaged size and data only: 533K parameters, 3.872 NLL (2 seeds)
        "corpus": "benchmark",
        "config": dict(_RECIPE),
        "settings": {"learning_rate": 5e-2, "weight_decay": 0.1, "steps": 3000},
    },
    "packaged": {  # the original recipe: 7.04 NLL at 1,500 steps
        "corpus": "benchmark",
        "config": {},
        "settings": {"learning_rate": 3e-3, "weight_decay": 0.01, "steps": 1500},
    },
}


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="compact")
    parser.add_argument("--corpus", choices=["benchmark", "stdlib", "stdlib_full"],
                        help="training corpus (default: the preset's)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config field, e.g. --set expert_kind=mlp")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "build" / "cache")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    assets = ROOT / "assets"
    preset = PRESETS[args.preset]
    base = copy.deepcopy(V1_RECIPIENT_CONFIG)
    base.update(preset["config"])
    for item in args.set:
        key, _, value = item.partition("=")
        base[key] = _parse_value(value)
    config = DendritronRecipientConfig.from_dict(base)
    settings = TrainingSettings(**{
        **preset["settings"],
        **{k: v for k, v in (("steps", args.steps), ("learning_rate", args.lr),
                              ("weight_decay", args.weight_decay),
                              ("batch_size", args.batch_size)) if v is not None},
        "seed": args.seed,
    })

    corpus = args.corpus or preset["corpus"]
    data = build_data(assets, config, corpus=corpus, cache_dir=args.cache_dir)
    print(f"corpus {corpus}: {data['train_tokens']} training tokens, "
          f"{len(data['train']['x'])} windows; extension={data['extension']}")
    torch.manual_seed(settings.seed)
    model = DendritronRecipientLM(config)
    run = train(model, data, settings, log_every=max(1, settings.steps // 10))
    validation = evaluate(model, data["val"])
    validation["zero_phrase_value_nll"] = evaluate(model, data["val"], zero_phrase_values=True)["nll"]
    # Keys read by the demo engine's status panel.
    validation["dense_nll"] = validation["nll"]
    validation["sparse_next_token_accuracy"] = validation["sparse_accuracy"]
    training_subset = evaluate(model, data["train"], limit=len(data["val"]["x"]))
    metadata = {
        "recipe": args.preset,
        "corpus": corpus,
        "train_tokens": data["train_tokens"],
        "extension": data["extension"],
        "settings": settings.__dict__,
        "training_seconds": run["seconds"],
        "history": run["history"],
        "evaluation": validation,
        "train_subset_nll": training_subset["nll"],
    }
    model.save_checkpoint(args.out, metadata=metadata)
    print(json.dumps({"validation": validation, "train_subset_nll": training_subset["nll"],
                      "parameters": model.parameter_report()}, indent=2))


if __name__ == "__main__":
    main()
