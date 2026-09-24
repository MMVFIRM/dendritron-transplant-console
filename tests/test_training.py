from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from dendritron_transplant.config import DendritronRecipientConfig
from dendritron_transplant.model import DendritronRecipientLM
from dendritron_transplant.public_fixture import DefinitionRecord, scan_python_stdlib
from dendritron_transplant.training import extended_training_records


def _small_config(**overrides) -> DendritronRecipientConfig:
    values = dict(
        vocab_size=128, model_width=16, donor_width=8, max_sequence_length=16,
        expert_count=4, branches_per_expert=3, branch_hidden_width=4,
        vocabulary_clusters=8, use_hash_memory=False,
    )
    values.update(overrides)
    return DendritronRecipientConfig(**values)


def _inputs() -> torch.Tensor:
    return torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(0))


def test_new_fields_default_to_the_packaged_behaviour() -> None:
    config = _small_config()
    assert (config.residual_mode, config.residual_dropout, config.training_loss) == (
        "postnorm", 0.0, "sparse")
    restored = DendritronRecipientConfig.from_dict(
        _small_config(residual_mode="prenorm", residual_dropout=0.2, training_loss="full").to_dict())
    assert (restored.residual_mode, restored.residual_dropout, restored.training_loss) == (
        "prenorm", 0.2, "full")
    with pytest.raises(ValueError):
        _small_config(residual_mode="sideways")
    with pytest.raises(ValueError):
        _small_config(residual_dropout=1.0)


@pytest.mark.parametrize("mode", ["postnorm", "prenorm"])
def test_both_residual_modes_record_every_recurrent_visit(mode: str) -> None:
    torch.manual_seed(0)
    model = DendritronRecipientLM(_small_config(residual_mode=mode)).eval()
    output = model(_inputs(), return_stats=True)
    stats = output.core_stats
    assert len(stats.visits) == 2 * model.config.loop_rounds
    for visit in stats.visits:
        assert visit.moe.selected_expert_indices.shape == (2, 10, model.config.expert_top_k)
        assert torch.isfinite(visit.relative_change).all()
    assert torch.isfinite(output.hidden).all()


def test_residual_dropout_is_inactive_at_inference() -> None:
    torch.manual_seed(0)
    model = DendritronRecipientLM(_small_config(residual_mode="prenorm", residual_dropout=0.5))
    inputs = _inputs()
    model.eval()
    assert torch.equal(model(inputs).hidden, model(inputs).hidden)
    model.train()
    assert not torch.equal(model(inputs).hidden, model(inputs).hidden)


def test_full_training_loss_is_full_vocabulary_cross_entropy() -> None:
    torch.manual_seed(0)
    model = DendritronRecipientLM(_small_config(training_loss="full"))
    inputs = _inputs()
    targets = torch.roll(inputs, -1, dims=1)
    output = model(inputs, target_ids=targets, include_target_cluster=True)
    loss = model.loss(output, targets)
    dense = model.vocabulary_head.dense_scores(output.hidden, model.token_embeddings.weight)
    expected = (F.cross_entropy(dense.reshape(-1, 128), targets.reshape(-1))
                + model.config.cluster_loss_weight * loss.vocabulary.cluster_loss)
    assert torch.allclose(loss.total, expected)
    sparse_model_loss = DendritronRecipientLM(_small_config()).loss(output, targets)
    assert torch.allclose(sparse_model_loss.total, loss.vocabulary.total)


def _write_module(root: Path, name: str, functions: dict[str, str]) -> None:
    body = [f'"""Module {name} docstring that is long enough to keep."""']
    for function, doc in functions.items():
        body.append(f'def {function}():\n    """{doc}"""\n')
    (root / f"{name}.py").write_text("\n".join(body), encoding="utf-8")


def test_extended_corpus_never_contains_validation_docstrings(tmp_path: Path) -> None:
    _write_module(tmp_path, "alpha", {
        "held_out": "Validation docstring that must never be used for training.",
        "kept": "Training docstring that may be used freely by the recipient.",
    })
    _write_module(tmp_path, "beta", {
        "copy": "Validation docstring that must never be used for training.",
        "extra": "Additional docstring that only the extended corpus contains.",
    })
    scanned = {record.qualified_name: record for record in scan_python_stdlib(tmp_path, limit=None)}
    validation = (scanned["alpha.held_out"],
                  DefinitionRecord("py_missing", "gone", "A validation record absent here.", "", ""))
    train = (scanned["alpha.kept"],)

    for full_text in (False, True):
        records, report = extended_training_records(
            train, validation, full_text=full_text, stdlib_root=tmp_path)
        texts = {record.definition for record in records}
        assert "Validation docstring that must never be used for training." not in texts
        assert "Additional docstring that only the extended corpus contains." in texts
        assert report.unmatched_validation_records == 1
        assert report.excluded_as_validation >= 2  # the original and its verbatim copy
        order = [hashlib.sha256(r.record_id.encode()).digest() for r in records]
        assert order == sorted(order)
