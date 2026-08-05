from __future__ import annotations

from demo_server.engine import DemoEngine


def test_status_has_bound_provenance_and_no_live_donor(engine: DemoEngine) -> None:
    status = engine.status()
    assert status["ready"] is True
    assert status["donor_loaded"] is False
    assert status["recipient_loaded"] is True
    assert status["donor"]["model_id"] == "Qwen/Qwen2.5-0.5B"
    assert status["donor"]["revision"] == "060db6499f32faf8b98477b0a26969ef7d8b9987"
    assert status["donor"]["weight_sha256"] == "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
    assert status["donor"]["parameters"] == 494_032_768
    assert status["recipient"]["conditional_active_fraction"] < 0.10
    assert status["memory"]["rows"] == 2_000
    assert status["memory"]["definition_rows"] == 5_000


def test_control_invariants_preserve_addresses(engine: DemoEngine) -> None:
    report = engine.status()["memory"]["control_invariants"]
    assert report["rows"] == 2_000
    assert report["shuffled_fixed_points"] == 0
    assert report["semantic_fixed_points"] == 0
    assert report["shuffled_unique_rows"] == 2_000
    assert report["semantic_unique_rows"] == 2_000
    assert report["same_order_shuffled"] is True
    assert report["same_order_semantic"] is True


def test_live_trace_exposes_exact_memory_and_sparse_routing(engine: DemoEngine) -> None:
    result = engine.analyze(
        {"example_id": "true_if_the", "mode": "correct", "max_new_tokens": 1}
    )
    assert result["engine"] == "live"
    step = result["steps"][0]
    assert step["target"]["token"] == "the"
    assert step["target"]["rank"] == 1
    assert step["phrase_hits"]
    assert step["phrase_hits"][-1]["address_row"] == 34
    assert step["phrase_hits"][-1]["value_row"] == 34
    assert len(step["routing"]) == 4
    for visit in step["routing"]:
        assert len(visit["selected_experts"]) == 2
        assert all(len(expert["branches"]) == 2 for expert in visit["selected_experts"])
        assert visit["active_conditional_fraction"] < 0.10
    assert result["resource_summary"]["donor_loaded"] is False


def test_fixed_model_memory_swap_changes_diagnostic_result(engine: DemoEngine) -> None:
    result = engine.compare({"example_id": "true_if_the"})
    modes = {item["mode"]: item for item in result["modes"]}
    assert set(modes) == {
        "correct",
        "shuffled",
        "semantic_opposite",
        "random_frozen",
        "zero",
        "hash_only",
    }
    correct = modes["correct"]
    assert correct["target"]["rank"] == 1
    for mode in ("shuffled", "semantic_opposite", "random_frozen", "zero", "hash_only"):
        assert correct["target"]["nll"] < modes[mode]["target"]["nll"]
    address_rows = {
        item["phrase_hits"][-1]["address_row"]
        for item in result["modes"]
        if item["phrase_hits"]
    }
    assert address_rows == {34}
    assert modes["shuffled"]["phrase_hits"][-1]["value_row"] != 34
    assert modes["semantic_opposite"]["phrase_hits"][-1]["value_row"] != 34
    assert modes["zero"]["phrase_hits"][-1]["value_row"] is None
    assert modes["hash_only"]["phrase_hits"][-1]["exact_enabled"] is False
