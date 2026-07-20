"""P6-01 coverage matrix: map required scenarios to owning tests.

This module does not re-execute every scenario. It asserts that the owning
test modules remain importable so the release gate cannot drop a whole slice
of the matrix without failing CI.
"""

from importlib import import_module


# scenario_key -> pytest module that owns the regression
_OWNERS: dict[str, str] = {
    "mainline_work_delivered": "tests.objective.test_mainline",
    "verifier_reject_fresh_work": "tests.objective.test_mainline",
    "session_slot_second_active_rejected": "tests.objective.e2e.test_mainline_matrix",
    "false_user_rejected": "tests.objective.e2e.test_mainline_matrix",
    "waiting_user_reply_fresh_root_run": "tests.objective.e2e.test_waiting_user_reply",
    "pause_and_create_idempotency": "tests.objective.e2e.test_pause_and_idempotency",
    "crash_claim_redispatch": "tests.objective.e2e.test_recovery_crash",
    "crash_outcome_complete_outbox": "tests.objective.e2e.test_recovery_crash",
    "budget_boundaries": "tests.objective.test_budget",
    "llm_admit_deny": "tests.objective.test_llm_budget",
    "drain_pause": "tests.objective.test_drain",
    "active_time_window": "tests.objective.test_active_time",
    "running_user_input_inject": "tests.objective.test_running_user_input",
    "fault_boundaries": "tests.objective.test_fault_boundaries",
    "plan_latch_verification_e2e": "tests.objective.e2e.test_plan_latch_verification",
    "startup_recovery_classify": "tests.objective.test_recovery",
    "architecture_guards": "tests.objective.test_architecture",
    "objective_metrics": "tests.objective.test_metrics",
    "terminal_release_allows_next_objective": "tests.objective.test_store",
}


def test_p6_01_owner_modules_importable() -> None:
    for scenario, module_name in _OWNERS.items():
        mod = import_module(module_name)
        assert mod is not None, scenario
