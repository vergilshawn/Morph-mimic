"""Select test suites without importing Isaac or starting a simulator.

Plain pytest runs everything. The public check command explicitly selects basic.
Keep optional cases here rather than scattering markers throughout test bodies.
"""
from pathlib import Path

import pytest


SUITES = ("basic", "reference", "planner", "auxiliary", "diagnostics", "physics", "all")
MODULE_SUITES = {
    "test_foundation.py": {"basic", "planner", "physics"},
    "test_reference.py": {"basic", "reference", "physics"},
    "test_contacts.py": {"basic", "physics"},
    "test_control.py": {"basic", "physics"},
    "test_training.py": {"basic", "planner", "auxiliary"},
    "test_evaluation.py": {"basic", "diagnostics", "physics"},
    "test_workflows.py": {"basic", "diagnostics"},
}
REFERENCE_BASICS = {
    "TestGeometricSchedule", "TestMotionStateContract", "TestReferenceScene",
    "TestSceneIntegration", "TestFilteredSceneContacts", "SettleReferenceTests",
}
PLANNER_CASES = {
    "TestHindsight", "TestPlanner", "TestBuffer", "TestPlannerAttention",
    "test_stage3_always_uses_predicted_contacts",
    "test_split_planner_gradients_equal_full_batch_with_uneven_contacts",
    "test_planner_resume_rejects_mismatched_buffer_before_changing_weights",
    "test_planner_nonfinite_gradient_is_rejected_before_optimizer_step",
}
AUXILIARY_CASES = {"TestTeacherSupervisor", "TestStandingMetrics", "TestStandingSupervisor"}
DIAGNOSTIC_CASES = {
    "TestFrameAudit",
    "test_continuation_suppresses_only_root_failure_and_preserves_original_decision",
    "test_continued_reference_end_never_becomes_accepted_completion",
    "test_trajectory_records_first_strict_failure_and_diagnostic_metadata",
    "test_strict_trajectory_marks_suffix_as_diagnostic_without_relaxing_physics",
    "test_coordinator_module_does_not_import_cuda_or_array_libraries",
    "test_each_segment_uses_fresh_process_and_aggregates_only_published_artifacts",
    "test_child_failure_preserves_explicit_incomplete_summary",
}
PHYSICS_CLASSES = {
    "TestExplicitMjcf", "StableHoldIntegrationTests", "TestBoxSourceFk",
    "TestSignedCollisionDistances", "TestReferenceProjection", "TestSceneReferenceDiagnostic",
    "TestSceneRetarget", "TestEffectiveSelfCollision", "TestOmniConversion",
}
PHYSICS_METHODS = {
    "test_source_fk_joint_permutation_keeps_body_and_joint_pivots",
    "test_generic_projection_refuses_either_scene_binding_marker_without_writing",
    "test_append_retains_every_source_array_prefix_and_rebinds_scene",
    "test_static_fk_is_complete_and_continuous_support_has_no_future_onset",
}


def pytest_addoption(parser):
    parser.addoption("--suite", action="append", choices=SUITES, default=None,
                     help="test group; repeat to combine; plain pytest defaults to all")


def pytest_ignore_collect(collection_path, config):
    """Avoid importing training dependencies for an unrelated optional suite."""
    available = MODULE_SUITES.get(collection_path.name)
    requested = set(config.getoption("--suite") or ["all"])
    return available is not None and "all" not in requested and not requested & available


def _suite_tags(item):
    module = Path(item.path).name
    case = item.cls.__name__ if item.cls else item.originalname or item.name
    tags = {"reference"} if module == "test_reference.py" else {"basic"}
    if case in REFERENCE_BASICS:
        tags.add("basic")
    for name, cases in (("planner", PLANNER_CASES), ("auxiliary", AUXILIARY_CASES),
                        ("diagnostics", DIAGNOSTIC_CASES)):
        if case in cases:
            tags.discard("basic")
            tags.add(name)
    # Classify independently of whether MuJoCo happens to be installed today.
    if case in PHYSICS_CLASSES or (item.originalname or item.name) in PHYSICS_METHODS:
        tags.discard("basic")
        tags.add("physics")
    return tags


def pytest_collection_modifyitems(config, items):
    requested = set(config.getoption("--suite") or ["all"])
    selected, deselected = [], []
    for item in items:
        target = selected if "all" in requested or requested & _suite_tags(item) else deselected
        target.append(item)
    order = {name: index for index, name in enumerate(MODULE_SUITES)}
    selected.sort(key=lambda item: order.get(Path(item.path).name, len(order)))
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
