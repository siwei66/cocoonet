"""Tests for the optional public output-record namespace."""

import cocoonet.types as cocoonet_types
from cocoonet._contracts.execution import ModelReference
from cocoonet.configuration import WorkerConfiguration


def test_types_exports_existing_output_record_classes() -> None:
    """Expose task-owned output records without duplicating their classes."""
    assert cocoonet_types.ModelReference is ModelReference
    assert cocoonet_types.WorkerConfiguration is WorkerConfiguration
    assert cocoonet_types.__all__ == [
        "ModelReference",
        "WorkerConfiguration",
    ]
