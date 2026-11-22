"""Root-side prospective evaluation configuration records."""

from dataclasses import dataclass
from enum import StrEnum

from cocoonet._contracts.execution import ModelFamily, TaskKind
from cocoonet._contracts.identity import ScalarLabel


class WorkerKind(StrEnum):
    """Identify the configured execution topology of a worker.

    Attributes
    ----------
    ROOT_LOCAL : str
        Worker explicitly configured on the root machine.
    ONLINE : str
        Worker reached through a later online connection boundary.

    Notes
    -----
    Public/private online ownership is intentionally not represented here.
    Node usage scope: Root - Defines root-side worker topology configuration.
    """

    ROOT_LOCAL = "root-local"
    ONLINE = "online"


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelRegistration:
    """Describe one opaque model input proposed for registration.

    Parameters
    ----------
    model_input : object
        In-memory model input retained without inspection or copying.
    model_label : ScalarLabel
        User-provided label that later registry operations require to be
        unique within an evaluation task.
    family : ModelFamily
        Explicitly declared model-construction family.
    task_kind : TaskKind
        Explicitly declared regression or classification task.

    Notes
    -----
    Construction records the supplied values without validating or invoking
    the opaque model input. Validation is a separate configuration boundary.
    Node usage scope: Root - Defines root-side prospective model input.
    """

    model_input: object
    model_label: ScalarLabel
    family: ModelFamily
    task_kind: TaskKind
