"""Root-side prospective evaluation configuration records."""

from enum import StrEnum


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
