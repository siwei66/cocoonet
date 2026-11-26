"""Public interface for Cocoonet."""

from importlib.metadata import version as _version

from cocoonet.evaluation_task import EvaluationTask

__all__ = ["EvaluationTask", "get_version"]


def get_version() -> str:
    """Return the installed CocooNet version.

    Returns
    -------
    str
        The installed distribution's version.

    Raises
    ------
    importlib.metadata.PackageNotFoundError
        If CocooNet's installation metadata is unavailable.

    Notes
    -----
    Node usage scope: Shared - Provides version inspection to external users.
    """
    # Read distribution metadata to keep pyproject.toml as the version source.
    return _version("cocoonet")
