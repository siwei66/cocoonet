"""Public interface for the CocooNet release scaffold."""

from importlib.metadata import version as _version

__all__ = ["get_version"]


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
    """
    # Read distribution metadata to keep pyproject.toml as the version source.
    return _version("cocoonet")
