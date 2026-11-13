"""Focused atomic-publication support for Cocoonet storage."""


class StorageError(RuntimeError):
    """Report an operational storage failure without defining retry policy.

    Notes
    -----
    Storage operations provide the operation and exact target path in the
    exception message and chain the underlying exception. This exception does
    not decide whether an operation is retried.

    Node usage scope: Shared - Used by both root and worker storage.
    """
