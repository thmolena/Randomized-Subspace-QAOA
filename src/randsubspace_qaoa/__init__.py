"""DEPRECATED location.

The canonical package moved to ``code/src/rsqaoa/`` and is imported as ``rsqaoa``.
Install it with::

    cd code && pip install .

This shim only exists because the hosting sync blocks file deletion; it is not
packaged or shipped.
"""
raise ImportError(
    "randsubspace_qaoa has moved to the 'rsqaoa' package under code/. "
    "Run: cd code && pip install ."
)
