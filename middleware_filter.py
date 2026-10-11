"""Backwards-compatible entry point. Prefer ``from uf_lite import uf_middleware``."""

from uf_lite import UFMiddleware, genesis_boot, uf_middleware

__all__ = ["UFMiddleware", "genesis_boot", "uf_middleware"]
