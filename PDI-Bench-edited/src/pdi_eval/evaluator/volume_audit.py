"""Compatibility import; implementation lives in pdi_eval.v1.rigidity."""

from importlib import import_module

_module = import_module("pdi_eval.v1.rigidity")
__all__ = [name for name in vars(_module) if not name.startswith("_")]

def __getattr__(name):
    return getattr(_module, name)

def __dir__():
    return sorted(set(globals()) | set(dir(_module)))
