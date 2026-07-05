"""Registry of available method adapters.

`run_benchmark.py` never needs editing when a new method is added -- it just
looks the name up here, importing the corresponding module lazily so unused
adapters (and their heavy, pinned dependencies) never get imported.
"""
import importlib

_REGISTRY = {}


def register(name):
    def _decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return _decorator


def get_adapter_cls(name):
    if name not in _REGISTRY:
        importlib.import_module(f"soc_bench.adapters.{name}")
    if name not in _REGISTRY:
        raise KeyError(f"No adapter registered under '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]
