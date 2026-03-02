"""
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py

Refactored for strict type safety and standard logging.
"""

import importlib
import logging
import types
from typing import Any, Optional


class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.

    `paddle` and `ffmpeg` are examples of modules that are large and not always
    needed, and this allows them to only be loaded when they are used.
    """

    def __init__(
        self, local_name: str, parent_module_globals: dict[str, Any], name: str, warning: Optional[str] = None
    ):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._warning = warning

        # These members allow doctest to correctly process this module member without
        # triggering self._load(). self._load() mutates parent_module_globals and
        # triggers a dictionary mutated during iteration error from doctest.py.
        self.__module__ = name.rsplit(".", 1)[0]
        self.__wrapped__ = None

        super().__init__(name)

    def _load(self) -> types.ModuleType:
        """Load the module and insert it into the parent's globals."""
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        if self._warning:
            logging.warning(self._warning)
            # Make sure to only warn once.
            self._warning = None

        # Update this object's dict so that if someone keeps a reference to the
        # LazyLoader, lookups are efficient.
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item: str) -> Any:
        module = self._load()
        return getattr(module, item)

    def __repr__(self) -> str:
        # Carefull not to trigger _load, since repr may be called in sensitive places.
        return f"<LazyLoader {self.__name__} as {self._local_name}>"

    def __dir__(self) -> list:
        module = self._load()
        return dir(module)
