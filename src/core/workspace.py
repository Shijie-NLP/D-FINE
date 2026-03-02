"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Optimized for strict typing, readability, and safe dictionary resolution.
"""

import functools
import importlib
import inspect
from collections import defaultdict
from typing import Any, Callable, Optional, Union


# Global registry for schemas and instantiated modules
GLOBAL_CONFIG: dict[str, Any] = defaultdict(dict)


def register(dct: Any = GLOBAL_CONFIG, name: Optional[str] = None, force: bool = False) -> Callable:
    """
    Decorator to register a class or function into a configuration dictionary
    or as an attribute of a specific module/class.

    Args:
        dct (Any): The target dictionary or class for registration.
        name (str, optional): The name to register under. Defaults to the object's __name__.
        force (bool): Whether to forcefully overwrite an existing registration.
    """

    def decorator(foo: Callable) -> Callable:
        register_name = foo.__name__ if name is None else name

        # Validation to prevent accidental overwrites
        if not force:
            if inspect.isclass(dct):
                assert not hasattr(dct, foo.__name__), f"Module {dct.__name__} already has attribute {foo.__name__}"
            else:
                assert foo.__name__ not in dct, f"Target {foo.__name__} has already been registered."

        if inspect.isfunction(foo):

            @functools.wraps(foo)
            def wrap_func(*args: Any, **kwargs: Any) -> Any:
                return foo(*args, **kwargs)

            # Bind function based on the type of target registry
            if isinstance(dct, dict):
                dct[foo.__name__] = wrap_func
            elif inspect.isclass(dct):
                setattr(dct, foo.__name__, wrap_func)
            else:
                raise AttributeError(f"Unsupported registry target type: {type(dct)}")
            return wrap_func

        elif inspect.isclass(foo):
            # Extract and store the initialization signature schema for classes
            dct[register_name] = extract_schema(foo)

        else:
            raise ValueError(f"Registration does not support type {type(foo)}")

        return foo

    return decorator


def extract_schema(module: type) -> dict[str, Any]:
    """
    Parses a class's __init__ signature to build a configuration schema,
    identifying standard arguments, default values, shared variables, and injected dependencies.
    """
    argspec = inspect.getfullargspec(module.__init__)
    arg_names = [arg for arg in argspec.args if arg != "self"]
    num_defaults = len(argspec.defaults) if argspec.defaults is not None else 0
    num_requires = len(arg_names) - num_defaults

    schema: dict[str, Any] = {}
    schema["_name"] = module.__name__
    schema["_pymodule"] = importlib.import_module(module.__module__)

    # Metadata for dependency injection and shared state
    schema["_inject"] = getattr(module, "__inject__", [])
    schema["_share"] = getattr(module, "__share__", [])
    schema["_kwargs"] = {}

    for i, arg_name in enumerate(arg_names):
        if arg_name in schema["_share"]:
            assert i >= num_requires, f"Shared config '{arg_name}' must have a default value."
            value = argspec.defaults[i - num_requires]

        elif i >= num_requires:
            value = argspec.defaults[i - num_requires]

        else:
            value = None

        schema[arg_name] = value
        schema["_kwargs"][arg_name] = value

    return schema


def create(
    type_or_name: Union[type, str],
    global_cfg: dict[str, Any] = GLOBAL_CONFIG,
    **kwargs: Any,
) -> Any:
    """
    Dynamically instantiates a registered object by resolving its schema,
    injecting dependencies recursively, and binding shared configurations.
    """
    assert isinstance(type_or_name, (type, str)), "Target must be a class type or a registered string name."

    name = type_or_name if isinstance(type_or_name, str) else type_or_name.__name__

    if name in global_cfg:
        # If the object is already instantiated (identified by having a __dict__),
        # return the singleton instance.
        if hasattr(global_cfg[name], "__dict__"):
            return global_cfg[name]
    else:
        raise ValueError(f"The module '{name}' is not registered.")

    cfg = global_cfg[name]

    # Handle configuration aliases/redirection via 'type' parameter
    if isinstance(cfg, dict) and "type" in cfg:
        _cfg: dict[str, Any] = global_cfg[cfg["type"]]

        # Warning: This mutates the global registry dictionary in-place.
        # Clean non-private arguments
        _keys = [k for k in _cfg.keys() if not k.startswith("_")]
        for _arg in _keys:
            del _cfg[_arg]

        _cfg.update(_cfg.get("_kwargs", {}))  # Restore default args safely
        _cfg.update(cfg)  # Load config args
        _cfg.update(kwargs)  # Override with provided kwargs
        target_name = _cfg.pop("type")  # Pop extra key `type` (from cfg)

        return create(target_name, global_cfg)

    module = getattr(cfg["_pymodule"], name)
    module_kwargs: dict[str, Any] = {}
    module_kwargs.update(cfg)

    # Resolve shared variables
    for k in cfg.get("_share", []):
        if k in global_cfg:
            module_kwargs[k] = global_cfg[k]
        else:
            module_kwargs[k] = cfg[k]

    # Resolve dependency injections recursively
    for k in cfg.get("_inject", []):
        _k = cfg[k]

        if _k is None:
            continue

        if isinstance(_k, str):
            if _k not in global_cfg:
                raise ValueError(f"Missing inject config for '{_k}'.")

            _cfg_inject = global_cfg[_k]

            if isinstance(_cfg_inject, dict):
                module_kwargs[k] = create(_cfg_inject["_name"], global_cfg)
            else:
                module_kwargs[k] = _cfg_inject

        elif isinstance(_k, dict):
            if "type" not in _k.keys():
                raise ValueError("Missing 'type' key for dictionary-based injection.")

            _type = str(_k["type"])
            if _type not in global_cfg:
                raise ValueError(f"Missing '{_type}' registered target in injection stage.")

            _cfg_nested: dict[str, Any] = global_cfg[_type]

            # Warning: In-place mutation logic preserved.
            # Clean non-private arguments
            _keys = [key for key in _cfg_nested.keys() if not key.startswith("_")]
            for _arg in _keys:
                del _cfg_nested[_arg]

            _cfg_nested.update(_cfg_nested.get("_kwargs", {}))  # Restore default values
            _cfg_nested.update(_k)  # Load config args
            target_inject_name = _cfg_nested.pop("type")  # Pop extra key (`type` from _k)

            module_kwargs[k] = create(target_inject_name, global_cfg)

        else:
            raise ValueError(f"Injection does not support type {_k}")

    # Strip out internal schema metadata before passing to the __init__ function
    module_kwargs = {k: v for k, v in module_kwargs.items() if not k.startswith("_")}

    # Optional: Catch unhandled kwargs to prevent silent errors
    # extra_args = set(module_kwargs.keys()) - set(arg_names)
    # if len(extra_args) > 0:
    #     raise RuntimeError(f'Error: unknown args {extra_args} for {module}')

    return module(**module_kwargs)
