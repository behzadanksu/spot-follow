"""Configuration loading: YAML file merged over built-in defaults."""
import copy
import os

import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'config', 'default.yaml')


class Config(dict):
    """dict with attribute access, recursively (cfg.follow.standoff_m)."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return value

    def __setattr__(self, key, value):
        self[key] = value


def _wrap(obj):
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides=None):
    """Load default.yaml, then `path` on top, then a dict of dotted overrides."""
    with open(DEFAULT_PATH, 'r') as f:
        data = yaml.safe_load(f)
    if path and os.path.abspath(path) != DEFAULT_PATH:
        with open(path, 'r') as f:
            data = _merge(data, yaml.safe_load(f) or {})
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = data
        keys = dotted.split('.')
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return _wrap(data)
