from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys


_TARGET = "sglang.srt.managers.detokenizer_manager"


def _flatten_token_ids(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_token_ids(item))
        return out
    return [int(value)]


def _patch_detokenizer_module(module) -> None:
    cls = getattr(module, "DetokenizerManager", None)
    if cls is None or getattr(cls, "_chess_rl_flatten_decode_ids", False):
        return

    def _decode_one(tokenizer, ids, skip_special_tokens, spaces_between_special_tokens):
        kwargs = {
            "skip_special_tokens": skip_special_tokens,
            "spaces_between_special_tokens": spaces_between_special_tokens,
        }
        try:
            return tokenizer.decode(ids, **kwargs)
        except TypeError as exc:
            if "spaces_between_special_tokens" not in str(exc):
                raise
            kwargs.pop("spaces_between_special_tokens", None)
            return tokenizer.decode(ids, **kwargs)

    def _grouped_batch_decode_flattened(self, ids_list, skip_list, space_list):
        return [
            _decode_one(self.tokenizer, _flatten_token_ids(ids), skip, space)
            for ids, skip, space in zip(ids_list, skip_list, space_list, strict=False)
        ]

    cls._grouped_batch_decode = _grouped_batch_decode_flattened
    cls._chess_rl_flatten_decode_ids = True


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        create_module = getattr(self._wrapped, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module) -> None:
        self._wrapped.exec_module(module)
        _patch_detokenizer_module(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or isinstance(spec.loader, _PatchLoader):
            return spec
        spec.loader = _PatchLoader(spec.loader)
        return spec


if _TARGET in sys.modules:
    _patch_detokenizer_module(sys.modules[_TARGET])
else:
    sys.meta_path.insert(0, _PatchFinder())
