"""
Deprecated import name for DerivFlow Finance; use ``derivflow_finance``.

DerivFlow Finance 0.4 renamed its import package from ``derivflow`` to
``derivflow_finance``, matching the distribution name ``derivflow-finance``.
Throughout the 0.4 series the old name keeps working: ``import derivflow``
returns the ``derivflow_finance`` package itself, and
``from derivflow.<module> import <name>`` returns the same objects as
``from derivflow_finance.<module> import <name>`` (each module is loaded once,
under its new name). The first import of the old name emits a FutureWarning.

This alias will be removed in 0.5.0. To migrate, replace ``derivflow`` with
``derivflow_finance`` in import statements, for example
``from derivflow_finance import price_european_option``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings
from types import ModuleType
from typing import Any

_OLD = "derivflow"
_NEW = "derivflow_finance"

warnings.warn(
    "'derivflow' is a deprecated import name and will be removed in DerivFlow Finance 0.5.0; "
    "import 'derivflow_finance' instead (for example 'from derivflow_finance import ...').",
    FutureWarning,
    stacklevel=2,
)

_package = importlib.import_module(_NEW)


def _new_name(old_name: str) -> str:
    return _NEW + old_name[len(_OLD) :]


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``derivflow.<sub>`` to the module object ``derivflow_finance.<sub>``.

    The import system sets ``__spec__`` on the module a loader returns, so the
    module's own spec is saved in ``create_module`` and put back in
    ``exec_module``; the module keeps its ``derivflow_finance`` identity.
    """

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(_OLD + "."):
            return None
        if importlib.util.find_spec(_new_name(fullname)) is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, self)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(_new_name(spec.name))
        spec.loader_state = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        state = module.__spec__.loader_state if module.__spec__ is not None else None
        if isinstance(state, importlib.machinery.ModuleSpec):
            module.__spec__ = state


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

# Modules the package has already loaded are registered under the old names
# directly; the finder covers the ones loaded later (the optional extras).
for _name, _module in list(sys.modules.items()):
    if _name.startswith(_NEW + "."):
        sys.modules.setdefault(_OLD + _name[len(_NEW) :], _module)

# ``import derivflow`` evaluates to the derivflow_finance package.
sys.modules[_OLD] = _package
