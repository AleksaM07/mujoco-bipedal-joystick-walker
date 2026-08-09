"""MJX backend helpers.

MJX-Warp is the production path in this repo.  The helpers are intentionally
small: validate the installed packages, convert the host MuJoCo model, and pass
the correct static contact capacities when MJX Data is allocated.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from dataclasses import asdict, dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from mujoco import mjx


PhysicsBackend = Literal["mjx_jax", "mjx_warp"]
WarpGraphModeName = Literal["warp", "warp_staged", "warp_staged_ex"]

# REF: PROJECT-WARP-PACKAGE-PINS
# TYPE: ENGINEERING_DEFAULT
OFFICIAL_WARP_INSTALL_COMMAND = (
    'python -m pip install "mujoco==3.10.0" '
    '"mujoco-mjx[warp]==3.10.0"'
)

# REF: MJX-WARP-CAPACITY-DEFAULTS
# TYPE: REFERENCE_CODE_DERIVED
DEFAULT_WARP_CONTACTS_PER_WORLD = 8
DEFAULT_WARP_NJMAX_PER_WORLD = 64


@dataclass(frozen=True)
class WarpCapacityPlan:
    """Resolved MJX-Warp capacities and where they came from."""

    num_worlds: int
    naconmax: int
    njmax: int
    contacts_per_world: int
    naconmax_source: str
    njmax_source: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def resolve_warp_capacities(
    num_worlds: int,
    warp_naconmax: int | None = None,
    warp_njmax: int | None = None,
) -> WarpCapacityPlan:
    """Resolve Warp capacities for a batched environment."""
    if num_worlds < 1:
        raise ValueError("num_worlds must be at least 1")
    if warp_naconmax is not None and warp_naconmax < num_worlds:
        raise ValueError(
            "warp_naconmax is total contact capacity across all worlds and "
            f"must be >= num_worlds ({num_worlds}); got {warp_naconmax}."
        )
    if warp_njmax is not None and warp_njmax < 1:
        raise ValueError("warp_njmax must be at least 1")

    naconmax = (
        warp_naconmax
        if warp_naconmax is not None
        else DEFAULT_WARP_CONTACTS_PER_WORLD * num_worlds
    )
    njmax = warp_njmax or DEFAULT_WARP_NJMAX_PER_WORLD
    return WarpCapacityPlan(
        num_worlds=num_worlds,
        naconmax=naconmax,
        njmax=njmax,
        contacts_per_world=max(1, naconmax // num_worlds),
        naconmax_source=(
            "USER_OVERRIDE"
            if warp_naconmax is not None
            else "ENGINEERING_DEFAULT_8_CONTACTS_PER_WORLD"
        ),
        njmax_source=(
            "USER_OVERRIDE"
            if warp_njmax is not None
            else "ENGINEERING_DEFAULT_64_CONSTRAINTS_PER_WORLD"
        ),
    )


def package_version(name: str) -> str | None:
    """Return an installed distribution version without raising."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def backend_dependency_report() -> dict[str, str | None]:
    """Return installed package versions relevant for MJX-Warp."""
    return {
        name: package_version(name)
        for name in ("mujoco", "mujoco-mjx", "warp-lang", "jax", "jaxlib")
    }


def validate_backend_available(backend: PhysicsBackend) -> None:
    """Fail early when the requested backend cannot run."""
    if backend == "mjx_jax":
        return

    versions = backend_dependency_report()
    errors: list[str] = []
    if versions["mujoco-mjx"] is None:
        errors.append("mujoco-mjx is not installed")
    if versions["warp-lang"] is None:
        errors.append("warp-lang is not installed")
    if versions["mujoco"] != versions["mujoco-mjx"]:
        errors.append(
            "mujoco and mujoco-mjx versions do not match "
            f"({versions['mujoco']} != {versions['mujoco-mjx']})"
        )
    if "impl" not in inspect.signature(mjx.put_model).parameters:
        errors.append("installed mjx.put_model does not support impl='warp'")
    try:
        importlib.import_module("mujoco.mjx.warp")
    except ImportError as exc:
        errors.append(f"mujoco.mjx.warp cannot be imported: {exc}")

    if errors:
        raise RuntimeError(
            "MJX-Warp is the selected backend but is not available.\n"
            f"Reason: {'; '.join(errors)}\n"
            f"Install with: {OFFICIAL_WARP_INSTALL_COMMAND}"
        )


def warp_graph_mode_enum() -> tuple[Any, str]:
    """Return the GraphMode enum exported by the installed Warp packages."""
    errors: list[str] = []
    for module_name in ("mujoco.mjx.warp", "warp._src.jax_experimental.ffi"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: import failed ({exc})")
            continue
        graph_mode = getattr(module, "GraphMode", None)
        if graph_mode is None:
            errors.append(f"{module_name}: GraphMode not exported")
            continue
        return graph_mode, module_name
    raise RuntimeError(
        "Installed MJX-Warp packages do not expose GraphMode "
        f"({'; '.join(errors)}). Install with: {OFFICIAL_WARP_INSTALL_COMMAND}"
    )


def resolve_warp_graph_mode(name: WarpGraphModeName) -> Any:
    """Resolve a readable graph-mode name to the installed enum value."""
    graph_mode, _ = warp_graph_mode_enum()
    return {
        "warp": graph_mode.WARP,
        "warp_staged": graph_mode.WARP_STAGED,
        "warp_staged_ex": graph_mode.WARP_STAGED_EX,
    }[name]


def put_model_for_backend(
    host_model: Any,
    backend: PhysicsBackend,
    graph_mode: WarpGraphModeName = "warp",
) -> mjx.Model:
    """Convert a host MuJoCo model to MJX for the selected backend."""
    validate_backend_available(backend)
    if backend == "mjx_warp":
        # REF: BULLET-WARP-BACKEND
        # TYPE: REFERENCE_CODE_DERIVED
        return mjx.put_model(
            host_model,
            impl="warp",
            graph_mode=resolve_warp_graph_mode(graph_mode),
        )
    return mjx.put_model(host_model, impl="jax")


def make_data_kwargs(
    backend: PhysicsBackend,
    naconmax: int | None,
    njmax: int | None,
) -> dict[str, int | str]:
    """Build kwargs for ``mujoco_playground._src.mjx_env.make_data``."""
    if backend == "mjx_jax":
        return {"impl": "jax"}
    if naconmax is None or njmax is None:
        raise ValueError("MJX-Warp requires resolved naconmax and njmax values")
    return {"impl": "warp", "naconmax": naconmax, "njmax": njmax}


def select_data(
    current: mjx.Data,
    condition: jax.Array,
    replacement: mjx.Data,
    backend: PhysicsBackend,
) -> mjx.Data:
    """Select reset data while avoiding Warp fields that are not vmap-able."""
    if backend == "mjx_warp":
        where = getattr(current, "where", None)
        if where is not None:
            return where(condition, replacement)
        return _select_data_tree(
            current,
            condition,
            replacement,
            skip_warp_non_vmap=True,
        )

    return _select_data_tree(
        current,
        condition,
        replacement,
        skip_warp_non_vmap=False,
    )


def _data_path_name(path: jax.tree_util.KeyPath) -> str:
    """Return MJX-Warp's flattened Data field name for a PyTree path."""
    if any(isinstance(part, jax.tree_util.SequenceKey) for part in path):
        sequence_flags = [
            isinstance(part, jax.tree_util.SequenceKey) for part in path
        ]
        path = path[: sequence_flags.index(True)]
    names = [
        part.name
        for part in path
        if isinstance(part, jax.tree_util.GetAttrKey) and part.name != "_impl"
    ]
    return "__".join(names)


def _warp_non_vmap_fields() -> set[str]:
    """Return Warp Data fields that should not be selected per environment."""
    try:
        from mujoco.mjx._src import types as mjx_types

        return set(getattr(mjx_types.mjxw_types, "DATA_NON_VMAP", ()))
    except (AttributeError, ImportError):
        return set()


def _select_data_tree(
    current: Any,
    condition: jax.Array,
    replacement: Any,
    *,
    skip_warp_non_vmap: bool,
) -> Any:
    """PyTree selection compatible with MJX-JAX and packaged MJX-Warp."""
    warp_non_vmap = _warp_non_vmap_fields() if skip_warp_non_vmap else set()

    def select_leaf(new_value: jax.Array, old_value: jax.Array) -> jax.Array:
        if not hasattr(old_value, "shape"):
            return old_value
        mask = condition
        if getattr(mask, "shape", ()):
            if not old_value.shape or old_value.shape[0] != mask.shape[0]:
                return old_value
            mask = jnp.reshape(
                mask,
                [mask.shape[0]] + [1] * (len(old_value.shape) - 1),
            )
        return jnp.where(mask, new_value, old_value)

    if not warp_non_vmap:
        return jax.tree.map(select_leaf, replacement, current)

    def select_leaf_with_path(
        path: jax.tree_util.KeyPath,
        new_value: jax.Array,
        old_value: jax.Array,
    ) -> jax.Array:
        if _data_path_name(path) in warp_non_vmap and getattr(condition, "shape", ()):
            return old_value
        return select_leaf(new_value, old_value)

    return jax.tree_util.tree_map_with_path(
        select_leaf_with_path,
        replacement,
        current,
    )
