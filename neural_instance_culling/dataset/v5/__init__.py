"""GCOF-PVS V5 geometry-only preprocessing assets.

The package intentionally has no imports from the legacy dataset builders.  A
V5 scene can be compiled from geometry alone; labels and external-hit probes
are separate, permission-checked source-training assets.
"""

from .schemas import (
    EXTERNAL_HIT_PROBE_SCHEMA,
    FOLD_ACCESS_SCHEMA,
    LOCAL_SURFACE_SCHEMA,
    REGION_SUPPORT_SCHEMA,
    RELATION_SCHEMA,
    SYNTHETIC_SCENE_SCHEMA,
)

__all__ = [
    "EXTERNAL_HIT_PROBE_SCHEMA",
    "FOLD_ACCESS_SCHEMA",
    "LOCAL_SURFACE_SCHEMA",
    "REGION_SUPPORT_SCHEMA",
    "RELATION_SCHEMA",
    "SYNTHETIC_SCENE_SCHEMA",
]
