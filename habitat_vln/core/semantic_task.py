"""Validated weak priors permitted in the single-building indoor task."""

import json
from pathlib import Path

ALLOWED_BUILDING_PRIOR_FIELDS = {
    "building_id",
    "start_region",
    "known_entrances",
    "floor_labels",
    "footprint_summary",
    "public_structure_summary",
    "known_regions",
}


def load_building_prior(path):
    """Load a small, target-free building prior from a JSON object.

    The schema intentionally accepts descriptive public information only. It
    rejects unknown fields so a target coordinate or route cannot silently enter
    the policy context under an arbitrary key.
    """
    prior_path = Path(path).expanduser().resolve()
    with prior_path.open(encoding="utf-8") as handle:
        prior = json.load(handle)
    if not isinstance(prior, dict):
        raise ValueError("--building-prior-file must contain one JSON object")
    unknown_fields = sorted(set(prior) - ALLOWED_BUILDING_PRIOR_FIELDS)
    if unknown_fields:
        raise ValueError(
            "--building-prior-file contains unsupported fields "
            f"{unknown_fields}; allowed fields are "
            f"{sorted(ALLOWED_BUILDING_PRIOR_FIELDS)}"
        )
    return prior
