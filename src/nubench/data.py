"""Locating and loading NuBench prediction parquet files.

Given a model/detector/feature selection, finds and loads the right
parquet file from a local directory of already-downloaded NuBench files -
the same per-model filename construction the original NuBench_Plots
notebooks' `download_files` helper did, but reading from disk. Fetching
over the network could be layered in later without changing
`find_prediction_file`/`load_predictions`' interface.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from scipy.special import expit  # numerically stable sigmoid

from nubench.style import MODEL_COLORS

PathLike = Union[str, Path]

# Each task mapped to the filename substring NuBench's own prediction
# files use for it - the `target` argument in the original notebooks'
# `download_files` (e.g. "training_arca_full_direction_test_results
# .parquet" for "direction").
FEATURE_TARGETS: Dict[str, str] = {
    "energy": "initial_state_energy",
    "direction": "direction",
    "vertex": "position",
    "inelasticity": "visible_inelasticity",
    "classification": "track",
}

# The four paper models, used to tell DynEdge's conventionally
# unprefixed files from the others.
KNOWN_MODELS = list(MODEL_COLORS)

# NuBench's own column names per task. `is_track` is never stored in any
# of these files - it is always derived (see `add_is_track_column`).
DEFAULT_COLUMNS: Dict[str, Dict[str, str]] = {
    "energy": {
        "truth_col": "initial_state_energy",
        "pred_col": "energy_pred",
    },
    "direction": {
        "truth_zenith_col": "initial_state_zenith",
        "truth_azimuth_col": "initial_state_azimuth",
        "pred_x_col": "dir_x_pred",
        "pred_y_col": "dir_y_pred",
        "pred_z_col": "dir_z_pred",
        "energy_col": "initial_state_energy",
        "muon_zenith_col": "muon_zenith",
        "muon_azimuth_col": "muon_azimuth",
    },
    "vertex": {
        "truth_x_col": "initial_state_x",
        "truth_y_col": "initial_state_y",
        "truth_z_col": "initial_state_z",
        "pred_x_col": "position_x_pred",
        "pred_y_col": "position_y_pred",
        "pred_z_col": "position_z_pred",
        "energy_col": "initial_state_energy",
    },
    "inelasticity": {
        "truth_col": "visible_inelasticity",
        "pred_col": "inelasticity_pred",
        "energy_col": "initial_state_energy",
    },
    "classification": {
        "truth_col": "is_track",
        "score_col": "target_pred",
        "energy_col": "initial_state_energy",
    },
}

# Two column names aren't consistent across download sources: the
# classifier score is "target_pred" in some files and "track_pred" in
# others, and the predicted inelasticity is "inelasticity_pred" for
# DynEdge/ParticleNeT but "visible_inelasticity_pred" for GRIT. A single
# multi-panel call can mix detectors from different sources, so the
# renaming happens once per DataFrame at load time rather than once per
# call - otherwise one detector gets the other's name applied.
CANONICAL_SCORE_COLUMN = "target_pred"
SCORE_COLUMN_CANDIDATES = ["target_pred", "track_pred"]
CANONICAL_INELASTICITY_PRED_COLUMN = "inelasticity_pred"
INELASTICITY_PRED_COLUMN_CANDIDATES = [
    "inelasticity_pred", "visible_inelasticity_pred",
]


def resolve_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """The first of `candidates` that is an actual column of `df`.

    Raises KeyError if none is.
    """
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(
        f"None of {candidates} is a column of this DataFrame "
        f"(columns: {list(df.columns)})"
    )


def _normalize_column(
    df: pd.DataFrame, candidates: List[str], canonical: str
) -> pd.DataFrame:
    """Rename whichever of `candidates` is present to `canonical`."""
    found = resolve_column(df, candidates)
    if found == canonical:
        return df
    return df.rename(columns={found: canonical})


def normalize_score_column(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize the classifier score's column name *and* its scale.

    Some files store a raw logit instead of a probability (e.g NuBench's own
    arca/GRIT track file). Applying the sigmoid to those is necessary.

    For example the track-score histogram's bins span the combined
    range across models, so one logit-valued model would otherwise
    stretch the axis and squash every other model into a sliver.
    """
    df = _normalize_column(
        df, SCORE_COLUMN_CANDIDATES, CANONICAL_SCORE_COLUMN
    )
    scores = df[CANONICAL_SCORE_COLUMN]
    if scores.max() > 1.0:
        # `assign` returns a new frame sharing the untouched columns,
        # so the caller's DataFrame is never mutated.
        df = df.assign(**{CANONICAL_SCORE_COLUMN: expit(scores)})
    return df


def normalize_inelasticity_pred_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the predicted-inelasticity column to its canonical name."""
    return _normalize_column(
        df,
        INELASTICITY_PRED_COLUMN_CANDIDATES,
        CANONICAL_INELASTICITY_PRED_COLUMN,
    )


def _is_track(df: pd.DataFrame) -> pd.Series:
    return (df["interaction"] == 1) & (df["initial_state_type"].abs() == 14)


def add_is_track_column(
    df: pd.DataFrame, column: str = "is_track"
) -> pd.DataFrame:
    """Add the standard track/cascade (CC/NC) selection column.

    NuBench files never store this. Every task's `*_by_topology` split,
    and classification's truth label, use the same muon-neutrino
    restricted convention:
    `(interaction == 1) & (initial_state_type.abs() == 14)`.
    """
    df = df.copy()
    df[column] = _is_track(df)
    return df


def filter_to_track_events(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to muon-neutrino CC ("track") events.

    Same convention as `add_is_track_column`, returning the subset rather
    than a column - what inelasticity's plots need.
    """
    return df[_is_track(df)]


# What each feature needs doing to a freshly loaded DataFrame, in order.
# Every feature but inelasticity wants the derived track/cascade column,
# to split by topology or as the classifier's truth label. Inelasticity
# instead wants the data *filtered* to track/CC events via the function
# `filter_to_track_events`: NC and non-muon events follow
# different physics and would otherwise dominate the residual.
FEATURE_PREPARATION: Dict[str, List[Callable[[pd.DataFrame], pd.DataFrame]]]
FEATURE_PREPARATION = {
    "energy": [add_is_track_column],
    "direction": [add_is_track_column],
    "vertex": [add_is_track_column],
    "classification": [add_is_track_column, normalize_score_column],
    "inelasticity": [
        filter_to_track_events,
        normalize_inelasticity_pred_column,
    ],
}


def find_dataset_dir(root: PathLike, detector: str) -> Path:
    """The subdirectory of `root` holding one detector's files.

    `detector` is matched as a case-insensitive substring ("arca" matches
    "nubench_arca_dynedge") rather than by an exact convention, since
    different downloads name these differently. Raises FileNotFoundError
    if nothing matches, ValueError if several do.
    """
    root = Path(root)
    matches = sorted(
        p for p in root.iterdir()
        if p.is_dir() and detector.lower() in p.name.lower()
    )
    if not matches:
        raise FileNotFoundError(
            f"No directory matching detector={detector!r} found under {root}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple directories match detector={detector!r} under "
            f"{root}: {[p.name for p in matches]} - pass the exact "
            "directory instead."
        )
    return matches[0]


def find_prediction_file(
    dataset_dir: PathLike,
    feature: str,
    model: Optional[str] = None,
) -> Path:
    """The parquet file holding one model's predictions for one task.

    Both real layouts work. NuBench's per-feature layout has one file per
    task, matched by the `FEATURE_TARGETS` filename substring. The
    paper's own "Predictions" downloads instead give one file per
    *model*, holding every task's columns with no per-feature name - so
    when no filename matches the substring, every parquet file becomes a
    candidate, which is correct either way.

    `model` is matched as a case-insensitive substring, except for
    "DynEdge" in the per-feature layout: those files omit the model name
    entirely, so "DynEdge's" there means "doesn't look like any other
    known model's". Raises ValueError on an unknown feature or an
    ambiguous match, FileNotFoundError if nothing matches.
    """
    if feature not in FEATURE_TARGETS:
        raise ValueError(
            f"Unknown feature {feature!r}; must be one of "
            f"{sorted(FEATURE_TARGETS)}"
        )
    dataset_dir = Path(dataset_dir)
    candidates = sorted(
        dataset_dir.glob(f"*{FEATURE_TARGETS[feature]}*.parquet")
    ) or sorted(dataset_dir.glob("*.parquet"))
    if model is not None:
        filtered = [p for p in candidates if model.lower() in p.name.lower()]
        if not filtered and model == "DynEdge":
            # NuBench's per-feature downloads (arca, orca) omit the model
            # name for DynEdge alone, so fall back to elimination - but
            # only once nothing names DynEdge outright. This rule claims
            # *any* unprefixed file, so a model outside `KNOWN_MODELS`
            # would otherwise be silently returned as DynEdge's. Name
            # generated files `<Model>_...` and it never has to run.
            others = [m.lower() for m in KNOWN_MODELS if m != "DynEdge"]
            filtered = [
                p for p in candidates
                if not any(other in p.name.lower() for other in others)
            ]
        # No fallback to the unfiltered list: with one file left, that
        # would hand back another model's predictions under this name.
        candidates = filtered
    model_part = f", model={model!r}" if model else ""
    if not candidates:
        raise FileNotFoundError(
            f"No prediction file found for feature={feature!r}{model_part} "
            f"in {dataset_dir}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple candidate files found for feature={feature!r}"
            f"{model_part} in {dataset_dir}: "
            f"{[p.name for p in candidates]} - pass `model` to disambiguate."
        )
    return candidates[0]


def load_predictions(
    dataset_dir: PathLike,
    feature: str,
    model: Optional[str] = None,
) -> pd.DataFrame:
    """`find_prediction_file` plus `pandas.read_parquet`."""
    return pd.read_parquet(
        find_prediction_file(dataset_dir, feature, model=model)
    )


def load_feature_predictions(
    data_root: PathLike,
    detector: str,
    feature: str,
    models: List[str],
) -> Dict[str, pd.DataFrame]:
    """One detector's `{model: DataFrame}` predictions for `feature`.

    `data_root` holds one subdirectory per detector. Applies whichever
    derived columns and column-name normalizations `feature` needs, so
    every DataFrame downstream uses the same schema.
    """
    dataset_dir = find_dataset_dir(data_root, detector)
    predictions = {}
    for model in models:
        df = load_predictions(dataset_dir, feature, model=model)
        for prepare in FEATURE_PREPARATION[feature]:
            df = prepare(df)
        predictions[model] = df
    return predictions
