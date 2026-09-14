"""Locating and loading NuBench prediction parquet files.

Given a --model/--detector/--feature selection, finds and loads the right
prediction parquet file from a local directory of already-downloaded
NuBench files - generalizing the original NuBench_Plots notebooks' own
`download_files` helper (base URL + per-model filename construction), but
reading what's already on disk rather than fetching from the network.
Downloading directly from the network can be layered in later without
changing `find_prediction_file`/`load_predictions`'s own interface.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from nubench.style import MODEL_COLORS

PathLike = Union[str, Path]

# Maps each reconstruction task this package supports to the filename
# substring NuBench's own prediction files use for it - the `target`
# argument in the original notebooks' `download_files` helper (e.g.
# "training_arca_full_direction_test_results.parquet" for "direction").
FEATURE_TARGETS: Dict[str, str] = {
    "energy": "initial_state_energy",
    "direction": "direction",
    "vertex": "position",
    "inelasticity": "visible_inelasticity",
    "classification": "track",
}

# The four paper models (see `nubench.style`), used to disambiguate
# DynEdge's own conventionally-unprefixed files from the others.
KNOWN_MODELS = list(MODEL_COLORS)

# NuBench's own standard column names per task, for the DataFrame schema
# real prediction parquet files actually use. `is_track` is not a stored
# column at all in any of these files - it's always derived the same way
# (see `add_is_track_column`), restricted to muon-neutrino events.
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

# The classifier score column's name isn't consistent across every real
# download source - e.g. it's "target_pred" in some files and
# "track_pred" in others, and a single multi-panel call can mix
# detectors from *different* sources (each with its own convention) in
# one go - so `normalize_score_column` renames whichever one is present
# to this single canonical name at load time, once per DataFrame, rather
# than resolving it once per call and risking applying the wrong name to
# a detector that uses the other convention.
CANONICAL_SCORE_COLUMN = "target_pred"
SCORE_COLUMN_CANDIDATES = ["target_pred", "track_pred"]


def resolve_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return whichever of `candidates` is an actual column of `df`.

    Some NuBench column names aren't perfectly consistent across every
    real download source (see `SCORE_COLUMN_CANDIDATES`). Rather than
    hardcoding one name, callers that need to tolerate this pass every
    name they know about, in preference order.

    Args:
        df: DataFrame to check.
        candidates: Column names to try, in order of preference.

    Returns:
        The first name in `candidates` that's an actual column of `df`.

    Raises:
        KeyError: If none of `candidates` is a column of `df`.
    """
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(
        f"None of {candidates} is a column of this DataFrame "
        f"(columns: {list(df.columns)})"
    )


def normalize_score_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whichever classifier-score column is present to one
    canonical name (`CANONICAL_SCORE_COLUMN`).

    Applied once per DataFrame at load time, so every DataFrame
    downstream code ever sees already uses the same name - correct even
    when a single call mixes detectors from different download sources
    with different conventions (see `SCORE_COLUMN_CANDIDATES`), unlike
    resolving the name once per call from a single representative
    DataFrame.

    Args:
        df: DataFrame containing one of `SCORE_COLUMN_CANDIDATES`.

    Returns:
        `df`, unchanged if it already uses `CANONICAL_SCORE_COLUMN`,
        otherwise a copy with the found column renamed to it.
    """
    found = resolve_column(df, SCORE_COLUMN_CANDIDATES)
    if found == CANONICAL_SCORE_COLUMN:
        return df
    return df.rename(columns={found: CANONICAL_SCORE_COLUMN})


# Same real-world inconsistency as the classifier score column, just for
# the predicted inelasticity value: DynEdge's and ParticleNeT's own
# combined-file downloads use "inelasticity_pred", but GRIT's use
# "visible_inelasticity_pred" instead - found by exercising a real
# multi-model, multi-detector comparison (DynEdge/ParticleNeT/GRIT/
# DeepIce together), which a single-model check never surfaces.
CANONICAL_INELASTICITY_PRED_COLUMN = "inelasticity_pred"
INELASTICITY_PRED_COLUMN_CANDIDATES = [
    "inelasticity_pred", "visible_inelasticity_pred",
]


def normalize_inelasticity_pred_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whichever predicted-inelasticity column is present to one
    canonical name (`CANONICAL_INELASTICITY_PRED_COLUMN`).

    Same reasoning as `normalize_score_column`: applied once per
    DataFrame at load time (not once per call from a single
    representative DataFrame), so a call mixing models with different
    conventions - e.g. DynEdge and GRIT together - still works.

    Args:
        df: DataFrame containing one of
            `INELASTICITY_PRED_COLUMN_CANDIDATES`.

    Returns:
        `df`, unchanged if it already uses
        `CANONICAL_INELASTICITY_PRED_COLUMN`, otherwise a copy with the
        found column renamed to it.
    """
    found = resolve_column(df, INELASTICITY_PRED_COLUMN_CANDIDATES)
    if found == CANONICAL_INELASTICITY_PRED_COLUMN:
        return df
    return df.rename(columns={found: CANONICAL_INELASTICITY_PRED_COLUMN})


# Every feature except inelasticity needs the derived track/cascade (CC/
# NC) column, either to split by topology (energy/direction/vertex) or as
# the classifier's own truth label (classification).
FEATURES_NEEDING_IS_TRACK = {"energy", "direction", "vertex", "classification"}

# inelasticity doesn't split by topology - it needs the data *filtered*
# down to track/CC events instead. The original NuBench_Plots notebooks'
# own `plot_inelasticity_vs_energy`/`plot_inelasticity_dist` both apply
# this same restriction before computing anything at all: inelasticity
# for NC/non-muon events follows different physics and would otherwise
# dominate the residual with events the paper's own plots never include.
FEATURES_NEEDING_TRACK_ONLY_FILTER = {"inelasticity"}


def _is_track(df: pd.DataFrame) -> pd.Series:
    return (df["interaction"] == 1) & (df["initial_state_type"].abs() == 14)


def add_is_track_column(
    df: pd.DataFrame, column: str = "is_track"
) -> pd.DataFrame:
    """Add the standard track/cascade (CC/NC) selection column.

    NuBench prediction files never store this directly - every task's
    `*_by_topology` split, and classification's own truth label, use the
    same muon-neutrino-restricted convention instead:
    `(interaction == 1) & (initial_state_type.abs() == 14)`.

    Args:
        df: DataFrame containing `interaction` and `initial_state_type`.
        column: Name for the new boolean column.

    Returns:
        A copy of `df` with `column` added.
    """
    df = df.copy()
    df[column] = _is_track(df)
    return df


def filter_to_track_events(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to muon-neutrino CC ("track") events.

    Same convention as `add_is_track_column`, but returning the
    restricted subset itself rather than a boolean column - what
    inelasticity's own plots need (see `FEATURES_NEEDING_TRACK_ONLY_
    FILTER`).

    Args:
        df: DataFrame containing `interaction` and `initial_state_type`.

    Returns:
        The subset of `df` where
        `(interaction == 1) & (initial_state_type.abs() == 14)`.
    """
    return df[_is_track(df)]


def find_dataset_dir(root: PathLike, detector: str) -> Path:
    """Locate the directory holding one detector's downloaded files.

    Args:
        root: Directory containing one subdirectory per detector (e.g.
            the local `Dataset/` folder these files were downloaded
            into).
        detector: Detector key to look for, matched as a case-
            insensitive substring of the subdirectory name (e.g. "arca"
            matches a directory named "nubench_arca_dynedge") - a
            substring match rather than an exact naming convention,
            since different downloads may name these directories
            differently.

    Returns:
        The single matching subdirectory.

    Raises:
        FileNotFoundError: If no subdirectory matches.
        ValueError: If more than one subdirectory matches - pass a more
            specific `detector`, or skip this and call
            `find_prediction_file`/`load_predictions` directly with the
            exact directory.
    """
    root = Path(root)
    matches = sorted(
        p for p in root.iterdir()
        if p.is_dir() and detector.lower() in p.name.lower()
    )
    if not matches:
        raise FileNotFoundError(
            f"No directory matching detector={detector!r} found under "
            f"{root}"
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
    """Locate one model's prediction parquet file for one task.

    Two real file layouts exist and are both supported: NuBench's
    per-feature layout (one parquet file per task, matched by the
    `feature`/`target` filename substring below - e.g. the files this
    project's own local test data uses), and the combined-file layout
    the paper's own "Predictions" downloads actually give you (one
    parquet per *model*, holding every task's columns together, with no
    per-feature filename distinction at all). If no file matches the
    per-feature substring, every parquet file in `dataset_dir` becomes a
    candidate instead - correct either way, since a combined file
    already contains whichever columns `feature` needs.

    Args:
        dataset_dir: Directory holding one detector's downloaded
            prediction files (see `find_dataset_dir`).
        feature: One of the five reconstruction tasks this package
            supports - a key of `FEATURE_TARGETS`.
        model: Model name to disambiguate by, e.g. "DynEdge" - matched
            as a case-insensitive substring of the filename, except for
            "DynEdge" itself in the per-feature layout: those files
            conventionally omit any model name at all for DynEdge
            (following the original notebooks' own download-naming
            quirk), so "belongs to DynEdge" there means "doesn't look
            like it belongs to one of the other known models" rather
            than a literal name match. If None, every candidate file is
            eligible - fine as long as exactly one exists.

    Returns:
        The single matching parquet file.

    Raises:
        ValueError: If `feature` isn't recognized, or more than one file
            matches (pass `model` to disambiguate).
        FileNotFoundError: If no file matches.
    """
    if feature not in FEATURE_TARGETS:
        raise ValueError(
            f"Unknown feature {feature!r}; must be one of "
            f"{sorted(FEATURE_TARGETS)}"
        )
    dataset_dir = Path(dataset_dir)
    target = FEATURE_TARGETS[feature]
    candidates = sorted(dataset_dir.glob(f"*{target}*.parquet"))
    if not candidates:
        # No per-feature file matched - fall back to the combined-file
        # layout, where every parquet file is a candidate (one per
        # model, each holding every task's columns).
        candidates = sorted(dataset_dir.glob("*.parquet"))
    if model is not None:
        if model == "DynEdge":
            other_models = [m for m in KNOWN_MODELS if m != "DynEdge"]
            filtered = [
                p for p in candidates
                if not any(
                    other.lower() in p.name.lower() for other in other_models
                )
            ]
        else:
            filtered = [
                p for p in candidates if model.lower() in p.name.lower()
            ]
        # Fall back to the unfiltered candidates if the model filter
        # leaves nothing - better to surface an unambiguous single match
        # (or a clear "multiple candidates" error below) than to hide a
        # file that simply doesn't follow the expected naming quirk.
        if filtered:
            candidates = filtered
    if not candidates:
        model_part = f", model={model!r}" if model else ""
        raise FileNotFoundError(
            f"No prediction file found for feature={feature!r}{model_part} "
            f"in {dataset_dir}"
        )
    if len(candidates) > 1:
        model_part = f", model={model!r}" if model else ""
        raise ValueError(
            f"Multiple candidate files found for feature={feature!r}"
            f"{model_part} in {dataset_dir}: "
            f"{[p.name for p in candidates]} - pass `model` to "
            "disambiguate."
        )
    return candidates[0]


def load_predictions(
    dataset_dir: PathLike,
    feature: str,
    model: Optional[str] = None,
) -> pd.DataFrame:
    """Load one model's prediction parquet file for one task.

    Thin wrapper around `find_prediction_file` + `pandas.read_parquet` -
    see `find_prediction_file` for how the file is located.

    Args:
        dataset_dir: Directory holding one detector's downloaded
            prediction files.
        feature: One of the five reconstruction tasks this package
            supports.
        model: Model name to disambiguate by, if more than one model's
            files live in `dataset_dir` for the same `feature`.

    Returns:
        The loaded prediction DataFrame.
    """
    path = find_prediction_file(dataset_dir, feature, model=model)
    return pd.read_parquet(path)
