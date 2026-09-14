"""Unit tests for `nubench.data`."""

from pathlib import Path

import pandas as pd
import pytest

from nubench.data import (
    add_is_track_column,
    filter_to_track_events,
    find_dataset_dir,
    find_prediction_file,
    load_predictions,
    normalize_inelasticity_pred_column,
    normalize_score_column,
    resolve_column,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


# The two real filenames that make disambiguating DynEdge from ParticleNeT
# a genuine (not merely hypothetical) case: both exist side by side for
# "arca"/"energy" in the actual downloaded dataset.
DYNEDGE_ENERGY_FILENAME = (
    "training_arca_full_initial_state_energy_test_results.parquet"
)
PARTICLENET_ENERGY_FILENAME = "ParticleNeT_" + DYNEDGE_ENERGY_FILENAME


def test_find_dataset_dir_matches_substring(tmp_path: Path) -> None:
    """Should find a subdirectory whose name contains the detector key."""
    (tmp_path / "nubench_arca_dynedge").mkdir()

    result = find_dataset_dir(tmp_path, "arca")

    assert result == tmp_path / "nubench_arca_dynedge"


def test_find_dataset_dir_is_case_insensitive(tmp_path: Path) -> None:
    """The substring match shouldn't care about case."""
    (tmp_path / "nubench_ARCA_dynedge").mkdir()

    result = find_dataset_dir(tmp_path, "arca")

    assert result == tmp_path / "nubench_ARCA_dynedge"


def test_find_dataset_dir_raises_when_missing(tmp_path: Path) -> None:
    """No matching subdirectory should raise FileNotFoundError."""
    (tmp_path / "nubench_orca_dynedge").mkdir()

    with pytest.raises(FileNotFoundError):
        find_dataset_dir(tmp_path, "arca")


def test_find_dataset_dir_raises_when_ambiguous(tmp_path: Path) -> None:
    """More than one matching subdirectory should raise ValueError."""
    (tmp_path / "nubench_arca_dynedge").mkdir()
    (tmp_path / "nubench_arca_particlenet").mkdir()

    with pytest.raises(ValueError):
        find_dataset_dir(tmp_path, "arca")


def test_find_prediction_file_single_match(tmp_path: Path) -> None:
    """A single matching file should be found without needing `model`."""
    _touch(tmp_path / "training_arca_full_direction_test_results.parquet")

    result = find_prediction_file(tmp_path, feature="direction")

    assert result.name == "training_arca_full_direction_test_results.parquet"


def test_find_prediction_file_unknown_feature_raises(tmp_path: Path) -> None:
    """An unrecognized `feature` should raise ValueError, not fail silently."""
    with pytest.raises(ValueError):
        find_prediction_file(tmp_path, feature="not_a_real_feature")


def test_find_prediction_file_missing_raises(tmp_path: Path) -> None:
    """No matching file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        find_prediction_file(tmp_path, feature="direction")


def test_find_prediction_file_ambiguous_without_model_raises(
    tmp_path: Path,
) -> None:
    """Two candidate files with no `model` given should raise ValueError."""
    _touch(tmp_path / DYNEDGE_ENERGY_FILENAME)
    _touch(tmp_path / PARTICLENET_ENERGY_FILENAME)

    with pytest.raises(ValueError):
        find_prediction_file(tmp_path, feature="energy")


def test_find_prediction_file_dynedge_is_the_unprefixed_file(
    tmp_path: Path,
) -> None:
    """model="DynEdge" should pick the file with no other model's name in it.

    Matches the real NuBench download convention: DynEdge's own files
    never carry a model prefix, unlike every other model's.
    """
    dynedge_file = tmp_path / DYNEDGE_ENERGY_FILENAME
    _touch(dynedge_file)
    _touch(tmp_path / PARTICLENET_ENERGY_FILENAME)

    result = find_prediction_file(tmp_path, feature="energy", model="DynEdge")

    assert result == dynedge_file


def test_find_prediction_file_other_model_matches_its_prefix(
    tmp_path: Path,
) -> None:
    """model="ParticleNeT" should pick the file carrying that prefix."""
    particlenet_file = tmp_path / PARTICLENET_ENERGY_FILENAME
    _touch(tmp_path / DYNEDGE_ENERGY_FILENAME)
    _touch(particlenet_file)

    result = find_prediction_file(
        tmp_path, feature="energy", model="ParticleNeT"
    )

    assert result == particlenet_file


def test_load_predictions_reads_the_located_file(tmp_path: Path) -> None:
    """Should locate and load the file, returning a real DataFrame."""
    path = tmp_path / "training_arca_full_direction_test_results.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(path)

    result = load_predictions(tmp_path, feature="direction")

    assert list(result["a"]) == [1, 2, 3]


def test_add_is_track_column_matches_known_convention() -> None:
    """Should match NuBench's own track/CC selection convention exactly."""
    df = pd.DataFrame(
        {
            "interaction": [1, 1, 2, 1],
            "initial_state_type": [14, -14, 14, 12],
        }
    )

    result = add_is_track_column(df)

    assert list(result["is_track"]) == [True, True, False, False]


def test_add_is_track_column_does_not_mutate_input() -> None:
    """Should return a copy, not modify the caller's DataFrame in place."""
    df = pd.DataFrame({"interaction": [1], "initial_state_type": [14]})

    add_is_track_column(df)

    assert "is_track" not in df.columns


def test_add_is_track_column_custom_name(tmp_path: Path) -> None:
    """The output column name should be configurable."""
    df = pd.DataFrame({"interaction": [1], "initial_state_type": [14]})

    result = add_is_track_column(df, column="is_cc")

    assert "is_cc" in result.columns


def test_filter_to_track_events_keeps_only_cc_muon_events() -> None:
    """Should match the same convention as `add_is_track_column`, but
    return the restricted subset itself rather than a boolean column.
    """
    df = pd.DataFrame(
        {
            "interaction": [1, 1, 2, 1],
            "initial_state_type": [14, -14, 14, 12],
            "value": ["cc_numu", "cc_antinumu", "nc_numu", "cc_nue"],
        }
    )

    result = filter_to_track_events(df)

    assert list(result["value"]) == ["cc_numu", "cc_antinumu"]
    assert "is_track" not in result.columns


def test_find_prediction_file_falls_back_to_combined_file_layout(
    tmp_path: Path,
) -> None:
    """When no file matches the per-feature substring, every parquet
    file should become a candidate instead - the real "one combined
    file per model" download layout, where filenames carry no
    per-feature distinction at all (e.g. "DynEdge_v1.0_flowerxl.parquet"
    holds every task's columns together).
    """
    combined_file = tmp_path / "DynEdge_v1.0_flowerxl.parquet"
    _touch(combined_file)

    result = find_prediction_file(tmp_path, feature="energy", model="DynEdge")

    assert result == combined_file


def test_find_prediction_file_combined_layout_disambiguates_by_model(
    tmp_path: Path,
) -> None:
    """The combined-file fallback should still disambiguate by model."""
    dynedge_file = tmp_path / "DynEdge_v1.0_flowerxl.parquet"
    _touch(dynedge_file)
    _touch(tmp_path / "ParticleNeT_v1.0_flowerxl.parquet")
    _touch(tmp_path / "GRIT_v1.0_flowerxl.parquet")
    _touch(tmp_path / "DeepIce_v1.0_flowerxl.parquet")

    result = find_prediction_file(tmp_path, feature="energy", model="DynEdge")

    assert result == dynedge_file


def test_resolve_column_returns_first_match() -> None:
    """Should return whichever candidate name is actually present."""
    df = pd.DataFrame({"track_pred": [0.1, 0.2]})

    result = resolve_column(df, ["target_pred", "track_pred"])

    assert result == "track_pred"


def test_resolve_column_prefers_earlier_candidates() -> None:
    """When both candidates are present, the earlier one should win."""
    df = pd.DataFrame({"target_pred": [0.1], "track_pred": [0.2]})

    result = resolve_column(df, ["target_pred", "track_pred"])

    assert result == "target_pred"


def test_resolve_column_raises_when_none_match() -> None:
    """No matching candidate should raise a clear error, not KeyError
    from a raw column lookup somewhere downstream.
    """
    df = pd.DataFrame({"something_else": [0.1]})

    with pytest.raises(KeyError):
        resolve_column(df, ["target_pred", "track_pred"])


def test_normalize_score_column_renames_to_canonical_name() -> None:
    """A DataFrame using the non-canonical name should get renamed."""
    df = pd.DataFrame({"track_pred": [0.1, 0.2], "other": [1, 2]})

    result = normalize_score_column(df)

    assert "target_pred" in result.columns
    assert "track_pred" not in result.columns
    assert list(result["target_pred"]) == [0.1, 0.2]


def test_normalize_score_column_leaves_canonical_name_untouched() -> None:
    """A DataFrame already using the canonical name shouldn't change."""
    df = pd.DataFrame({"target_pred": [0.1, 0.2]})

    result = normalize_score_column(df)

    assert list(result.columns) == ["target_pred"]


def test_normalize_inelasticity_pred_column_renames_to_canonical_name() -> (
    None
):
    """GRIT's own "visible_inelasticity_pred" should get renamed - found
    by exercising a real multi-model comparison (DynEdge/ParticleNeT
    use "inelasticity_pred", GRIT uses this other name instead).
    """
    df = pd.DataFrame({"visible_inelasticity_pred": [0.1, 0.2]})

    result = normalize_inelasticity_pred_column(df)

    assert "inelasticity_pred" in result.columns
    assert "visible_inelasticity_pred" not in result.columns
    assert list(result["inelasticity_pred"]) == [0.1, 0.2]


def test_normalize_inelasticity_pred_column_leaves_canonical_untouched() -> (
    None
):
    """A DataFrame already using the canonical name shouldn't change."""
    df = pd.DataFrame({"inelasticity_pred": [0.1, 0.2]})

    result = normalize_inelasticity_pred_column(df)

    assert list(result.columns) == ["inelasticity_pred"]
