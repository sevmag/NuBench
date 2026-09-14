"""Unit tests for `nubench.cli`."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # No display available/needed for tests.

from nubench.cli import (  # noqa: E402
    build_arg_parser,
    load_feature_predictions,
    main,
    make_figure,
)


def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A synthetic DataFrame with every column every feature needs.

    Matches NuBench's real column-naming convention (see
    `nubench.data.DEFAULT_COLUMNS`), so `make_figure`/`load_predictions`
    can be exercised end-to-end without a real downloaded file.
    """
    rng = np.random.default_rng(seed=seed)
    energy = rng.uniform(10, 10_000, size=n)
    zenith = rng.uniform(0, np.pi, size=n)
    azimuth = rng.uniform(0, 2 * np.pi, size=n)
    dir_x = np.sin(zenith) * np.cos(azimuth)
    dir_y = np.sin(zenith) * np.sin(azimuth)
    dir_z = np.cos(zenith)
    noise = rng.normal(scale=0.05, size=n)
    x = rng.uniform(-100, 100, size=n)
    y = rng.uniform(-100, 100, size=n)
    z = rng.uniform(-100, 100, size=n)
    visible_inelasticity = rng.uniform(0, 1, size=n)
    interaction = rng.integers(1, 3, size=n)  # 1 == CC, 2 == NC
    initial_state_type = rng.choice([14, -14], size=n)
    return pd.DataFrame(
        {
            "initial_state_energy": energy,
            "energy_pred": energy * (1 + noise),
            "initial_state_zenith": zenith,
            "initial_state_azimuth": azimuth,
            "dir_x_pred": dir_x + noise,
            "dir_y_pred": dir_y + noise,
            "dir_z_pred": dir_z + noise,
            "muon_zenith": zenith + noise,
            "muon_azimuth": azimuth + noise,
            "initial_state_x": x,
            "initial_state_y": y,
            "initial_state_z": z,
            "position_x_pred": x + noise * 10,
            "position_y_pred": y + noise * 10,
            "position_z_pred": z + noise * 10,
            "visible_inelasticity": visible_inelasticity,
            "inelasticity_pred": np.clip(
                visible_inelasticity + noise, 0.0, 1.0
            ),
            "target_pred": rng.uniform(0, 1, size=n),
            "interaction": interaction,
            "initial_state_type": initial_state_type,
        }
    )


def _make_dataset_dir(root: Path, detector: str = "arca") -> Path:
    dataset_dir = root / f"nubench_{detector}_dynedge"
    dataset_dir.mkdir(parents=True)
    df = _make_df()
    for feature, target in [
        ("energy", "initial_state_energy"),
        ("direction", "direction"),
        ("vertex", "position"),
        ("inelasticity", "visible_inelasticity"),
        ("classification", "track"),
    ]:
        df.to_parquet(
            dataset_dir
            / f"training_{detector}_full_{target}_test_results.parquet"
        )
    return dataset_dir


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    _make_dataset_dir(tmp_path)
    return tmp_path


@pytest.fixture
def multi_dataset_root(tmp_path: Path) -> Path:
    _make_dataset_dir(tmp_path, detector="arca")
    _make_dataset_dir(tmp_path, detector="orca")
    return tmp_path


def test_load_feature_predictions_adds_is_track_where_needed(
    dataset_root: Path,
) -> None:
    """energy/direction/vertex/classification should get `is_track`."""
    predictions = load_feature_predictions(
        str(dataset_root), "arca", "energy", ["DynEdge"]
    )

    assert "is_track" in predictions["DynEdge"].columns


def test_load_feature_predictions_skips_is_track_for_inelasticity(
    dataset_root: Path,
) -> None:
    """inelasticity doesn't split by topology, so no column is needed."""
    predictions = load_feature_predictions(
        str(dataset_root), "arca", "inelasticity", ["DynEdge"]
    )

    assert "is_track" not in predictions["DynEdge"].columns


def test_load_feature_predictions_filters_inelasticity_to_track_only(
    dataset_root: Path,
) -> None:
    """inelasticity should be restricted to CC muon-neutrino events -
    matching the original NuBench_Plots notebooks' own
    `plot_inelasticity_vs_energy`/`plot_inelasticity_dist`, which both
    apply this same filter before computing anything.
    """
    predictions = load_feature_predictions(
        str(dataset_root), "arca", "inelasticity", ["DynEdge"]
    )

    df = predictions["DynEdge"]
    assert len(df) < 200  # strictly fewer than the full synthetic set
    assert (df["interaction"] == 1).all()
    assert (df["initial_state_type"].abs() == 14).all()


def test_load_feature_predictions_one_entry_per_model(
    dataset_root: Path,
) -> None:
    """Should return one DataFrame per requested model."""
    predictions = load_feature_predictions(
        str(dataset_root), "arca", "energy", ["DynEdge"]
    )

    assert set(predictions.keys()) == {"DynEdge"}


@pytest.mark.parametrize(
    "feature",
    ["energy", "direction", "vertex", "inelasticity", "classification"],
)
def test_make_figure_returns_a_figure_for_every_feature(
    dataset_root: Path, feature: str
) -> None:
    """Every supported feature should produce a savable Figure."""
    predictions = load_feature_predictions(
        str(dataset_root), "arca", feature, ["DynEdge"]
    )

    fig = make_figure(feature, predictions)

    assert fig is not None
    assert len(fig.axes) > 0


def test_make_figure_unknown_feature_raises(dataset_root: Path) -> None:
    """An unrecognized feature should raise, not silently do nothing."""
    predictions = load_feature_predictions(
        str(dataset_root), "arca", "energy", ["DynEdge"]
    )

    with pytest.raises(ValueError):
        make_figure("not_a_real_feature", predictions)


def test_build_arg_parser_requires_core_arguments() -> None:
    """Missing required arguments should exit with an error."""
    parser = build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_arg_parser_rejects_unknown_feature() -> None:
    """An unsupported --feature value should be rejected up front."""
    parser = build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--data-root", "/tmp",
                "--detector", "arca",
                "--feature", "not_a_real_feature",
                "--model", "DynEdge",
            ]
        )


def test_build_arg_parser_accepts_multiple_models() -> None:
    """Repeated --model flags should accumulate into a list."""
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--data-root", "/tmp",
            "--detector", "arca",
            "--feature", "energy",
            "--model", "DynEdge",
            "--model", "ParticleNeT",
        ]
    )

    assert args.models == ["DynEdge", "ParticleNeT"]


def test_main_saves_the_expected_output_file(
    dataset_root: Path, tmp_path: Path
) -> None:
    """End-to-end: parse args, build the figure, save it to --output."""
    output = tmp_path / "out.png"

    main(
        [
            "--data-root", str(dataset_root),
            "--detector", "arca",
            "--feature", "energy",
            "--model", "DynEdge",
            "--output", str(output),
        ]
    )

    assert output.exists()


def test_main_defaults_output_to_feature_name(
    dataset_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no --output, the file should default to '<feature>.png'."""
    monkeypatch.chdir(tmp_path)

    main(
        [
            "--data-root", str(dataset_root),
            "--detector", "arca",
            "--feature", "inelasticity",
            "--model", "DynEdge",
        ]
    )

    assert (tmp_path / "inelasticity.png").exists()


def test_main_rejects_multiple_detectors_without_multi_panel(
    multi_dataset_root: Path,
) -> None:
    """Repeated --detector without --multi-panel should error clearly."""
    with pytest.raises(SystemExit):
        main(
            [
                "--data-root", str(multi_dataset_root),
                "--detector", "arca",
                "--detector", "orca",
                "--feature", "energy",
                "--model", "DynEdge",
            ]
        )


def test_main_multi_panel_saves_a_grid_figure(
    multi_dataset_root: Path, tmp_path: Path
) -> None:
    """--multi-panel with two --detector flags should save one figure."""
    output = tmp_path / "multi.png"

    main(
        [
            "--data-root", str(multi_dataset_root),
            "--detector", "arca",
            "--detector", "orca",
            "--feature", "energy",
            "--model", "DynEdge",
            "--multi-panel",
            "--output", str(output),
        ]
    )

    assert output.exists()


def test_main_works_against_the_combined_file_download_layout(
    tmp_path: Path,
) -> None:
    """End-to-end: the real "one combined file per model" download
    layout (no per-feature filenames, and a "track_pred" score column
    instead of "target_pred") should work with no special handling from
    the caller.
    """
    dataset_dir = tmp_path / "nubench_trident_dynedge"
    dataset_dir.mkdir(parents=True)
    df = _make_df().rename(columns={"target_pred": "track_pred"})
    df.to_parquet(dataset_dir / "DynEdge_v1.0_flowerxl.parquet")
    output = tmp_path / "classification.png"

    main(
        [
            "--data-root", str(tmp_path),
            "--detector", "trident",
            "--feature", "classification",
            "--model", "DynEdge",
            "--output", str(output),
        ]
    )

    assert output.exists()
