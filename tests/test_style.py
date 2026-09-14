"""Unit tests for `nubench.style`."""

from nubench.style import (
    detector_color,
    detector_display_name,
    model_color,
    model_linestyle,
)

# The four models and seven detectors covered by the NuBench paper.
KNOWN_MODELS = ("DynEdge", "ParticleNeT", "GRIT", "DeepIce")
KNOWN_DETECTOR_KEYS = (
    "arca",
    "orca",
    "trident",
    "icecube_water",
    "icecube_ice",
    "gvd",
    "pone",
)


def test_model_color_covers_every_known_model() -> None:
    """Every paper model should have a color, not just some of them."""
    for model in KNOWN_MODELS:
        assert model_color(model) is not None


def test_model_linestyle_covers_every_known_model() -> None:
    """Every paper model should have a linestyle, not just some of them."""
    for model in KNOWN_MODELS:
        assert model_linestyle(model) is not None


def test_model_color_is_unique_per_model() -> None:
    """No two models should accidentally share the same color."""
    colors = [model_color(model) for model in KNOWN_MODELS]
    assert len(set(colors)) == len(colors)


def test_model_linestyle_is_unique_per_model() -> None:
    """No two models should accidentally share the same linestyle."""
    linestyles = [model_linestyle(model) for model in KNOWN_MODELS]
    assert len(set(linestyles)) == len(linestyles)


def test_model_color_unknown_model_returns_none() -> None:
    """An unrecognized model should not raise - just fall back to None."""
    assert model_color("SomeFutureModel") is None


def test_model_linestyle_unknown_model_returns_none() -> None:
    """An unrecognized model should not raise - just fall back to None."""
    assert model_linestyle("SomeFutureModel") is None


def test_detector_display_name_covers_every_known_detector() -> None:
    """Every paper detector key should map to a display name."""
    for detector in KNOWN_DETECTOR_KEYS:
        name = detector_display_name(detector)
        assert isinstance(name, str)
        assert len(name) > 0


def test_detector_display_name_known_values() -> None:
    """Spot-check a few of the actual detector -> display-name mappings."""
    assert detector_display_name("arca") == "Flower L"
    assert detector_display_name("icecube_water") == "Hexagon"
    assert detector_display_name("icecube_ice") == "Hexagon Ice LE"


def test_detector_display_name_unknown_key_raises() -> None:
    """Unlike model lookups, an unrecognized detector key should raise."""
    try:
        detector_display_name("not_a_real_detector")
    except KeyError:
        pass
    else:
        raise AssertionError("Expected a KeyError for an unknown detector.")


def test_detector_color_covers_every_known_detector() -> None:
    """Every detector's display name should have a color."""
    for detector in KNOWN_DETECTOR_KEYS:
        name = detector_display_name(detector)
        assert detector_color(name) is not None


def test_detector_color_is_unique_per_detector() -> None:
    """No two detectors should accidentally share the same color."""
    colors = [
        detector_color(detector_display_name(d)) for d in KNOWN_DETECTOR_KEYS
    ]
    assert len(set(colors)) == len(colors)
