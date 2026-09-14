"""Shared visual constants (colors, linestyles, display names).

Centralizes the paper's visual language - transcribed from the NuBench_Plots
notebooks' own `model_to_ls`, `get_model_colors`, `dataset_rename`, and
`dataset_color` helpers, which were previously duplicated across all five
notebooks. One place, reused by every plotting function in
`nubench.evaluation.plotting`, so that e.g. "DynEdge" is drawn in the same
color everywhere rather than repeating hex codes throughout the codebase.
"""

from typing import Optional

MODEL_LINESTYLES = {
    "DynEdge": "--",
    "ParticleNeT": "-",
    "GRIT": "-.",
    "DeepIce": ":",
}

MODEL_COLORS = {
    "DynEdge": "#e377c2",
    "ParticleNeT": "#7f7f7f",
    "GRIT": "#bcbd22",
    "DeepIce": "black",
}

DETECTOR_DISPLAY_NAMES = {
    "arca": "Flower L",
    "orca": "Flower S",
    "trident": "Flower XL",
    "icecube_water": "Hexagon",
    "icecube_ice": "Hexagon Ice LE",
    "gvd": "Cluster",
    "pone": "Triangle",
}

DETECTOR_COLORS = {
    "Flower L": "tab:blue",
    "Flower S": "tab:orange",
    "Flower XL": "tab:green",
    "Hexagon": "tab:red",
    "Hexagon Ice LE": "#17becf",
    "Cluster": "tab:purple",
    "Triangle": "tab:brown",
}


def model_color(model: str) -> Optional[str]:
    """Return the paper's canonical color for `model`.

    Args:
        model: A model name, e.g. "DynEdge".

    Returns:
        The matplotlib color string used for this model in the paper, or
        None if `model` isn't one of the four paper models - callers
        should treat None as "let matplotlib pick a color automatically"
        rather than as an error, so unknown/custom model names still work.
    """
    return MODEL_COLORS.get(model)


def model_linestyle(model: str) -> Optional[str]:
    """Return the paper's canonical linestyle for `model`.

    Args:
        model: A model name, e.g. "DynEdge".

    Returns:
        The matplotlib linestyle string used for this model in the paper,
        or None if `model` isn't one of the four paper models (see
        `model_color` for why this returns None rather than raising).
    """
    return MODEL_LINESTYLES.get(model)


def detector_display_name(detector: str) -> str:
    """Return the paper's display name for an internal detector key.

    Args:
        detector: The internal detector key used in NuBench file paths,
            e.g. "arca".

    Returns:
        The paper's display name, e.g. "Flower L". Unlike `model_color`/
        `model_linestyle`, this raises `KeyError` for an unrecognized
        detector rather than returning a fallback, since there's no
        sensible "unstyled" display name to fall back to - an unknown
        detector key is more likely a typo than a legitimately new one.
    """
    return DETECTOR_DISPLAY_NAMES[detector]


def detector_color(display_name: str) -> str:
    """Return the paper's canonical color for a detector's display name.

    Args:
        display_name: A detector's paper display name, e.g. "Flower L"
            (the output of `detector_display_name`, not the internal key).

    Returns:
        The matplotlib color string used for this detector in the paper.
    """
    return DETECTOR_COLORS[display_name]
