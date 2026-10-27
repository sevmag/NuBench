"""The paper's colors, linestyles, and display names."""

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
    """The paper's color for `model`, or None if it isn't a paper model.

    Callers pass None straight to matplotlib, which then picks a color
    itself - so custom model names still plot fine.
    """
    return MODEL_COLORS.get(model)


def model_linestyle(model: str) -> Optional[str]:
    """The paper's linestyle for `model`, None if unknown (see
    `model_color`).
    """
    return MODEL_LINESTYLES.get(model)


def detector_display_name(detector: str) -> str:
    """The paper's display name for a detector key: "arca" -> "Flower L".

    Raises KeyError for an unknown key rather than falling back, since
    there is no sensible default display name and a bad key is more
    likely a typo than a new detector.
    """
    return DETECTOR_DISPLAY_NAMES[detector]


def detector_color(display_name: str) -> str:
    """The paper's color for a detector's *display* name ("Flower L")."""
    return DETECTOR_COLORS[display_name]
