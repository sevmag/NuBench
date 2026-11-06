# NuBench

This repository provides access to datasets, model predictions, and model artifacts for the NuBench dataset catalogue presented in [NuBench: An Open Benchmark for Deep Learning–Based Event Reconstruction in Neutrino Telescopes](https://arxiv.org/pdf/2511.13111). Datasets are available in two formats (SQLite and Parquet) - users may choose their preferred format.

---
![alt](./figures/geometry_birdseye.png "Birds Eye View of Geometries")
---
![alt](./figures/geometry.png "Side-view of Geometries")
---
The catalogue contains seven datasets simulated across six distinct detector geometries that are inspired by, but not strictly identical to, existing or proposed neutrino telescopes.

## Available Datasets
| Dataset | SQLite | Parquet | Predictions | Model Artifacts |
|---------|--------|---------|-------------|-----------------|
| Cluster | [Download](https://sid.erda.dk/share_redirect/EBamFwOU2D) | [Download](https://sid.erda.dk/share_redirect/C01vI5u9BJ) | [Download](https://sid.erda.dk/share_redirect/gZRbQvu0Ml) | [Download](https://sid.erda.dk/share_redirect/efqZPPTYBU) |
| Flower L | [Download](https://sid.erda.dk/share_redirect/EJylHQXkBr) | [Download](https://sid.erda.dk/share_redirect/HBIS5alCHj) | [Download](https://sid.erda.dk/share_redirect/E0uj9YnZVZ) | [Download](https://sid.erda.dk/share_redirect/fklOMcPIyB) |
| Flower S | [Download](https://sid.erda.dk/share_redirect/cUPqNKMRbQ) | [Download](https://sid.erda.dk/share_redirect/d08ELPHt2J) | [Download](https://sid.erda.dk/share_redirect/Ao3E9h8wLr) | [Download](https://sid.erda.dk/share_redirect/c5iAXJ5F8d) |
| Flower XL | [Download](https://sid.erda.dk/share_redirect/foVpx81yBz) | [Download](https://sid.erda.dk/share_redirect/EjHdSveUxc) | [Download](https://sid.erda.dk/share_redirect/AOjybKOo2Q) | [Download](https://sid.erda.dk/share_redirect/cR2WNVmk4l) |
| Hexagon | [Download](https://sid.erda.dk/share_redirect/GTf1gIlBbZ) | [Download](https://sid.erda.dk/share_redirect/GepbTY2MF4) | [Download](https://sid.erda.dk/share_redirect/A4hMrsbbl3) | [Download](https://sid.erda.dk/share_redirect/BLx7Z17wQe) |
| Hexagon Ice LE | [Download](https://sid.erda.dk/share_redirect/b9VHSF9X64) | [Download](https://sid.erda.dk/share_redirect/Cx2PVxHusa) | [Download](https://sid.erda.dk/share_redirect/hCIaEIm3sn) | [Download](https://sid.erda.dk/share_redirect/gckjyhPlul) |
| Triangle | [Download](https://sid.erda.dk/share_redirect/ER3B0TlPqR) | [Download](https://sid.erda.dk/share_redirect/ediHXAsygn) | [Download](https://sid.erda.dk/share_redirect/DZFXwMcQmP) | [Download](https://sid.erda.dk/share_redirect/ctNHDFuzPN) |


## File Types
- **SQLite/Parquet**: Dataset files in your preferred format. Both formats contain the same data, and are compatible with the Dataset classes in [GraphNeT](https://github.com/graphnet-team/graphnet), which you may reuse in your own experiments. We recommend the SQLite Dataset Class. Test/train partitions are defined in the selection files.
- **Predictions**: Model predictions on the test-partition of the dataset from [ParticleNet](https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/models/gnn/particlenet.py), [DynEdge](https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/models/gnn/dynedge.py), [GRIT](https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/models/gnn/grit.py), [DeepIce](https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/models/gnn/icemix.py).
- **Model Artifacts**: Trained model files and associated artifacts for reproducibility. This includes weights, pickled models and configuration files that allow you to re-create the models in [GraphNeT](https://github.com/graphnet-team/graphnet).

## Download & Extraction via Terminal
To download the files using your terminal, copy the download link from the table above and run e.g. :
`wget -P your_local_directory download_link`

All files end in `.tar.gz` but only some have been subject to compression.

Extract the contents of model artifacts and SQLite downloads using

`tar -xzf filename.tar.gz`

The contents of predictions and Parquet downloads should be extracted using

`tar -xf filename.tar.gz`

## Plotting
The `nubench` package reproduces the paper's figures from prediction files, for any combination of detectors, models and tasks.
### Install

```bash
pip install -e .
```

Run inside the repository folder. This also installs the `nubench-plot` command.

### Folder layout
`--data-root` is a folder containing one subfolder per detector, each holding
that detector's prediction files (`.parquet`):

```
<data-root>/
├── <folder containing the detector key>/
│   ├── <prediction files>.parquet
│   └── ...
└── ...
```

`--detector` selects a subfolder by a case-insensitive part of its name, and
must match exactly one subfolder. For example, with the NuBench prediction
downloads:

```
Dataset/                        <- --data-root Dataset
├── nubench_arca_dynedge/       <- --detector arca
└── nubench_orca_dynedge/       <- --detector orca
```

The paper's detector keys are labelled with their display names in
multi-panel figures:

| key | detector |
|---|---|
| `arca` | Flower L |
| `orca` | Flower S |
| `trident` | Flower XL |
| `icecube_water` | Hexagon |
| `icecube_ice` | Hexagon Ice LE |
| `gvd` | Cluster |
| `pone` | Triangle |

Within a folder, files are matched by model name (e.g. `GRIT_...`) and by
task (`initial_state_energy`, `direction`, `position`,
`visible_inelasticity`, `track`). Files without a model prefix are read as
DynEdge's, as in the NuBench downloads.

### Example

```bash
# Several models on one detector
nubench-plot --data-root Dataset --detector arca --feature energy \
    --model DynEdge --model GRIT --output energy.pdf

# One panel per detector
nubench-plot --data-root Dataset --detector arca --detector orca \
    --feature energy --model DynEdge --multi-panel --output energy_grid.pdf
```

Model names: `DynEdge`, `ParticleNeT`, `GRIT`, `DeepIce`, written exactly like this.

### Features and figures

`--feature` selects the task: `energy`, `direction`, `inelasticity`,
`vertex` or `classification`. Each draws its default figure.

Two tasks have a second figure, selected with `--figure` (only needed for
these):

- `--feature vertex --figure vertex-contour`
- `--feature classification --figure track-score`

A model can only be plotted for a detector that has prediction files for it.
In a single-detector figure, a missing model stops with an error. In a
multi-panel figure, it is skipped for that detector, with a note.

Run `nubench-plot --help` for all options.

## Citation

NuBench - https://arxiv.org/pdf/2511.13111
Prometheus - https://arxiv.org/pdf/2304.14526
GraphNeT - https://arxiv.org/abs/2501.03817
