# DXF trajectory and Wi-Fi AP extractor

This project combines two engineering sources:

- `input/geometry.dxf`: tunnel geometry containing `WALLS` and `TRAJECTORY` layers;
- `input/ap_plan.pdf`: the matching plan containing colored AP symbols and AP names.

It extracts the DXF trajectory, detects and names APs from the configured PDF
page, finds the PDF-to-DXF coordinate transformation from their shared vector
geometry, and writes CSV and PDF results.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python extract_trajectory_and_aps.py
```

To use another configuration:

```bash
.venv/bin/python extract_trajectory_and_aps.py --config config.json
```

## Inputs

Input files have stable names under `input/`:

- `geometry.dxf`
- `ap_plan.pdf`

The DXF parser reads `LINE` and `LWPOLYLINE` entities from the `WALLS` and
`TRAJECTORY` layers. The configured PDF page is zero-based; page index `14`
means page 15.

## Outputs

The script replaces these generated files on every successful run:

- `output/trajectory.csv`: DXF trajectory vertices and entity IDs;
- `output/ap_positions.csv`: AP name, PDF point/pixel position, transformed DXF
  position, detection metadata, and registration error;
- `output/combined_map.pdf`: zoomable DXF geometry with the trajectory in blue
  and named AP positions in orange/violet.

PDF pixel coordinates use `pdf.render_dpi` from `config.json`. PDF points always
use 72 units per inch. DXF coordinates preserve the coordinate units of the
input DXF.

## Registration

Registration uses black vector plan geometry from the PDF and wall vertices
from the DXF. It fits translation, rotation, uniform scale, and optional
reflection with a robust trimmed nearest-neighbour objective. This avoids the
incorrect bounding-box-only mapping used by the previous implementation.

The resulting median and 95th-percentile wall residuals are recorded in the AP
CSV and printed after each run. This is an automated engineering registration;
survey control points should still be used if certified coordinates are needed.

## Configuration

`config.json` controls input/output paths, PDF page and plan area, AP symbol
colors, clustering limits, and registration sampling. Paths are resolved
relative to the configuration file.
