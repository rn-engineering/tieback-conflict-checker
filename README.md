# Tieback Conflict Checker

A Streamlit application for checking spatial conflicts between soldier pile tieback anchors.

## Features

- Editable and CSV/Excel-importable tieback data, with decimal inputs to 0.01; click **Apply edits** to save changes and refresh results
- Automatic sequential `TB-###` IDs for blank tieback IDs
- Downloadable CSV of the current tieback inputs for reuse
- Exact 3D line-segment nearest-distance calculations
- User-defined conflict and warning distances
- Plan-view visualization with interactive conflict markers
- Anchor geometry and conflict report tables
- Multi-sheet Excel export

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Data fields

- `Tieback ID`
- `X`: local anchor-head coordinate
- `Y`: local anchor-head coordinate
- `Start Elev`: anchor-head elevation
- `Direction`: horizontal direction clockwise from positive Y, in degrees
- `Drill Angle`: inclination below horizontal, in degrees
- `Length`: total anchor length
