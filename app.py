import hashlib
from itertools import combinations
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Tieback Conflict Checker", layout="wide")

INPUT_COLUMNS = [
    "Tieback ID", "X", "Y", "Start Elev",
    "Direction", "Drill Angle", "Length",
]
NUMERIC_COLUMNS = ["X", "Y", "Start Elev", "Direction", "Drill Angle", "Length"]

SAMPLE_DATA = pd.DataFrame([
    ["TB-001", 0.0, 0.0, 100.0, 45.0, 15.0, 80.0],
    ["TB-002", 10.0, 0.0, 100.0, 315.0, 15.0, 80.0],
], columns=INPUT_COLUMNS)


def normalize_input(data):
    data = data.copy()
    for column in INPUT_COLUMNS:
        if column not in data.columns:
            data[column] = "" if column == "Tieback ID" else np.nan

    data = data[INPUT_COLUMNS]
    data["Tieback ID"] = data["Tieback ID"].fillna("").astype(str).str.strip()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    has_content = data["Tieback ID"].ne("") | data[NUMERIC_COLUMNS].notna().any(axis=1)
    return data.loc[has_content].reset_index(drop=True)


def assign_automatic_ids(data):
    result = data.copy()
    used = set(result.loc[result["Tieback ID"].ne(""), "Tieback ID"])
    next_number = 1

    for index in result.index:
        if not result.at[index, "Tieback ID"]:
            while f"TB-{next_number:03d}" in used:
                next_number += 1
            tieback_id = f"TB-{next_number:03d}"
            result.at[index, "Tieback ID"] = tieback_id
            used.add(tieback_id)
            next_number += 1
    return result


def calculate_endpoints(data):
    result = data.copy()
    direction = np.radians(result["Direction"])
    drill_angle = np.radians(result["Drill Angle"])
    horizontal_length = result["Length"] * np.cos(drill_angle)

    result["End X"] = result["X"] + horizontal_length * np.sin(direction)
    result["End Y"] = result["Y"] + horizontal_length * np.cos(direction)
    result["End Elev"] = result["Start Elev"] - result["Length"] * np.sin(drill_angle)
    return result


def closest_points_on_segments(p0, p1, q0, q1):
    """Return exact closest points for two 3D line segments."""
    p0, p1, q0, q1 = map(lambda p: np.asarray(p, dtype=float), (p0, p1, q0, q1))
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    epsilon = 1e-12
    candidates = []

    def add_candidate(s, t):
        s = float(np.clip(s, 0.0, 1.0))
        t = float(np.clip(t, 0.0, 1.0))
        delta = w + s * u - t * v
        candidates.append((float(np.dot(delta, delta)), s, t))

    denominator = a * c - b * b
    if denominator > epsilon:
        s = (b * e - c * d) / denominator
        t = (a * e - b * d) / denominator
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            add_candidate(s, t)

    for s in (0.0, 1.0):
        t = (e + b * s) / c if c > epsilon else 0.0
        add_candidate(s, t)

    for t in (0.0, 1.0):
        s = (b * t - d) / a if a > epsilon else 0.0
        add_candidate(s, t)

    for s in (0.0, 1.0):
        for t in (0.0, 1.0):
            add_candidate(s, t)

    _, s_best, t_best = min(candidates, key=lambda item: item[0])
    point_a = p0 + s_best * u
    point_b = q0 + t_best * v
    distance = float(np.linalg.norm(point_a - point_b))
    return point_a, point_b, distance, s_best, t_best


def find_conflicts(geometry, conflict_limit, warning_limit):
    results = []
    for index_a, index_b in combinations(geometry.index, 2):
        row_a = geometry.loc[index_a]
        row_b = geometry.loc[index_b]

        point_a, point_b, distance, fraction_a, fraction_b = closest_points_on_segments(
            [row_a["X"], row_a["Y"], row_a["Start Elev"]],
            [row_a["End X"], row_a["End Y"], row_a["End Elev"]],
            [row_b["X"], row_b["Y"], row_b["Start Elev"]],
            [row_b["End X"], row_b["End Y"], row_b["End Elev"]],
        )

        if distance <= warning_limit:
            results.append({
                "Tieback A": row_a["Tieback ID"],
                "Tieback B": row_b["Tieback ID"],
                "Min Distance": distance,
                "Status": "Conflict" if distance <= conflict_limit else "Warning",
                "A Distance Along": fraction_a * row_a["Length"],
                "B Distance Along": fraction_b * row_b["Length"],
                "A Closest X": point_a[0], "A Closest Y": point_a[1], "A Closest Elev": point_a[2],
                "B Closest X": point_b[0], "B Closest Y": point_b[1], "B Closest Elev": point_b[2],
                "Marker X": (point_a[0] + point_b[0]) / 2,
                "Marker Y": (point_a[1] + point_b[1]) / 2,
                "Marker Elev": (point_a[2] + point_b[2]) / 2,
            })
    return pd.DataFrame(results)


def make_plan_figure(geometry, conflicts):
    fig = go.Figure()
    conflict_ids, warning_ids = set(), set()

    if not conflicts.empty:
        conflict_rows = conflicts[conflicts["Status"] == "Conflict"]
        warning_rows = conflicts[conflicts["Status"] == "Warning"]
        conflict_ids.update(conflict_rows["Tieback A"])
        conflict_ids.update(conflict_rows["Tieback B"])
        warning_ids.update(warning_rows["Tieback A"])
        warning_ids.update(warning_rows["Tieback B"])
        warning_ids -= conflict_ids

    for _, row in geometry.iterrows():
        tieback_id = row["Tieback ID"]
        color = "#d62728" if tieback_id in conflict_ids else (
            "#e6a700" if tieback_id in warning_ids else "#2ca02c"
        )
        fig.add_trace(go.Scatter(
            x=[row["X"], row["End X"]], y=[row["Y"], row["End Y"]],
            mode="lines+markers", line={"color": color, "width": 2},
            marker={"size": [8, 5]}, name=tieback_id,
            hovertemplate=f"<b>{tieback_id}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
        ))
        fig.add_annotation(x=row["X"], y=row["Y"], text=tieback_id,
                           showarrow=False, xshift=8, yshift=8)

    if not conflicts.empty:
        marker_colors = ["#d62728" if status == "Conflict" else "#e6a700"
                         for status in conflicts["Status"]]
        hover_text = [
            f"<b>{row['Status']}: {row['Tieback A']} / {row['Tieback B']}</b><br>"
            f"Minimum distance: {row['Min Distance']:.2f} ft<br>"
            f"Midpoint X: {row['Marker X']:.2f} ft<br>"
            f"Midpoint Y: {row['Marker Y']:.2f} ft<br>"
            f"Midpoint elevation: {row['Marker Elev']:.2f} ft"
            for _, row in conflicts.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=conflicts["Marker X"], y=conflicts["Marker Y"],
            mode="markers", name="Conflict markers",
            marker={"color": marker_colors, "size": 14, "symbol": "x",
                    "line": {"width": 2}},
            text=hover_text,
            customdata=list(range(len(conflicts))),
            hovertemplate="%{text}<extra>Click marker for details</extra>",
        ))

    fig.update_layout(
        title="Tieback Plan View", xaxis_title="Local X (ft)", yaxis_title="Local Y (ft)",
        yaxis={"scaleanchor": "x", "scaleratio": 1}, hovermode="closest",
        legend_title="Tieback", height=650,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    return fig


st.title("Tieback Conflict Checker")
st.caption("Direction is clockwise from +Y; drill angle is below horizontal.")

with st.sidebar:
    st.header("Conflict Criteria")
    conflict_limit = st.number_input("Conflict distance (ft)", min_value=0.0, value=2.0, step=0.5)
    warning_limit = st.number_input("Warning distance (ft)", min_value=0.0, value=5.0, step=0.5)
    st.caption("Pairs at or below the conflict limit are conflicts; pairs between limits are warnings.")
    st.divider()
    uploaded_file = st.file_uploader("Import tiebacks (CSV or Excel)", type=["csv", "xlsx", "xls"])

if warning_limit < conflict_limit:
    st.error("Warning distance must be greater than or equal to conflict distance.")
    st.stop()

input_data = SAMPLE_DATA.copy()
if uploaded_file is not None:
    try:
        input_data = (pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv")
                      else pd.read_excel(uploaded_file))
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

input_data = normalize_input(input_data)

input_source = (
    hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if uploaded_file is not None else "sample"
)
if st.session_state.get("tieback_input_source") != input_source:
    st.session_state["tieback_input_source"] = input_source
    st.session_state["tieback_input_rows"] = assign_automatic_ids(input_data)
    st.session_state["tieback_editor_version"] = (
        st.session_state.get("tieback_editor_version", 0) + 1
    )

st.subheader("Tieback Inputs")
st.caption("Blank Tieback IDs are assigned the next available TB-### label.")
editor_key = f"tieback_editor_{st.session_state['tieback_editor_version']}"
with st.form("tieback_inputs_form"):
    edited_data = st.data_editor(
        st.session_state["tieback_input_rows"],
        num_rows="dynamic", hide_index=True, use_container_width=True,
        column_config={
            "Tieback ID": st.column_config.TextColumn("Tieback ID"),
            "Direction": st.column_config.NumberColumn("Direction (°)", step=0.01, format="%.2f"),
            "Drill Angle": st.column_config.NumberColumn("Drill Angle (°)", step=0.01, format="%.2f"),
            "Length": st.column_config.NumberColumn("Length (ft)", step=0.01, format="%.2f"),
            "Start Elev": st.column_config.NumberColumn("Start Elev (ft)", step=0.01, format="%.2f"),
            "X": st.column_config.NumberColumn("X (ft)", step=0.01, format="%.2f"),
            "Y": st.column_config.NumberColumn("Y (ft)", step=0.01, format="%.2f"),
        }, key=editor_key,
    )
    apply_edits = st.form_submit_button("Apply edits")

st.caption("Apply edits to save your changes and refresh the conflict results.")
edited_data = normalize_input(edited_data)
if apply_edits:
    assigned_data = assign_automatic_ids(edited_data)
    st.session_state["tieback_input_rows"] = assigned_data
    st.session_state["tieback_editor_version"] += 1
    st.rerun()

edited_data = st.session_state["tieback_input_rows"]
assigned_data = assign_automatic_ids(edited_data)
input_csv = edited_data.to_csv(index=False)
st.download_button(
    "Save tieback inputs (CSV)", data=input_csv,
    file_name="tieback_inputs.csv", mime="text/csv",
)

geometry_input = assigned_data
if geometry_input.empty:
    st.info("Add tieback data in the table or import a CSV/Excel file.")
    st.stop()

if geometry_input[INPUT_COLUMNS].isna().any(axis=1).any():
    st.error("Complete all numeric inputs for each tieback.")
    st.stop()

if geometry_input["Tieback ID"].duplicated().any():
    duplicates = geometry_input.loc[geometry_input["Tieback ID"].duplicated(False), "Tieback ID"].unique()
    st.error("Tieback IDs must be unique. Duplicate IDs: " + ", ".join(duplicates))
    st.stop()

if (geometry_input["Length"] <= 0).any():
    st.error("Tieback lengths must be greater than zero.")
    st.stop()

if not geometry_input["Drill Angle"].between(0.0, 90.0).all():
    st.error("Drill angles must be between 0° and 90°.")
    st.stop()

geometry = calculate_endpoints(geometry_input)
conflicts = find_conflicts(geometry, conflict_limit, warning_limit)

st.subheader("Plan View")
chart_event = st.plotly_chart(
    make_plan_figure(geometry, conflicts), use_container_width=True,
    on_select="rerun", selection_mode="points", key="plan_view",
)

try:
    selected_points = chart_event.selection.points
except AttributeError:
    selected_points = []

if selected_points:
    custom_data = selected_points[0].get("customdata")
    if isinstance(custom_data, list) and len(custom_data) >= 1:
        selected_index = int(custom_data[0])
    else:
        selected_index = int(selected_points[0].get("pointIndex", -1))

    if 0 <= selected_index < len(conflicts):
        selected = conflicts.iloc[selected_index]
        st.markdown(
            f"**Selected: {selected['Tieback A']} / {selected['Tieback B']} — "
            f"{selected['Status']} ({selected['Min Distance']:.2f} ft apart)**"
        )
        left, right = st.columns(2)
        left.write(
            f"{selected['Tieback A']} closest point: X {selected['A Closest X']:.2f} ft, "
            f"Y {selected['A Closest Y']:.2f} ft, elevation {selected['A Closest Elev']:.2f} ft; "
            f"{selected['A Distance Along']:.2f} ft along anchor."
        )
        right.write(
            f"{selected['Tieback B']} closest point: X {selected['B Closest X']:.2f} ft, "
            f"Y {selected['B Closest Y']:.2f} ft, elevation {selected['B Closest Elev']:.2f} ft; "
            f"{selected['B Distance Along']:.2f} ft along anchor."
        )

st.subheader("Calculated Tieback Geometry")
st.dataframe(
    geometry[INPUT_COLUMNS + ["End X", "End Y", "End Elev"]].round(3),
    hide_index=True, use_container_width=True,
)

st.subheader("Conflict Report")
if conflicts.empty:
    st.success("No tieback pairs are within the warning distance.")
else:
    report_columns = [
        "Tieback A", "Tieback B", "Min Distance", "Status",
        "A Distance Along", "B Distance Along", "A Closest X", "A Closest Y", "A Closest Elev",
        "B Closest X", "B Closest Y", "B Closest Elev",
    ]
    report = conflicts[report_columns]

    def style_closest_elevations(row):
        styles = [""] * len(row)
        a_elev = row["A Closest Elev"]
        b_elev = row["B Closest Elev"]
        if a_elev > b_elev:
            styles[row.index.get_loc("A Closest Elev")] = "background-color: #b7e1cd; color: #14532d"
            styles[row.index.get_loc("B Closest Elev")] = "background-color: #f4cccc; color: #7f1d1d"
        elif b_elev > a_elev:
            styles[row.index.get_loc("A Closest Elev")] = "background-color: #f4cccc; color: #7f1d1d"
            styles[row.index.get_loc("B Closest Elev")] = "background-color: #b7e1cd; color: #14532d"
        return styles

    report_styler = report.style.apply(style_closest_elevations, axis=1).format(precision=3)
    st.dataframe(report_styler, hide_index=True, use_container_width=True)

output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    geometry[INPUT_COLUMNS + ["End X", "End Y", "End Elev"]].to_excel(
        writer, sheet_name="Tieback Geometry", index=False
    )
    conflicts.to_excel(writer, sheet_name="Conflict Report", index=False)
    pd.DataFrame({
        "Setting": ["Conflict distance (ft)", "Warning distance (ft)"],
        "Value": [conflict_limit, warning_limit],
    }).to_excel(writer, sheet_name="Settings", index=False)

st.download_button(
    "Download Excel report", data=output.getvalue(),
    file_name="tieback_conflict_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
