"""Interactive Chemiscop visualization tool built with Streamlit.

This app links structure-property exploration with an interactive 3D
molecular viewer. It supports two data sources:
  * A directory with `.xyz` files (optionally paired with a CSV that
    contains a `filename` column and any number of additional property
    columns).
  * An ASE database file containing stored structures and attributes.

The UI is split into two panels:
  * Left: Plotly-based scatter plot (2D or 3D) for data exploration.
  * Right: py3Dmol viewer showing the 3D structure for the selected point.

Requirements (install via pip):
  streamlit, pandas, numpy, plotly, py3Dmol, streamlit-plotly-events, ase
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from difflib import SequenceMatcher
from ase import Atoms
from ase.db import connect
from ase.io import read as ase_read

try:
    import py3Dmol
except ImportError:  # pragma: no cover - runtime dependency
    py3Dmol = None

try:
    from streamlit_plotly_events import plotly_events
except ImportError:  # pragma: no cover - runtime dependency
    plotly_events = None


# Columns reserved for internal bookkeeping; anything else is treated as metadata.
BASE_COLUMNS = {
    "identifier",
    "label",
    "path",
    "source",
    "db_path",
    "db_id",
    "selection_id",
    "__index",
}


@dataclass
class ThemeConfig:
    """Lightweight holder for theme-specific styling options."""

    background: str
    plot_bg: str
    text_color: str
    highlight: str
    plot_template: str


THEMES: Dict[str, ThemeConfig] = {
    "Light": ThemeConfig(
        background="#FAFBFF",
        plot_bg="#F1F3F9",
        text_color="#0F172A",
        highlight="#2E86DE",
        plot_template="plotly_white",
    ),
    "Dark": ThemeConfig(
        background="#0E1117",
        plot_bg="#131722",
        text_color="#EBEBF5",
        highlight="#F39C12",
        plot_template="plotly_dark",
    ),
}


KEY_LISTENER_COMPONENT = components.declare_component(
    "chemiscop_key_listener",
    path=str(Path(__file__).parent / "components" / "key_listener"),
)


def inject_theme_css(theme: ThemeConfig) -> None:
    """Injects minimal CSS overrides for the chosen theme."""

    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {theme.background};
                color: {theme.text_color};
            }}
            .stApp div, .stApp span, .stApp label, .stApp textarea, .stApp p {{
                color: {theme.text_color} !important;
            }}
            .stApp .stMarkdown h1, .stApp .stMarkdown h2, .stApp .stMarkdown h3,
            .stApp .stMarkdown h4, .stApp .stMarkdown h5, .stApp .stMarkdown h6 {{
                color: {theme.text_color} !important;
            }}
            [data-testid="stSidebar"] {{
                background-color: {theme.plot_bg};
            }}
            [data-baseweb="input"] input {{
                color: {theme.text_color};
                background-color: {theme.background};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_xyz_metadata(dir_path: str, csv_path: Optional[str]) -> pd.DataFrame:
    """Return metadata for xyz files inside the given directory."""

    directory = Path(dir_path).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    xyz_files = sorted(p for p in directory.glob("*.xyz"))
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {directory}")

    properties: Dict[str, Dict[str, Any]] = {}
    if csv_path:
        csv_file = Path(csv_path).expanduser().resolve()
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        df_props = pd.read_csv(csv_file)
        if "filename" not in df_props.columns:
            raise ValueError("CSV file must contain a 'filename' column")
        properties = df_props.set_index("filename").to_dict("index")

    records = []
    for idx, xyz_file in enumerate(xyz_files):
        metadata = properties.get(xyz_file.name, {})
        record: Dict[str, Any] = {
            "identifier": xyz_file.name,
            "label": metadata.get("label", xyz_file.stem),
            "path": str(xyz_file),
            "source": "xyz",
            "selection_id": f"xyz::{xyz_file.name}",
            "__index": idx,
        }
        record.update(metadata)
        records.append(record)

    df = pd.DataFrame(records)
    return df.convert_dtypes()


@st.cache_data(show_spinner=False)
def load_ase_metadata(db_path: str) -> pd.DataFrame:
    """Return metadata for rows stored in an ASE database."""

    database = Path(db_path).expanduser().resolve()
    if not database.exists():
        raise FileNotFoundError(f"ASE database not found: {database}")

    records = []
    with connect(str(database)) as handle:
        for row in handle.select():
            properties = extract_row_properties(row)
            label = properties.get("label") or row.get("name") or row.formula
            record: Dict[str, Any] = {
                "identifier": str(row.id),
                "label": label or f"id_{row.id}",
                "db_path": str(database),
                "db_id": row.id,
                "source": "ase_db",
                "selection_id": f"ase::{row.id}",
                "__index": row.id,
            }
            record.update(properties)
            records.append(record)

    if not records:
        raise ValueError(f"No rows found in ASE database {database}")

    df = pd.DataFrame(records)
    return df.convert_dtypes()


def extract_row_properties(row: Any) -> Dict[str, Any]:
    """Collect scalar metadata from an ASE database row."""

    props: Dict[str, Any] = {}
    # Combine key/value pairs and data dictionaries.
    key_value_sources: Iterable[Dict[str, Any]] = []

    if hasattr(row, "key_value_pairs"):
        key_value_sources = [row.key_value_pairs]
    if hasattr(row, "data"):
        key_value_sources = [*key_value_sources, row.data]

    for source in key_value_sources:
        for key, value in source.items():
            if is_scalar(value):
                props[key] = value

    # Capture commonly used scalar attributes if available.
    for attribute in ("energy", "charge", "magmom"):
        if hasattr(row, attribute):
            value = getattr(row, attribute)
            if is_scalar(value):
                props.setdefault(attribute, value)

    return props


def is_scalar(value: Any) -> bool:
    """Return True for scalar values that can be plotted or displayed."""

    return isinstance(value, (int, float, str, bool, np.number))


@st.cache_resource(show_spinner=False)
def load_atoms_from_xyz(path: str) -> Atoms:
    """Load an ASE Atoms object from an xyz file."""

    return ase_read(path)


@st.cache_resource(show_spinner=False)
def load_atoms_from_ase(db_path: str, row_id: int) -> Atoms:
    """Load an ASE Atoms object from a row in an ASE database."""

    with connect(db_path) as handle:
        row = handle.get(id=row_id)
        return row.toatoms()


def load_atoms_raw(record: pd.Series) -> Atoms:
    """Return an ASE Atoms instance for the selected record without UI handling."""

    if record.source == "xyz":
        return load_atoms_from_xyz(record.path)
    if record.source == "ase_db":
        return load_atoms_from_ase(record.db_path, int(record.db_id))
    raise ValueError(f"Unsupported source type: {record.source}")


def get_atoms(record: pd.Series) -> Optional[Atoms]:
    """Return an ASE Atoms instance for the selected record."""

    try:
        return load_atoms_raw(record)
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to load structure for {record.label}: {exc}")
    return None


def atoms_to_xyz_block(atoms: Atoms, name: str) -> str:
    """Convert ASE atoms object to an XYZ formatted string."""

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    xyz_lines = [str(len(atoms)), name]
    xyz_lines.extend(
        f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
        for symbol, pos in zip(symbols, positions)
    )
    return "\n".join(xyz_lines)


# ------------------ NEW/CHANGED: interactive multi-mode renderer ------------------
def render_atoms_html(
    atoms: Atoms,
    label: str,
    *,
    theme: ThemeConfig,
    sphere_radius: float,
    bond_radius: float,
    show_axes: bool,
    axis_scale: float,
    height: int,
    width: Optional[int] = 700,
    label_modes: Optional[Iterable[str]] = None,
    interaction_mode: str = "rotate",  # NEW
) -> str:
    """Render the atoms to a py3Dmol HTML snippet with multi-mode interactions."""

    if py3Dmol is None:
        raise RuntimeError(
            "py3Dmol is required for 3D rendering. Install it with 'pip install py3Dmol'."
        )

    xyz_block = atoms_to_xyz_block(atoms, label)
    viewer = py3Dmol.view(width=max(width or 700, 200), height=height)
    viewer.addModel(xyz_block, "xyz")

    # FIX: apply style to ALL atoms via empty selection
    base_style = {
        "sphere": {"radius": max(float(sphere_radius), 0.05)},
        "stick": {"radius": max(float(bond_radius), 0.02)},
    }
    viewer.setStyle({}, base_style)  # CHANGED

    viewer.setBackgroundColor(theme.background)
    viewer.zoomTo()

    if show_axes:
        add_axes(viewer, atoms, axis_scale)

    if label_modes:
        add_atom_labels(viewer, atoms, label_modes, theme)

    # Build atom metadata for JS overlay/measurements
    symbols = atoms.get_chemical_symbols()
    numbers = atoms.get_atomic_numbers()
    masses = atoms.get_masses()
    coords = atoms.get_positions()
    atom_metadata = []
    for idx, (sym, Z, m, (x, y, z)) in enumerate(zip(symbols, numbers, masses, coords)):
        atom_metadata.append(
            {
                "serial": idx + 1,  # 3Dmol uses 1-based serials
                "index": idx,
                "symbol": sym,
                "atomic_number": int(Z),
                "mass": float(m),
                "x": float(x),
                "y": float(y),
                "z": float(z),
            }
        )

    # Map friendly mode to implementation
    mode_presets = {
        "rotate": {"label": "Rotate / navigate", "max_atoms": 0},
        "select": {"label": "Select atom", "max_atoms": 1},
        "measure": {"label": "Measure (2–3 atoms)", "max_atoms": 3},
        "dihedral": {"label": "Dihedral (4 atoms)", "max_atoms": 4},
    }
    mode_key = interaction_mode if interaction_mode in mode_presets else "rotate"
    mode_cfg = mode_presets[mode_key]

    # Pick overlay colors based on background brightness
    def _brightness(hex_color: str) -> float:
        if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
            return 1.0
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return (0.299*r + 0.587*g + 0.114*b)/255.0

    bright = _brightness(theme.background)
    overlay_bg = "rgba(15,23,42,0.78)" if bright >= 0.45 else "rgba(248,250,252,0.82)"
    overlay_text = "#F8FAFC" if bright >= 0.45 else "#0F172A"

    payload = {
        "mode": mode_key,
        "modeLabel": mode_cfg["label"],
        "maxAtoms": mode_cfg["max_atoms"],
        "baseStyle": base_style,
        "highlightDelta": max(0.12, float(sphere_radius) * 0.35),
        "highlightPalette": ["#FF4136", "#2ECC40", "#0074D9", "#B10DC9", "#FF851B"],
        "atoms": atom_metadata,
        "overlay": {"bg": overlay_bg, "fg": overlay_text},
    }

    base_html = viewer._make_html()

    # Inject a small JS controller for selection/measurement + overlay
    custom_js = f"""
<script>
(function() {{
  var cfg = {payload!r};
  var viewer = (function() {{
    try {{
      // py3Dmol names the variable like "viewer_1234" and initializes it globally.
      for (var k in window) {{
        if (k.startsWith('viewer_') && window[k] && typeof window[k].setStyle === 'function') {{
          return window[k];
        }}
      }}
    }} catch(e) {{}}
    return null;
  }})();
  if (!viewer) return;

  // Build quick lookup table
  var atomMap = {{}};
  (cfg.atoms||[]).forEach(function(a) {{ atomMap[a.serial] = a; }});

  // Overlay setup
  var container = document.querySelector('[id^="3dmolviewer_"]');
  if (!container) return;
  container.style.position = container.style.position || 'relative';

  var overlay = container.querySelector('.chemiscop-overlay');
  if (!overlay) {{
    overlay = document.createElement('div');
    overlay.className = 'chemiscop-overlay';
    overlay.style.position = 'absolute';
    overlay.style.left = '10px';
    overlay.style.right = '10px';
    overlay.style.bottom = '10px';
    overlay.style.pointerEvents = 'none';
    overlay.style.background = cfg.overlay.bg;
    overlay.style.color = cfg.overlay.fg;
    overlay.style.fontFamily = 'Inter, system-ui, -apple-system, sans-serif';
    overlay.style.fontSize = '12px';
    overlay.style.padding = '8px 12px';
    overlay.style.borderRadius = '10px';
    overlay.style.boxShadow = '0 8px 25px rgba(15,23,42,0.25)';
    overlay.style.maxHeight = '45%';
    overlay.style.overflowY = 'auto';
    overlay.style.lineHeight = '1.45';
    container.appendChild(overlay);
  }}

  function setOverlay(lines) {{
    overlay.innerHTML = (lines||[]).map(function(s) {{
      return '<div>' + s + '</div>';
    }}).join('');
  }}

  // Geometry helpers
  function dist(a,b) {{
    var dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
    return Math.sqrt(dx*dx+dy*dy+dz*dz);
  }}
  function sub(a,b) {{ return [a.x-b.x, a.y-b.y, a.z-b.z]; }}
  function norm(v) {{
    var n=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); return n>0?[v[0]/n,v[1]/n,v[2]/n]:[0,0,0];
  }}
  function dot(a,b) {{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }}
  function cross(a,b) {{
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  }}
  function angleDeg(a,b,c) {{
    var v1=norm(sub(a,b)), v2=norm(sub(c,b));
    var ca=Math.max(-1,Math.min(1,dot(v1,v2)));
    return Math.acos(ca)*180/Math.PI;
  }}
  function dihedralDeg(a,b,c,d) {{
    var b0=sub(b,a), b1=sub(c,b), b2=sub(d,c);
    var n1=norm(cross(b0,b1)), n2=norm(cross(b1,b2)), m1=cross(n1,norm(b1));
    var x=dot(n1,n2), y=dot(m1,n2);
    return Math.atan2(y,x)*180/Math.PI;
  }}

  // Selection + highlighting
  var selection = [];
  var maxAtoms = cfg.maxAtoms || 0;

  function clone(o) {{ return JSON.parse(JSON.stringify(o||{{}})); }}

  function applyHighlight() {{
    // reset
    viewer.setStyle({{}}, clone(cfg.baseStyle));
    // highlight chosen atoms
    var hpal = cfg.highlightPalette||[];
    selection.forEach(function(sel, i) {{
      var col = hpal[i % hpal.length] || '#FF4136';
      var s = {{ sphere: {{ radius: (cfg.baseStyle.sphere.radius + cfg.highlightDelta), color: col }} }};
      if (cfg.baseStyle.stick && cfg.baseStyle.stick.radius) {{
        s.stick = {{ radius: cfg.baseStyle.stick.radius, color: col }};
      }}
      viewer.addStyle({{serial: sel.serial}}, s);
    }});
    viewer.render();
  }}

  function summaryLine(atom, i) {{
    return '[' + (i+1) + '] ' + atom.symbol + ' (index ' + atom.index + ', Z=' + atom.atomic_number + ')';
  }}

  function updateOverlay() {{
    var lines = [];
    lines.push('<strong>Mode:</strong> ' + cfg.modeLabel);

    if (cfg.mode === 'select' && selection.length) {{
      var a = atomMap[selection[selection.length-1].serial];
      if (a) {{
        lines.push(summaryLine(a, 0));
        lines.push('Coordinates: (' + a.x.toFixed(3) + ', ' + a.y.toFixed(3) + ', ' + a.z.toFixed(3) + ') Å');
        if (!Number.isNaN(a.mass)) lines.push('Mass: ' + a.mass.toFixed(3) + ' amu');
      }}
    }} else if (cfg.mode === 'measure' && selection.length >= 2) {{
      var a1=atomMap[selection[selection.length-2].serial];
      var a2=atomMap[selection[selection.length-1].serial];
      if (a1 && a2) {{
        lines.push(summaryLine(a1, 0));
        lines.push(summaryLine(a2, 1));
        lines.push('d(1–2) = ' + dist(a1,a2).toFixed(3) + ' Å');
      }}
      if (selection.length >= 3) {{
        var a3=atomMap[selection[selection.length-3].serial];
        if (a1 && a2 && a3) {{
          lines.push('d(2–3) = ' + dist(a2,a3).toFixed(3) + ' Å');
          lines.push('d(1–3) = ' + dist(a1,a3).toFixed(3) + ' Å');
          lines.push('∠(1–2–3) = ' + angleDeg(a1,a2,a3).toFixed(2) + '°');
        }}
      }}
    }} else if (cfg.mode === 'dihedral' && selection.length >= 4) {{
      var a1=atomMap[selection[selection.length-4].serial];
      var a2=atomMap[selection[selection.length-3].serial];
      var a3=atomMap[selection[selection.length-2].serial];
      var a4=atomMap[selection[selection.length-1].serial];
      if (a1 && a2 && a3 && a4) {{
        lines.push(summaryLine(a1, 0));
        lines.push(summaryLine(a2, 1));
        lines.push(summaryLine(a3, 2));
        lines.push(summaryLine(a4, 3));
        lines.push('d(1–2) = ' + dist(a1,a2).toFixed(3) + ' Å');
        lines.push('d(2–3) = ' + dist(a2,a3).toFixed(3) + ' Å');
        lines.push('d(3–4) = ' + dist(a3,a4).toFixed(3) + ' Å');
        lines.push('∠(1–2–3–4) = ' + dihedralDeg(a1,a2,a3,a4).toFixed(2) + '°');
      }}
    }}

    if (cfg.mode !== 'rotate' && maxAtoms > 0 && selection.length < maxAtoms) {{
      var remain = maxAtoms - selection.length;
      lines.push('Select ' + remain + ' more atom' + (remain>1?'s':'') + ' …');
    }}

    setOverlay(lines);
  }}

  function toggle(serial) {{
    // remove if present, else push with max buffer
    var idx = selection.findIndex(function(s) {{ return s.serial === serial; }});
    if (idx >= 0) selection.splice(idx,1);
    else {{
      if (maxAtoms>0 && selection.length >= maxAtoms) selection.shift();
      selection.push({{serial: serial}});
    }}
    applyHighlight();
    updateOverlay();
  }}

  // Configure click behavior by mode
  if (cfg.mode === 'rotate' || maxAtoms === 0) {{
    if (viewer.setClickable) viewer.setClickable({{}}, false);
    selection = [];
    applyHighlight();
    updateOverlay();
    return;
  }}

  if (viewer.setClickable) {{
    viewer.setClickable({{}}, function(a) {{
      if (!a || a.serial == null) return;
      toggle(a.serial);
    }});
  }}
  applyHighlight();
  updateOverlay();
}})();
</script>
"""

    # Mark viewer frame for key-listener integration
    marker_script = """
<script>
(function() {
    try {
        var frame = window.frameElement;
        if (frame) { frame.dataset.chemiscopViewer = "1"; frame.tabIndex = 0; }
    } catch (e) { console.warn('Chemiscop viewer marker failed', e); }
})();
</script>
"""

    html = base_html + custom_js
    if "</body>" in html:
        html = html.replace("</body>", marker_script + "</body>")
    else:
        html = html + marker_script
    return html
# ------------------ end NEW/CHANGED renderer ------------------


def add_atom_labels(
    viewer: Any,
    atoms: Atoms,
    label_modes: Iterable[str],
    theme: ThemeConfig,
) -> None:
    """Annotate atoms with labels based on the selected display modes."""

    modes = {mode.replace(" ", "_").lower() for mode in label_modes}
    if not modes:
        return

    symbols = atoms.get_chemical_symbols()
    numbers = atoms.get_atomic_numbers()
    coords = atoms.get_positions()

    for idx, (symbol, atomic_number) in enumerate(zip(symbols, numbers)):
        parts: list[str] = []
        if "symbol" in modes:
            parts.append(symbol)
        if "atomic_number" in modes:
            parts.append(f"Z={int(atomic_number)}")
        if "atom_index" in modes:
            parts.append(f"i={idx}")
        if not parts:
            continue

        x, y, z = map(float, coords[idx])
        viewer.addLabel(
            " | ".join(parts),
            {
                "position": {"x": x, "y": y, "z": z},
                "fontSize": 14,
                "fontColor": theme.highlight,
                "backgroundColor": theme_transparent(theme.background),
                "backgroundOpacity": 0.6,
                "borderThickness": 0,
                "alignment": "center",
                "inFront": True,
            },
        )


def add_axes(viewer: Any, atoms: Atoms, scale: float) -> None:
    """Add RGB axes arrows to the viewer based on the molecule span."""

    positions = atoms.get_positions()
    span = float(np.ptp(positions, axis=0).max()) or 1.0
    axis_length = span * scale

    min_coords = positions.min(axis=0)
    offset = 0.1 * span
    origin = {
        "x": float(min_coords[0] - offset),
        "y": float(min_coords[1] - offset),
        "z": float(min_coords[2] - offset),
    }
    arrows = [
        ("X-axis", {"x": axis_length, "y": 0.0, "z": 0.0}, "#FF4136"),
        ("Y-axis", {"x": 0.0, "y": axis_length, "z": 0.0}, "#2ECC40"),
        ("Z-axis", {"x": 0.0, "y": 0.0, "z": axis_length}, "#0074D9"),
    ]
    for name, end, color in arrows:
        arrow_end = {
            "x": origin["x"] + end["x"],
            "y": origin["y"] + end["y"],
            "z": origin["z"] + end["z"],
        }
        viewer.addArrow(
            {
                "start": origin,
                "end": arrow_end,
                "color": color,
                "radius": span * 0.008,
            }
        )
        viewer.addLabel(
            name,
            {
                "fontColor": color,
                "backgroundColor": theme_transparent(color),
                "fontSize": 14,
                "position": arrow_end,
                "inFront": True,
            },
        )


def theme_transparent(color: str) -> str:
    """Return a transparent RGBA hex string based on a solid hex color."""

    if color.startswith("#") and len(color) == 7:
        return f"{color}66"
    return "#00000000"


def build_scatter_figure(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    z_axis: Optional[str],
    color_by: Optional[str],
    size_by: Optional[str],
    theme: ThemeConfig,
) -> Any:
    """Create a Plotly scatter or scatter_3d figure from the dataframe."""

    hover_data = [col for col in df.columns if col not in BASE_COLUMNS]
    custom_data = df[["selection_id"]]

    if z_axis:
        fig = px.scatter_3d(
            df,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color=color_by,
            size=size_by,
            hover_name="label",
            hover_data=hover_data,
            custom_data=custom_data,
            template=theme.plot_template,
        )
    else:
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            size=size_by,
            hover_name="label",
            hover_data=hover_data,
            custom_data=custom_data,
            template=theme.plot_template,
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=theme.plot_bg,
        paper_bgcolor=theme.background,
        height=600,
    )

    return fig


def arrow_key_listener() -> Optional[str]:
    """Listen for arrow key navigation events from the viewer."""

    payload = KEY_LISTENER_COMPONENT(default=None, key="chemiscop-key-listener")
    if isinstance(payload, dict) and "direction" in payload:
        direction = str(payload["direction"])
        timestamp = payload.get("timestamp")
        last_timestamp = st.session_state.get("_chemiscop_last_key_ts")
        if timestamp is None or timestamp != last_timestamp:
            st.session_state["_chemiscop_last_key_ts"] = timestamp
            return direction
    return None


def pick_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns available for plotting."""

    numeric_cols = []
    for column in df.columns:
        if column in BASE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_cols.append(column)
    return numeric_cols


def pick_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return non-numeric columns available for coloring/hovering."""

    cat_cols = []
    for column in df.columns:
        if column in BASE_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            cat_cols.append(column)
    return cat_cols


def sidebar_controls(df: pd.DataFrame, *, enable_scatter: bool) -> Dict[str, Any]:
    """Render sidebar controls and return the chosen configuration."""

    st.sidebar.header("Data")

    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    z_axis: Optional[str] = None
    color_by: Optional[str] = None
    size_by: Optional[str] = None

    if enable_scatter:
        scatter_mode = st.sidebar.radio("Scatter dimensionality", ["2D", "3D"], index=0)

        numeric_cols = pick_numeric_columns(df)
        categorical_cols = pick_categorical_columns(df)

        if not numeric_cols:
            numeric_cols = ["__index"]

        x_axis = st.sidebar.selectbox("X axis", numeric_cols, index=0)
        y_axis = st.sidebar.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1))
        if scatter_mode == "3D":
            z_axis = st.sidebar.selectbox("Z axis", numeric_cols, index=min(2, len(numeric_cols) - 1))

        color_options = ["None", *numeric_cols, *categorical_cols]
        color_choice = st.sidebar.selectbox("Color by", color_options, index=0)
        color_by = None if color_choice == "None" else color_choice

        size_options = ["Uniform", *numeric_cols]
        size_choice = st.sidebar.selectbox("Size by", size_options, index=0)
        size_by = None if size_choice == "Uniform" else size_choice
    else:
        st.sidebar.info("Add numeric properties to enable scatter plotting.")

    with st.sidebar.expander("3D Viewer", expanded=False):
        # -------- NEW/CHANGED: mouse mode picker --------
        mode_label_to_key = {
            "Rotate / navigate (default)": "rotate",
            "Select atom (show attributes)": "select",
            "Measure (2–3 atoms: d & ∠)": "measure",
            "Dihedral (4 atoms: torsion)": "dihedral",
        }
        viewer_mode_label = st.selectbox(
            "Mouse mode",
            list(mode_label_to_key.keys()),
            index=0,
            help="Choose how mouse clicks interact with the 3D viewer.",
        )
        viewer_mode = mode_label_to_key[viewer_mode_label]
        # ------------------------------------------------

        sphere_radius = st.slider("Atom radius", min_value=0.1, max_value=0.8, value=0.3, step=0.05)
        bond_radius = st.slider("Bond radius", min_value=0.05, max_value=0.4, value=0.12, step=0.01)
        show_axes = st.checkbox("Show axis reference", value=False)
        axis_scale = st.slider("Axis scale", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
        atom_labels = st.multiselect(
            "Atom labels",
            ["Symbol", "Atomic number", "Atom index"],
            default=[],
            help="Choose which annotations to display in the 3D viewer",
        )

    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "color_by": color_by,
        "size_by": size_by,
        "sphere_radius": sphere_radius,
        "bond_radius": bond_radius,
        "show_axes": show_axes,
        "axis_scale": axis_scale,
        "atom_labels": atom_labels,
        "viewer_mode": viewer_mode,  # NEW
    }


def plot_and_select(df: pd.DataFrame, fig: Any) -> Optional[str]:
    """Render the scatter plot and return the selected selection_id."""

    st.subheader("Data Exploration")

    selected_id: Optional[str] = st.session_state.get("selected_id")

    if plotly_events is None:
        st.warning(
            "Install `streamlit-plotly-events` to enable point selection by clicking. "
            "Using dropdown fallback for structure selection."
        )
        selection = st.selectbox(
            "Choose a structure",
            df["selection_id"],
            format_func=lambda sid: df.loc[df["selection_id"] == sid, "label"].iloc[0],
            index=df["selection_id"].tolist().index(selected_id)
            if selected_id in df["selection_id"].values
            else 0,
        )
        st.plotly_chart(fig, use_container_width=True)
        selected_id = selection
    else:
        events = plotly_events(fig, click_event=True, select_event=False, override_height=600, key="scatter")
        st.plotly_chart(fig, use_container_width=True)
        if events:
            event = events[0]
            candidate: Optional[str] = None
            if "customdata" in event and event["customdata"]:
                candidate = event["customdata"][0]
            elif "pointNumber" in event:
                try:
                    candidate = df["selection_id"].iloc[event["pointNumber"]]
                except Exception:
                    candidate = None
            if candidate and candidate in df["selection_id"].values:
                selected_id = candidate

    if selected_id is None and not df.empty:
        selected_id = df["selection_id"].iloc[0]

    st.session_state["selected_id"] = selected_id
    return selected_id


def summarize_atoms(atoms: Atoms) -> Dict[str, Any]:
    """Return quick summary statistics for the provided atoms object."""

    cell_volume = None
    if atoms.cell is not None and atoms.cell.volume > 0:
        cell_volume = float(atoms.cell.volume)

    return {
        "formula": atoms.get_chemical_formula(),
        "num_atoms": int(len(atoms)),
        "mass_amu": float(atoms.get_masses().sum()),
        "center_x": float(atoms.get_center_of_mass()[0]),
        "center_y": float(atoms.get_center_of_mass()[1]),
        "center_z": float(atoms.get_center_of_mass()[2]),
        "cell_volume": cell_volume,
    }


def _score_match(query: str, target: str) -> float:
    query = query.lower()
    target = target.lower()
    if query in target:
        return 1.0
    return SequenceMatcher(None, query, target).ratio()


def structure_search_panel(
    df: pd.DataFrame,
    selected_id: Optional[str],
    *,
    key_prefix: str,
    title: str = "Search by filename",
    suggestion_limit: int = 10,
    container: Optional[Any] = None,
    use_expander: bool = True,
    expanded: bool = False,
    suggestions_collapsible: bool = False,
    suggestions_expanded: Optional[bool] = None,
) -> Optional[str]:
    """Configurable panel that provides text search with live suggestions."""

    options = df[["selection_id", "identifier", "label"]].to_dict("records")
    if not options:
        return selected_id

    parent = container if container is not None else st.sidebar
    panel = (
        parent.expander(title, expanded=expanded)
        if use_expander
        else parent.container()
    )

    with panel:
        if not use_expander:
            st.markdown(f"**{title}**")

        query_key = f"{key_prefix}_query"
        query = st.text_input(
            "Type filename or label",
            value=st.session_state.get(query_key, ""),
            key=query_key,
            placeholder="e.g. molecule_10_.xyz",
        )

        suggestions: list[tuple[str, str]] = []

        if query:
            scored: list[tuple[float, Dict[str, Any]]] = []
            for entry in options:
                identifier = entry.get("identifier") or ""
                label = entry.get("label") or identifier
                score = _score_match(query, f"{identifier} {label}")
                if score > 0.2:
                    scored.append((score, entry))
            scored.sort(key=lambda item: item[0], reverse=True)
            suggestions = [
                (entry["selection_id"], entry.get("label") or entry.get("identifier", ""))
                for _, entry in scored[:suggestion_limit]
            ]
        else:
            suggestions = [
                (entry["selection_id"], entry.get("label") or entry.get("identifier", ""))
                for entry in options[:suggestion_limit]
            ]

        suggestion_parent: Any
        if suggestions_collapsible:
            expand_flag = suggestions_expanded
            if expand_flag is None:
                expand_flag = bool(query)
            suggestion_parent = st.expander(
                "Suggestions",
                expanded=expand_flag,
            )
        else:
            suggestion_parent = st.container()

        with suggestion_parent:
            if suggestions:
                st.caption("Suggestions:")
                for sid, label in suggestions:
                    button_label = label or sid
                    if st.button(button_label, key=f"{key_prefix}_suggest_{sid}"):
                        st.session_state["selected_id"] = sid
                        return sid
            elif not suggestions_collapsible:
                if query:
                    st.caption("No matches found.")
                else:
                    st.caption("No structures available for search.")

        if suggestions_collapsible and not suggestions:
            if query:
                st.caption("No matches found.")
            else:
                st.caption("No structures available for search.")

    return selected_id


def jump_to_structure(
    df: pd.DataFrame,
    selected_id: Optional[str],
    *,
    key: str,
    label: str = "Jump to structure",
) -> Optional[str]:
    """Render a searchable selectbox to jump directly to a structure."""

    options = df["selection_id"].tolist()
    if not options:
        return selected_id

    labels_map = df.set_index("selection_id")["label"].to_dict()
    default_idx = 0
    if selected_id in options:
        default_idx = options.index(selected_id)

    choice = st.sidebar.selectbox(
        label,
        options,
        index=default_idx,
        format_func=lambda sid: labels_map.get(sid, sid),
        key=key,
    )

    st.session_state["selected_id"] = choice
    return choice


@st.cache_data(show_spinner=True)
def compute_dataset_statistics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Compute summary statistics across the dataset.

    Returns:
        summary_df: Per-structure table with num_atoms and mass.
        elements_df: Aggregated counts of element types.
        failures: List of selection_ids that failed to load.
    """

    summary_rows: list[Dict[str, Any]] = []
    element_counter: Counter[str] = Counter()
    failures: list[str] = []

    for _, record in df.iterrows():
        try:
            atoms = load_atoms_raw(record)
        except Exception:
            failures.append(record["selection_id"])
            continue

        summary_rows.append(
            {
                "selection_id": record["selection_id"],
                "label": record["label"],
                "num_atoms": int(len(atoms)),
                "mass_amu": float(atoms.get_masses().sum()),
            }
        )
        element_counter.update(atoms.get_chemical_symbols())

    summary_df = pd.DataFrame(summary_rows)
    elements_df = pd.DataFrame(
        [
            {"element": elem, "count": count}
            for elem, count in element_counter.most_common()
        ]
    )

    return summary_df, elements_df, failures


def render_distribution_charts(
    summary_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    theme: ThemeConfig,
) -> None:
    """Display dataset-level distribution visualizations."""

    if summary_df.empty and elements_df.empty:
        st.info("Unable to compute distributions for the current dataset.")
        return

    options = {
        "Number of atoms": "num_atoms",
        "Total mass": "mass_amu",
        "Atom types": "element",
    }
    choice = st.selectbox("Distribution", list(options.keys()), index=0, key="distribution_choice")

    color_sequence = [theme.highlight]

    if choice == "Number of atoms":
        if summary_df.empty:
            st.info("Number-of-atoms statistics unavailable for this dataset.")
            return
        fig = px.histogram(
            summary_df,
            x="num_atoms",
            nbins=min(30, max(5, summary_df["num_atoms"].nunique())),
            template=theme.plot_template,
            color_discrete_sequence=color_sequence,
        )
        fig.update_layout(xaxis_title="Number of atoms", yaxis_title="Count")
    elif choice == "Total mass":
        if summary_df.empty:
            st.info("Mass statistics unavailable for this dataset.")
            return
        fig = px.histogram(
            summary_df,
            x="mass_amu",
            nbins=30,
            template=theme.plot_template,
            color_discrete_sequence=color_sequence,
        )
        fig.update_layout(xaxis_title="Total mass (amu)", yaxis_title="Count")
    else:
        if elements_df.empty:
            st.info("Element histogram unavailable for this dataset.")
            return
        fig = px.bar(
            elements_df,
            x="element",
            y="count",
            template=theme.plot_template,
            color="element",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(xaxis_title="Element", yaxis_title="Frequency")

    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor=theme.plot_bg, paper_bgcolor=theme.background)
    st.plotly_chart(fig, use_container_width=True)


def show_details(record: pd.Series, atoms: Atoms) -> None:
    """Display metadata and structure summary for the selected record."""

    st.markdown(f"**Selected structure:** `{record.label}`")

    basic_info = summarize_atoms(atoms)
    basic_df = pd.DataFrame([basic_info])
    st.markdown("**Basic information**")
    st.dataframe(basic_df, use_container_width=True)

    metadata = {k: v for k, v in record.items() if k not in BASE_COLUMNS}
    if metadata:
        st.markdown("**Metadata**")
        st.dataframe(pd.DataFrame([metadata]), use_container_width=True)


def navigation_controls(
    df: pd.DataFrame,
    selected_id: Optional[str],
    *,
    search_key_prefix: str,
) -> Optional[str]:
    """Render navigation controls with inline structure search."""

    options = df["selection_id"].tolist()
    if not options:
        return selected_id

    if selected_id not in options:
        selected_id = options[0]

    current_idx = options.index(selected_id)

    prev_col, center_col, next_col = st.columns([1, 3, 1])

    prev_clicked = prev_col.button(
        "◀ Previous", use_container_width=True, key="nav_prev"
    )
    prev_col.caption("← key while hovering")

    next_clicked = next_col.button(
        "Next ▶", use_container_width=True, key="nav_next"
    )
    next_col.caption("→ key while hovering")

    new_idx = current_idx
    if prev_clicked:
        new_idx = (current_idx - 1) % len(options)
    elif next_clicked:
        new_idx = (current_idx + 1) % len(options)

    new_id = options[new_idx]

    with center_col:
        new_id = structure_search_panel(
            df,
            new_id,
            key_prefix=search_key_prefix,
            title="Search structures",
            container=center_col,
            use_expander=False,
            suggestions_collapsible=True,
        )
        new_idx = options.index(new_id)

        direction = arrow_key_listener()
        if direction == "prev":
            new_idx = (new_idx - 1) % len(options)
        elif direction == "next":
            new_idx = (new_idx + 1) % len(options)
        new_id = options[new_idx]

        current_label = df.loc[df["selection_id"] == new_id, "label"].iloc[0]
        st.markdown(f"**{current_label} ({new_idx + 1}/{len(options)})**")
        st.caption("Hover over the 3D view and use arrow keys.")

    st.session_state["selected_id"] = new_id
    return new_id


def main() -> None:
    st.set_page_config(page_title="Chemiscop", layout="wide")

    theme_name = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=1)
    theme = THEMES[theme_name]
    inject_theme_css(theme)

    st.title("Chemiscop: Structure-Property Explorer")

    st.sidebar.header("Data Source")
    source_type = st.sidebar.radio("Source", ["XYZ Directory", "ASE Database"], index=0)

    try:
        if source_type == "XYZ Directory":
            xyz_dir = st.sidebar.text_input(
                "XYZ directory", value="visualizer/test_xyz"
            )
            csv_path = st.sidebar.text_input("Optional CSV with properties")
            if not xyz_dir:
                st.info("Provide a directory containing .xyz files to begin.")
                return
            df = load_xyz_metadata(xyz_dir, csv_path or None)
        else:
            db_path = st.sidebar.text_input("ASE database path")
            if not db_path:
                st.info("Provide a path to an ASE database (.db) file to begin.")
                return
            df = load_ase_metadata(db_path)
    except Exception as exc:
        st.error(str(exc))
        return

    if df.empty:
        st.warning("No records found with the provided inputs.")
        return

    has_numeric_properties = bool(pick_numeric_columns(df))
    config = sidebar_controls(df, enable_scatter=has_numeric_properties)

    if "__index" not in df.columns:
        df["__index"] = np.arange(len(df))

    selected_id: Optional[str]

    nav_search_prefix = "scatter_search" if has_numeric_properties else "list_search"

    if has_numeric_properties:
        fig = build_scatter_figure(
            df,
            x_axis=config["x_axis"],
            y_axis=config["y_axis"],
            z_axis=config["z_axis"],
            color_by=config["color_by"],
            size_by=config["size_by"],
            theme=theme,
        )

        left_col, right_col = st.columns((1, 1))

        with left_col:
            selected_id = plot_and_select(df, fig)

        st.sidebar.divider()
        selected_id = jump_to_structure(
            df,
            selected_id,
            key="jump_select_scatter",
            label="Jump directly to a structure",
        )

        viewer_container = right_col
    else:
        st.sidebar.header("Structure Selection")
        selected_id = jump_to_structure(
            df,
            st.session_state.get("selected_id"),
            key="jump_select_list",
            label="Choose a structure",
        )
        viewer_container = st.container()

    if selected_id is None:
        st.info("Select a structure to view its 3D geometry.")
        return

    with viewer_container:
        selected_id = navigation_controls(
            df,
            selected_id,
            search_key_prefix=nav_search_prefix,
        )
        record = df.loc[df["selection_id"] == selected_id].iloc[0]
        atoms = get_atoms(record)
        if atoms is None:
            st.error("Unable to load atoms for the selected entry.")
            return
        try:
            html = render_atoms_html(
                atoms,
                record.label,
                theme=theme,
                sphere_radius=config["sphere_radius"],
                bond_radius=config["bond_radius"],
                show_axes=config["show_axes"],
                axis_scale=config["axis_scale"],
                height=600,
                width=700,
                label_modes=config["atom_labels"],
                interaction_mode=config["viewer_mode"],  # NEW
            )
            st.components.v1.html(
                html,
                height=600,
                width=700,
            )
        except Exception as exc:
            st.error(str(exc))

        show_details(record, atoms)

    with st.expander("Dataset distributions", expanded=False):
        with st.spinner("Computing dataset statistics..."):
            summary_df, elements_df, failures = compute_dataset_statistics(df)
        render_distribution_charts(summary_df, elements_df, theme)
        if failures:
            st.caption(
                "Skipped structures without available geometries: "
                + ", ".join(failures)
            )


if __name__ == "__main__":
    main()
