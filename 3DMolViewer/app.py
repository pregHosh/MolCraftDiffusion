"""3DMolViewer-style Streamlit app that renders molecules with NGL."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Tuple

from io import StringIO
from uuid import uuid4

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import streamlit as st

from ase import Atoms
from ase.db import connect
from ase.io import read as ase_read
from ase.io import write as ase_write

try:  # Optional dependency for scatter picking
    from streamlit_plotly_events import plotly_events  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency
    plotly_events = None

try:  # Optional dependency for keyboard shortcuts (legacy API)
    from st_hotkeys import st_hotkeys as legacy_st_hotkeys  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency
    legacy_st_hotkeys = None

try:  # Optional dependency for keyboard shortcuts (modern API)
    import streamlit_hotkeys as hotkeys  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency
    hotkeys = None


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_XYZ_DIR = str(PACKAGE_ROOT / "test_xyz")


BASE_COLUMNS = {
    "identifier",
    "label",
    "path",
    "source",
    "db_path",
    "db_id",
    "selection_id",
    "__index",
    "has_geometry",
}


SNAPSHOT_QUALITY_OPTIONS: List[Tuple[str, int]] = [
    ("Standard (1x)", 1),
    ("High (2x)", 2),
    ("Ultra (4x)", 4),
]


def sanitize_filename(label: str, suffix: str = ".png") -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label.strip())
    safe = safe or "snapshot"
    return f"{safe}{suffix}"


@dataclass
class ThemeConfig:
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


def inject_theme_css(theme: ThemeConfig) -> None:
    extra_light_css = ""
    if theme.background.lower() == "#fafbff":
        extra_light_css = """
            /* Light theme input overrides */
            .stApp input[type='text'],
            .stApp input[type='number'],
            .stApp input[type='search'],
            .stApp textarea,
            .stApp .stTextInput > div > div,
            .stApp .stNumberInput > div > div,
            .stApp .stSelectbox > div > div,
            .stApp .stMultiSelect > div > div,
            .stApp [data-baseweb='select'] > div,
            .stApp [data-baseweb='textarea'] > div {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
                border: 1px solid rgba(15, 23, 42, 0.18) !important;
                box-shadow: none !important;
            }
            .stApp [data-baseweb='select'] div[role='button'] {
                color: #0F172A !important;
            }
            .stApp .stDownloadButton button,
            .stApp .stDownloadButton button:hover,
            .stApp .stButton button,
            .stApp .stButton button:focus {
                color: #0F172A !important;
            }
            .stApp label, .stApp .stMarkdown p {
                color: #0F172A !important;
            }
        """

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
            [data-testid="stSidebar"] {{
                background-color: {theme.plot_bg};
            }}
            /* Sticky top navbar */
            .molviewer-navbar {{
                position: sticky;
                top: 0;
                z-index: 999;
                background: {theme.background};
                padding: 8px 12px;
                border-radius: 10px;
                border: 1px solid rgba(15,23,42,0.08);
                box-shadow: 0 6px 18px rgba(15,23,42,0.08);
                margin-bottom: 12px;
            }}
            .molviewer-navbar .small-label {{
                font-size: 12px;
                opacity: 0.7;
                margin-bottom: 4px;
                display: block;
            }}
            {extra_light_css}
        </style>
        """,
        unsafe_allow_html=True,
    )



@st.cache_data(show_spinner=False)
def load_xyz_metadata(dir_path: str, csv_path: Optional[str]) -> pd.DataFrame:
    directory = Path(dir_path).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    xyz_files = sorted(p for p in directory.glob("*.xyz"))
    filtered_xyz_missing_props = 0
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {directory}")

    properties: Dict[str, Dict[str, Any]] = {}
    csv_only_payload: Dict[str, Dict[str, Any]] = {}
    csv_ignored = False
    csv_metadata_path: Optional[str] = None
    csv_only_names: list[str] = []
    matched_names: set[str] = set()
    xyz_map = {p.name: p for p in xyz_files}
    if csv_path:
        csv_file = Path(csv_path).expanduser().resolve()
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        df_props = pd.read_csv(csv_file)
        if "filename" not in df_props.columns:
            raise ValueError("CSV file must contain a 'filename' column")
        df_props = df_props.copy()
        df_props["filename"] = df_props["filename"].astype(str)
        value_columns = [col for col in df_props.columns if col != "filename"]
        if value_columns:
            df_props = df_props.dropna(subset=value_columns, how="all")
        csv_metadata_path = str(csv_file)
        csv_records: Dict[str, Dict[str, Any]] = {}
        for _, row in df_props.iterrows():
            fname = str(row["filename"])
            payload: Dict[str, Any] = {}
            for col in value_columns:
                value = row[col]
                if pd.isna(value):
                    continue
                payload[col] = value
            csv_records[fname] = payload
        csv_names = set(csv_records.keys())
        xyz_names = set(xyz_map.keys())
        matched_names = csv_names & xyz_names
        csv_only_names = sorted(csv_names - xyz_names)
        filtered_xyz_missing_props = len(xyz_names - matched_names)
        if matched_names:
            properties = {name: csv_records[name] for name in matched_names}
            xyz_files = [p for p in xyz_files if p.name in matched_names]
        else:
            xyz_files = []
        csv_only_payload = {name: csv_records[name] for name in csv_only_names}
        csv_ignored = not matched_names and not csv_only_names

    records = []
    for xyz_file in xyz_files:
        metadata = properties.get(xyz_file.name, {})
        record: Dict[str, Any] = {
            "identifier": xyz_file.name,
            "label": metadata.get("label", xyz_file.stem),
            "path": str(xyz_file),
            "source": "xyz",
            "selection_id": f"xyz::{xyz_file.name}",
            "has_geometry": True,
        }
        record.update(metadata)
        record["__index"] = len(records)
        records.append(record)

    for name in csv_only_names:
        metadata = csv_only_payload.get(name, {})
        record = {
            "identifier": name,
            "label": metadata.get("label", Path(name).stem),
            "path": None,
            "source": "csv_only",
            "selection_id": f"csv::{name}",
            "has_geometry": False,
        }
        record.update(metadata)
        record["__index"] = len(records)
        records.append(record)

    df = pd.DataFrame(records).convert_dtypes()
    if csv_metadata_path:
        df.attrs["csv_properties_path"] = csv_metadata_path
    df.attrs["csv_properties_ignored"] = csv_ignored
    if filtered_xyz_missing_props:
        df.attrs["csv_xyz_filtered"] = filtered_xyz_missing_props
    if csv_only_names:
        df.attrs["csv_only_count"] = len(csv_only_names)
        df.attrs["csv_only_names"] = csv_only_names
    return df


@st.cache_data(show_spinner=False)
def load_ase_metadata(db_path: str) -> pd.DataFrame:
    database = Path(db_path).expanduser().resolve()
    if not database.exists():
        raise FileNotFoundError(f"ASE database not found: {database}")

    records = []
    with connect(str(database)) as handle:
        for row in handle.select():
            props = extract_row_properties(row)
            label = props.get("label") or row.get("name") or row.formula
            records.append(
                {
                    "identifier": str(row.id),
                    "label": label or f"id_{row.id}",
                    "db_path": str(database),
                    "db_id": row.id,
                    "source": "ase_db",
                    "selection_id": f"ase::{row.id}",
                    "__index": row.id,
                    **props,
                }
            )

    if not records:
        raise ValueError(f"No rows found in ASE database {database}")

    return pd.DataFrame(records).convert_dtypes()


def extract_row_properties(row: Any) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    if hasattr(row, "key_value_pairs"):
        props.update({k: v for k, v in row.key_value_pairs.items() if is_scalar(v)})
    if hasattr(row, "data"):
        props.update({k: v for k, v in row.data.items() if is_scalar(v)})
    for attr in ("energy", "charge", "magmom"):
        if hasattr(row, attr):
            value = getattr(row, attr)
            if is_scalar(value):
                props.setdefault(attr, value)
    return props


def is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool, np.number))


@st.cache_resource(show_spinner=False)
def load_atoms_from_xyz(path: str) -> Atoms:
    return ase_read(path)


@st.cache_resource(show_spinner=False)
def load_atoms_from_ase(db_path: str, row_id: int) -> Atoms:
    with connect(db_path) as handle:
        return handle.get(id=row_id).toatoms()


def load_atoms_raw(record: pd.Series) -> Atoms:
    if record.source == "xyz":
        return load_atoms_from_xyz(record.path)
    if record.source == "ase_db":
        return load_atoms_from_ase(record.db_path, int(record.db_id))
    raise ValueError(f"Unsupported source type: {record.source}")


def get_atoms(record: pd.Series) -> Optional[Atoms]:
    try:
        return load_atoms_raw(record)
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to load structure for {record.label}: {exc}")
    return None


def filter_hydrogens(atoms: Atoms, *, show_hydrogens: bool) -> Atoms:
    if show_hydrogens:
        return atoms
    symbols = atoms.get_chemical_symbols()
    keep_indices = [idx for idx, symbol in enumerate(symbols) if symbol != "H"]
    if len(keep_indices) == len(symbols):
        return atoms
    if not keep_indices:
        return Atoms()
    return atoms[keep_indices]


def atoms_to_pdb_block(atoms: Atoms) -> str:
    centered = atoms.copy()
    try:
        center = centered.get_center_of_mass()
    except Exception:  # fallback if masses undefined
        center = centered.get_positions().mean(axis=0)
    centered.translate(-center)
    buffer = StringIO()
    ase_write(buffer, centered, format="proteindatabank")
    return buffer.getvalue()


def render_ngl_view(
    atoms: Atoms,
    label: str,
    *,
    theme: ThemeConfig,
    sphere_radius: float,
    bond_radius: float,
    interaction_mode: str,
    height: int,
    width: int = 700,
    representation_style: str = "Ball + Stick",
    label_mode: Optional[str] = None,
    snapshot: Optional[Dict[str, Any]] = None,
    element_id: Optional[str] = None,
) -> str:
    pdb_block = atoms_to_pdb_block(atoms)
    metadata = [
        {
            "index": idx,
            "serial": idx + 1,
            "symbol": sym,
            "atomic_number": int(num),
            "mass": float(mass),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
        }
        for idx, (sym, num, mass, pos) in enumerate(
            zip(
                atoms.get_chemical_symbols(),
                atoms.get_atomic_numbers(),
                atoms.get_masses(),
                atoms.get_positions(),
            )
        )
    ]

    default_quality_options = [
        {"label": label, "value": str(factor)} for label, factor in SNAPSHOT_QUALITY_OPTIONS
    ]
    snapshot_cfg: Dict[str, Any] = {
        "transparent": False,
        "factor": 1.0,
        "antialias": True,
        "trim": True,
        "filename": sanitize_filename(label),
        "background": theme.background,
        "qualityOptions": default_quality_options,
    }
    if snapshot:
        for key, value in snapshot.items():
            if value is None:
                continue
            if key == "quality_options":
                snapshot_cfg["qualityOptions"] = value
                continue
            snapshot_cfg[key] = value
    try:
        snapshot_cfg["factor"] = float(snapshot_cfg.get("factor", 1.0))
    except (TypeError, ValueError):
        snapshot_cfg["factor"] = 1.0

    mode_presets = {
        "rotate": {"label": "Rotate / navigate", "maxAtoms": 0},
        "select": {"label": "Select atom", "maxAtoms": 1},
        "measurement": {"label": "Measurement", "maxAtoms": 4},
    }
    mode_key = interaction_mode if interaction_mode in mode_presets else "rotate"
    mode_cfg = mode_presets[mode_key]

    sphere_radius = max(float(sphere_radius), 0.1)
    bond_radius = max(float(bond_radius), 0.02)
    aspect_ratio = max(sphere_radius / bond_radius if bond_radius else 1.0, 1.0)
    highlight_ratio = max((sphere_radius + 0.2) / bond_radius if bond_radius else aspect_ratio, 1.0)

    label_modes = []
    if label_mode:
        label_modes.append(label_mode.lower().replace(" ", "_"))

    style_map = {
        "Ball + Stick": "ball+stick",
        "Licorice": "licorice",
        "Spacefilling": "spacefill",
        "Hyperball": "hyperball",
        "Line": "line",
        "Point Cloud": "point",
        "Surface": "surface",
    }
    style_key = style_map.get(representation_style, "ball+stick")

    style_params: Dict[str, Any] = {"colorScheme": "element"}
    if style_key == "ball+stick":
        style_params.update(
            {
                "multipleBond": "symmetric",
                "sphereDetail": 2,
                "radiusType": "size",
                "radiusSize": bond_radius,
                "aspectRatio": aspect_ratio,
            }
        )
    elif style_key == "licorice":
        style_params.update(
            {
                "multipleBond": "symmetric",
                "radiusType": "size",
                "radiusSize": bond_radius,
            }
        )
    elif style_key == "spacefill":
        style_params.update(
            {
                "radiusType": "size",
                "radiusSize": sphere_radius,
            }
        )
    elif style_key == "hyperball":
        style_params.update(
            {
                "multipleBond": "symmetric",
                "radius": max(bond_radius, 0.05),
                "shrink": 0.2,
            }
        )
    elif style_key == "line":
        style_params.update(
            {
                "linewidth": max(1, int(round(bond_radius * 30))),
                "multipleBond": "off",
            }
        )
    elif style_key == "point":
        style_params.update(
            {
                "pointSize": max(1.0, sphere_radius * 20.0),
                "alpha": 1.0,
            }
        )
    elif style_key == "surface":
        style_params.update(
            {
                "opacity": 0.85,
                "surfaceType": "msms",
                "probeRadius": 1.4,
                "contour": False,
            }
        )

    payload = {
        "pdb": pdb_block,
        "atoms": metadata,
        "mode": mode_key,
        "modeLabel": mode_cfg["label"],
        "maxAtoms": mode_cfg["maxAtoms"],
        "sphereRadius": sphere_radius,
        "bondRadius": bond_radius,
        "aspectRatio": aspect_ratio,
        "highlightAspectRatio": highlight_ratio,
        "labelModes": label_modes,
        "theme": {
            "background": theme.background,
            "text": theme.text_color,
            "highlight": theme.highlight,
        },
        "palette": ["#FF4136", "#2ECC40", "#0074D9", "#B10DC9", "#FF851B"],
        "style": style_key,
        "styleParams": style_params,
        "zoomStep": 0.18,
        "snapshot": snapshot_cfg,
    }

    quality_options_html = "".join(
        f"<option value=\"{opt['value']}\"{' selected' if str(snapshot_cfg['factor']) == str(opt['value']) else ''}>{opt['label']}</option>"
        for opt in snapshot_cfg.get("qualityOptions", [])
    )
    transparent_checked = "checked" if snapshot_cfg.get("transparent") else ""

    html = f"""
<div style=\"display:flex;justify-content:center;\">
  <div id=\"molviewer-stage-wrapper\" style=\"display:flex;flex-direction:column;align-items:stretch;width:{width}px;gap:12px;\">
    <div id=\"ngl-stage\" style=\"width:{width}px;height:{height}px;position:relative;background:{theme.background};border-radius:12px;overflow:hidden;\">
      <button id=\"molviewer-snapshot-download-float\" type=\"button\" title=\"Download PNG\" style=\"position:absolute;top:12px;left:12px;padding:6px 12px;border-radius:6px;border:0;background:{theme.highlight};color:#FFFFFF;font-weight:600;cursor:pointer;box-shadow:0 4px 12px rgba(15,23,42,0.18);z-index:25;pointer-events:auto;\">Save PNG</button>
    </div>
    <div id=\"molviewer-snapshot-bar\" style=\"display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;padding:8px 12px;border:1px solid rgba(15,23,42,0.12);border-radius:10px;background:{theme.plot_bg};box-shadow:0 4px 16px rgba(15,23,42,0.12);\">
      <div style=\"display:flex;flex-wrap:wrap;align-items:center;gap:12px;\">
        <label style=\"display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;\">
          <input id=\"molviewer-snapshot-transparent\" type=\"checkbox\" style=\"margin:0;\" {transparent_checked}>
          Transparent background
        </label>
        <label style=\"display:flex;align-items:center;gap:6px;font-size:13px;\">
          Quality
          <select id=\"molviewer-snapshot-quality\" style=\"padding:4px 8px;border-radius:6px;border:1px solid rgba(15,23,42,0.18);background:#FFFFFF;min-width:140px;\">
            {quality_options_html}
          </select>
        </label>
      </div>
      <div style=\"display:flex;align-items:center;gap:10px;\">
        <button id=\"molviewer-snapshot-download\" type=\"button\" style=\"padding:8px 16px;border-radius:6px;border:0;background:{theme.highlight};color:#FFFFFF;font-weight:600;cursor:pointer;box-shadow:0 4px 12px rgba(15,23,42,0.18);\">Download PNG</button>
      </div>
    </div>
    <div id=\"molviewer-snapshot-status\" style=\"font-size:12px;color:{theme.text_color};min-height:14px;\"></div>
  </div>
</div>
<script src=\"https://unpkg.com/ngl@2.0.0-dev.39/dist/ngl.js\"></script>
<script>
(function() {{
  var cfg = {json.dumps(payload)};
  var stage = new NGL.Stage('ngl-stage', {{ backgroundColor: cfg.theme.background }});
  window.addEventListener('resize', function() {{ stage.handleResize(); }}, false);

  var selection = [];
  var highlightReprs = [];
  var measurementReprs = [];

  function clearReprs(list) {{
    (list || []).forEach(function(repr) {{ try {{ repr.dispose(); }} catch (err) {{ }} }});
    list.length = 0;
  }}

  function ensureZoomControls() {{
    var node = document.getElementById('ngl-stage');
    if (!node) return;
    var controls = node.querySelector('.molviewer-zoom-controls');
    if (controls) return controls;

    controls = document.createElement('div');
    controls.className = 'molviewer-zoom-controls';
    Object.assign(controls.style, {{
      position: 'absolute', top: '12px', right: '12px', display: 'flex', gap: '6px',
      pointerEvents: 'auto', zIndex: 20
    }});

    function makeButton(label, title, onClick) {{
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = label;
      btn.title = title;
      Object.assign(btn.style, {{
        width: '34px', height: '34px', borderRadius: '6px', border: '1px solid rgba(15,23,42,0.2)',
        background: cfg.theme.background, color: cfg.theme.text, cursor: 'pointer',
        fontSize: '18px', lineHeight: '32px', padding: '0', boxShadow: '0 4px 12px rgba(15,23,42,0.12)'
      }});
      btn.addEventListener('click', function(ev) {{
        ev.preventDefault();
        ev.stopPropagation();
        onClick();
      }});
      btn.addEventListener('mouseenter', function() {{
        btn.style.background = cfg.theme.highlight;
        btn.style.color = cfg.theme.background;
      }});
      btn.addEventListener('mouseleave', function() {{
        btn.style.background = cfg.theme.background;
        btn.style.color = cfg.theme.text;
      }});
      return btn;
    }}

    var step = Math.max(Math.min(cfg.zoomStep || 0.18, 0.5), 0.02);
    var zoomIn = makeButton('+', 'Zoom in', function() {{
      try {{ stage.viewerControls.zoom(step); }} catch (err) {{ console.warn('Zoom in failed', err); }}
    }});
    var zoomOut = makeButton('\u2212', 'Zoom out', function() {{
      try {{ stage.viewerControls.zoom(-step); }} catch (err) {{ console.warn('Zoom out failed', err); }}
    }});
    var reset = makeButton('\u21BB', 'Reset view', function() {{
      try {{ stage.autoView(); }} catch (err) {{ console.warn('Reset view failed', err); }}
    }});

    controls.appendChild(zoomIn);
    controls.appendChild(zoomOut);
    controls.appendChild(reset);
    node.appendChild(controls);

    return controls;
  }}

  function enableGestureZoom() {{
    if (!stage || !stage.viewer) return;

    if (stage.viewer.container && stage.viewer.container.style.touchAction !== 'none') {{
      stage.viewer.container.style.touchAction = 'none';
    }}

    if (stage.mouseControls && stage.trackballControls) {{
      var hasScrollAction = stage.mouseControls.actionList && stage.mouseControls.actionList.some(function(action) {{
        return action && action.type === 'scroll';
      }});
      if (!hasScrollAction) {{
        stage.mouseControls.add('scroll', function(stageRef, delta) {{
          stageRef.trackballControls.zoom(delta);
        }});
      }}
    }}

    var canvas = stage.viewer && stage.viewer.renderer ? stage.viewer.renderer.domElement : null;
    if (!canvas || canvas.dataset.molviewerPinchBound) return;
    canvas.dataset.molviewerPinchBound = '1';

    var pinchState = null;
    function pinchDistance(touches) {{
      if (!touches || touches.length < 2) return 0;
      var dx = touches[0].pageX - touches[1].pageX;
      var dy = touches[0].pageY - touches[1].pageY;
      return Math.sqrt(dx * dx + dy * dy);
    }}

    canvas.addEventListener('touchstart', function(event) {{
      if (event.touches && event.touches.length === 2) {{
        pinchState = {{
          start: pinchDistance(event.touches) || 0,
          camera: stage.viewerControls ? stage.viewerControls.getCameraDistance() : stage.viewer ? stage.viewer.cameraDistance : 0
        }};
      }}
    }}, {{ passive: true }});

    canvas.addEventListener('touchmove', function(event) {{
      if (!pinchState || !event.touches || event.touches.length !== 2) return;
      event.preventDefault();
      var distance = pinchDistance(event.touches);
      if (!distance || !pinchState.start) return;
      var ratio = pinchState.start / distance;
      var target = pinchState.camera * ratio;
      if (stage.viewerControls && typeof stage.viewerControls.distance === 'function') {{
        stage.viewerControls.distance(target);
      }} else if (stage.viewer) {{
        stage.viewer.cameraDistance = Math.max(Math.abs(target), 0.2);
        stage.viewer.updateZoom();
      }}
    }}, {{ passive: false }});

    canvas.addEventListener('touchend', function(event) {{
      if (!event.touches || event.touches.length < 2) {{
        pinchState = null;
      }}
    }}, {{ passive: true }});
  }}

  var transparentInput = document.getElementById('molviewer-snapshot-transparent');
  var qualitySelect = document.getElementById('molviewer-snapshot-quality');
  var downloadButtons = [];
  var snapshotStatusNode = document.getElementById('molviewer-snapshot-status');

  function collectDownloadButtons() {{
    var buttons = [];
    var mainButton = document.getElementById('molviewer-snapshot-download');
    var floatingButton = document.getElementById('molviewer-snapshot-download-float');
    if (mainButton) buttons.push(mainButton);
    if (floatingButton) buttons.push(floatingButton);
    return buttons;
  }}

  function setDownloadButtonsDisabled(disabled) {{
    downloadButtons.forEach(function(btn) {{
      btn.disabled = !!disabled;
    }});
  }}

  function setSnapshotStatus(message, success) {{
    if (!snapshotStatusNode) {{
      snapshotStatusNode = document.getElementById('molviewer-snapshot-status');
    }}
    if (!snapshotStatusNode) return;
    snapshotStatusNode.textContent = message || '';
    if (!message) {{
      snapshotStatusNode.style.color = cfg.theme.text;
      return;
    }}
    snapshotStatusNode.style.color = success ? '#047857' : cfg.highlight;
  }}

  function triggerDownloadUrl(url, filename) {{
    var link = document.createElement('a');
    link.href = url;
    link.download = filename || 'snapshot.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }}

  function handleSnapshotRequest(event) {{
    if (event) {{
      event.preventDefault();
      event.stopPropagation();
    }}
    if (!stage) return;

    if (!transparentInput) transparentInput = document.getElementById('molviewer-snapshot-transparent');
    if (!qualitySelect) qualitySelect = document.getElementById('molviewer-snapshot-quality');
    if (!downloadButtons.length) downloadButtons = collectDownloadButtons();
    if (!downloadButtons.length) return;

    var transparent = transparentInput ? !!transparentInput.checked : false;
    var factorValue = qualitySelect ? parseFloat(qualitySelect.value || '1') : 1;
    if (!(factorValue > 0)) factorValue = 1;
    var antialias = cfg.snapshot && cfg.snapshot.antialias !== undefined ? !!cfg.snapshot.antialias : true;
    var trim = cfg.snapshot && cfg.snapshot.trim !== undefined ? !!cfg.snapshot.trim : true;
    var background = (!transparent && cfg.snapshot && cfg.snapshot.background) ? cfg.snapshot.background : cfg.theme.background;
    var filename = (cfg.snapshot && cfg.snapshot.filename) || 'snapshot.png';
    setSnapshotStatus('Rendering image...', false);
    setDownloadButtonsDisabled(true);

    stage.makeImage({{
      factor: factorValue,
      antialias: antialias,
      trim: trim,
      transparent: transparent,
      renderer: 'color',
      backgroundColor: transparent ? 'rgba(0,0,0,0)' : background
    }}).then(function(image) {{
      var completed = false;
      function finish(success, message) {{
        if (completed) return;
        completed = true;
        setDownloadButtonsDisabled(false);
        if (success) {{
          setSnapshotStatus('PNG downloaded', true);
        }} else {{
          setSnapshotStatus(message || 'Snapshot failed.', false);
        }}
      }}

      if (image instanceof Blob) {{
        var blobUrl = URL.createObjectURL(image);
        triggerDownloadUrl(blobUrl, filename);
        setTimeout(function() {{ URL.revokeObjectURL(blobUrl); }}, 4000);
        finish(true);
        return;
      }}

      try {{
        if (image && typeof image.download === 'function') {{
          image.download(filename);
          finish(true);
          return;
        }}
      }} catch (err) {{
        console.warn('Snapshot helper download failed', err);
      }}

      var blobCandidate = image ? (typeof image.blob === 'function' ? image.blob() : image.blob) : null;
      if (blobCandidate instanceof Blob) {{
        var url = URL.createObjectURL(blobCandidate);
        triggerDownloadUrl(url, filename);
        setTimeout(function() {{ URL.revokeObjectURL(url); }}, 4000);
        finish(true);
        return;
      }}

      var canvas = image ? (typeof image.getCanvas === 'function' ? image.getCanvas() : image.canvas) : null;
      if (canvas && canvas.toBlob) {{
        canvas.toBlob(function(blob) {{
          if (blob) {{
            var url = URL.createObjectURL(blob);
            triggerDownloadUrl(url, filename);
            setTimeout(function() {{ URL.revokeObjectURL(url); }}, 4000);
            finish(true);
          }} else {{
            finish(false, 'Snapshot unavailable.');
          }}
        }});
        return;
      }}

      var dataUrl = image ? (typeof image.getDataURL === 'function' ? image.getDataURL() : image.dataURL) : null;
      if (typeof dataUrl === 'string') {{
        triggerDownloadUrl(dataUrl, filename);
        finish(true);
        return;
      }}

      finish(false, 'Snapshot unavailable.');
    }}).catch(function(err) {{
      console.error('Snapshot failed', err);
      setDownloadButtonsDisabled(false);
      setSnapshotStatus('Snapshot failed. Check console for details.', false);
    }});
  }}

  function setupSnapshotControls() {{
    if (!transparentInput) transparentInput = document.getElementById('molviewer-snapshot-transparent');
    if (!qualitySelect) qualitySelect = document.getElementById('molviewer-snapshot-quality');
    var buttons = collectDownloadButtons();
    if (!buttons.length) return;
    downloadButtons = buttons;
    buttons.forEach(function(btn) {{
      if (btn.hasAttribute('data-bound')) return;
      btn.setAttribute('data-bound', '1');
      btn.addEventListener('click', handleSnapshotRequest);
    }});
  }}

  function overlayContainer() {{
    var node = document.getElementById('ngl-stage');
    if (!node) return null;
    var overlay = node.querySelector('.molviewer-overlay');
    if (overlay) return overlay;
    overlay = document.createElement('div');
    overlay.className = 'molviewer-overlay';
    Object.assign(overlay.style, {{
      position: 'absolute', left: '10px', right: '10px', bottom: '10px', pointerEvents: 'none',
      background: 'rgba(15,23,42,0.78)', color: '#F8FAFC', fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
      fontSize: '12px', padding: '8px 12px', borderRadius: '8px', boxShadow: '0 8px 25px rgba(15,23,42,0.2)',
      maxHeight: '45%', overflowY: 'auto', lineHeight: '1.45'
    }});
    node.appendChild(overlay);
    return overlay;
  }}

  function setOverlay(lines) {{
    var overlay = overlayContainer();
    if (!overlay) return;
    overlay.style.background = cfg.theme.background.toLowerCase() === '#0e1117' ? 'rgba(248,250,252,0.82)' : 'rgba(15,23,42,0.78)';
    overlay.style.color = cfg.theme.background.toLowerCase() === '#0e1117' ? '#0F172A' : '#F8FAFC';
    overlay.innerHTML = (lines || []).map(function(line) {{ return '<div>' + line + '</div>'; }}).join('');
  }}

  var blob = new Blob([cfg.pdb], {{ type: 'text/plain' }});
  stage.loadFile(blob, {{ ext: 'pdb' }}).then(function(component) {{
    var styleName = cfg.style || 'ball+stick';
    var baseParams = Object.assign({{}}, cfg.styleParams || {{}});
    component.addRepresentation(styleName, baseParams);
    stage.autoView();

    var structure = component.structure;

    ensureZoomControls();
    enableGestureZoom();
    setupSnapshotControls();

    if (Array.isArray(cfg.labelModes) && cfg.labelModes.length) {{
      var labelText = {{}};
      var hasLabels = false;
      cfg.atoms.forEach(function(atom) {{
        var parts = [];
        if (cfg.labelModes.indexOf('symbol') !== -1) parts.push(atom.symbol);
        if (cfg.labelModes.indexOf('atomic_number') !== -1) parts.push('Z=' + atom.atomic_number);
        if (cfg.labelModes.indexOf('atom_index') !== -1) parts.push('#' + atom.index);
        if (parts.length) {{
          labelText[atom.index] = parts.join(' \u00b7 ');
          hasLabels = true;
        }}
      }});
      if (hasLabels) {{
        var labelBackground = cfg.theme.background.toLowerCase() === '#0e1117' ? '#F8FAFC' : '#0F172A';
        component.addRepresentation('label', {{
          labelType: 'text',
          labelText: labelText,
          labelGrouping: 'atom',
          attachment: 'middle-center',
          color: cfg.theme.text,
          showBackground: true,
          backgroundColor: labelBackground,
          backgroundOpacity: 0.35,
          fixedSize: true,
          zOffset: 2
        }});
      }}
    }}

    function refreshHighlights() {{
      clearReprs(highlightReprs);
      selection.forEach(function(sel, idx) {{
        var repr = component.addRepresentation('ball+stick', {{
          sele: '@' + sel.index,
          color: cfg.palette[idx % cfg.palette.length],
          radiusType: 'size',
          radiusSize: cfg.bondRadius,
          aspectRatio: cfg.highlightAspectRatio
        }});
        highlightReprs.push(repr);
      }});
    }}

    function addDistancePairs(pairs, color) {{
      if (!pairs.length) return;
      var repr = component.addRepresentation('distance', {{
        atomPair: pairs.map(function(pair) {{
          return pair.map(function(idx) {{ return structure.getAtomProxy(idx); }});
        }}),
        labelVisible: true,
        color: color || cfg.palette[0]
      }});
      measurementReprs.push(repr);
    }}

    function measurementKind() {{
      if (cfg.mode !== 'measurement') return null;
      var count = selection.length;
      if (count >= 4) return 'dihedral';
      if (count === 3) return 'angle';
      if (count === 2) return 'distance';
      return null;
    }}

    function refreshMeasurements() {{
      clearReprs(measurementReprs);
      if (cfg.mode !== 'measurement') return;
      var kind = measurementKind();
      if (kind === 'distance') {{
        var ids = selection.slice(-2).map(function(sel) {{ return sel.index; }});
        addDistancePairs([[ids[0], ids[1]]], cfg.palette[0]);
      }} else if (kind === 'angle') {{
        var ida = selection.slice(-3).map(function(sel) {{ return sel.index; }});
        addDistancePairs([[ida[0], ida[1]], [ida[1], ida[2]]], cfg.palette[0]);
      }} else if (kind === 'dihedral') {{
        var idd = selection.slice(-4).map(function(sel) {{ return sel.index; }});
        addDistancePairs([[idd[0], idd[1]], [idd[1], idd[2]], [idd[2], idd[3]]], cfg.palette[0]);
      }}
    }}

    function clearSelection() {{
      selection.length = 0;
      refreshHighlights();
      refreshMeasurements();
      summaryLines();
    }}

    function summaryLines() {{
      var lines = [];
      lines.push('<strong>Mode:</strong> ' + cfg.modeLabel);
      if (cfg.mode === 'select' && selection.length) {{
        var atom = selection[selection.length-1];
        lines.push('[' + atom.serial + '] ' + atom.symbol + ' (index ' + atom.index + ', Z=' + atom.atomic_number + ')');
        lines.push('Coordinates: (' + atom.x.toFixed(3) + ', ' + atom.y.toFixed(3) + ', ' + atom.z.toFixed(3) + ') Angstrom');
        lines.push('Mass: ' + atom.mass.toFixed(3) + ' amu');
      }} else if (cfg.mode === 'measurement') {{
        var kind = measurementKind();
        if (kind === 'distance') {{
          var a = selection[selection.length-2];
          var b = selection[selection.length-1];
          var dist = Math.sqrt(Math.pow(a.x-b.x,2)+Math.pow(a.y-b.y,2)+Math.pow(a.z-b.z,2));
          lines.push('Distance = ' + dist.toFixed(3) + ' Angstrom');
        }} else if (kind === 'angle') {{
          var p = selection.slice(-3);
          var v1 = [p[0].x-p[1].x, p[0].y-p[1].y, p[0].z-p[1].z];
          var v2 = [p[2].x-p[1].x, p[2].y-p[1].y, p[2].z-p[1].z];
          var dot = v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
          var n1 = Math.sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]);
          var n2 = Math.sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2]);
          var ang = Math.acos(Math.max(-1, Math.min(1, dot/(n1*n2)))) * 180/Math.PI;
          lines.push('Angle = ' + ang.toFixed(2) + ' deg');
        }} else if (kind === 'dihedral') {{
          var q = selection.slice(-4);
          var subtract = function(a,b) {{ return [a.x-b.x, a.y-b.y, a.z-b.z]; }};
          var cross = function(a,b) {{ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }};
          var norm = function(v) {{ var n=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); return n? [v[0]/n,v[1]/n,v[2]/n]:[0,0,0]; }};
          var b0 = subtract(q[1], q[0]);
          var b1 = subtract(q[2], q[1]);
          var b2 = subtract(q[3], q[2]);
          var n1 = norm(cross(b0, b1));
          var n2 = norm(cross(b1, b2));
          var m1 = cross(n1, norm(b1));
          var x = n1[0]*n2[0]+n1[1]*n2[1]+n1[2]*n2[2];
          var y = m1[0]*n2[0]+m1[1]*n2[1]+m1[2]*n2[2];
          var dih = Math.atan2(y, x) * 180/Math.PI;
          lines.push('Dihedral = ' + dih.toFixed(2) + ' deg');
        }}
      }}

      if (cfg.mode !== 'rotate' && cfg.maxAtoms > 0 && selection.length < cfg.maxAtoms) {{
        var remaining = cfg.maxAtoms - selection.length;
        lines.push('Select ' + remaining + ' more atom' + (remaining > 1 ? 's' : '') + ' ...');
      }}
      setOverlay(lines);
    }}

    function toggleSelection(atom) {{
      var idx = selection.findIndex(function(sel) {{ return sel.index === atom.index; }});
      if (idx >= 0) selection.splice(idx, 1);
      else {{
        if (cfg.maxAtoms > 0 && selection.length >= cfg.maxAtoms) selection.shift();
        selection.push(atom);
      }}
      refreshHighlights();
      refreshMeasurements();
      summaryLines();
    }}

    var viewerContainer = stage.viewer && stage.viewer.container ? stage.viewer.container : null;
    if (viewerContainer && !viewerContainer.dataset.molviewerMeasurementResetBound) {{
      viewerContainer.dataset.molviewerMeasurementResetBound = '1';
      var measurementResetHandler = function(event) {{
        if (cfg.mode !== 'measurement') return;
        if (event) {{
          event.preventDefault();
          event.stopPropagation();
        }}
        clearSelection();
      }};
      viewerContainer.addEventListener('contextmenu', measurementResetHandler);
      viewerContainer.addEventListener('mousedown', function(event) {{
        if (!event || event.button !== 2) return;
        measurementResetHandler(event);
      }});
    }}

    if (cfg.mode !== 'rotate' && cfg.maxAtoms > 0) {{
      stage.signals.clicked.add(function(pickingProxy) {{
        if (!pickingProxy || !pickingProxy.atom) return;
        var atom = cfg.atoms[pickingProxy.atom.index];
        if (!atom) return;
        toggleSelection(atom);
      }});
    }}

    refreshHighlights();
    refreshMeasurements();
    summaryLines();
  }}).catch(function(err) {{
    console.error('NGL load failed', err);
    setOverlay(['Failed to load structure: ' + err]);
  }});
}})();
</script>
"""

    if element_id:
        suffix = element_id
        replacements = {
            "molviewer-stage-wrapper": f"{suffix}-stage-wrapper",
            "ngl-stage": f"{suffix}-stage",
            "molviewer-snapshot-bar": f"{suffix}-snapshot-bar",
            "molviewer-snapshot-status": f"{suffix}-snapshot-status",
            "molviewer-snapshot-transparent": f"{suffix}-snapshot-transparent",
            "molviewer-snapshot-quality": f"{suffix}-snapshot-quality",
            "molviewer-snapshot-download-float": f"{suffix}-snapshot-download-float",
            "molviewer-snapshot-download": f"{suffix}-snapshot-download",
        }
        for old, new in replacements.items():
            html = html.replace(old, new)

    return html


def build_scatter_figure(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    z_axis: Optional[str],
    color_by: Optional[str],
    size_by: Optional[str],
    theme: ThemeConfig,
    grid: bool = True,
    text_size: int = 14,
    axis_labels: Optional[Dict[str, str]] = None,
):
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
        font=dict(size=max(8, int(text_size))),
    )
    axis_labels = axis_labels or {}
    if z_axis:
        fig.update_scenes(
            xaxis=dict(
                title=str(axis_labels.get("x") or x_axis),
                showgrid=bool(grid),
            ),
            yaxis=dict(
                title=str(axis_labels.get("y") or y_axis),
                showgrid=bool(grid),
            ),
            zaxis=dict(
                title=str(axis_labels.get("z") or z_axis),
                showgrid=bool(grid),
            ),
        )
    else:
        fig.update_layout(
            xaxis=dict(
                title=str(axis_labels.get("x") or x_axis),
                showgrid=bool(grid),
            ),
            yaxis=dict(
                title=str(axis_labels.get("y") or y_axis),
                showgrid=bool(grid),
            ),
        )
    return fig


def pick_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BASE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


def pick_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BASE_COLUMNS and not pd.api.types.is_numeric_dtype(df[c])]


def sidebar_controls(
    df: pd.DataFrame, *, enable_scatter: bool, show_scatter_controls: bool = True
) -> Dict[str, Any]:
    st.sidebar.header("Data")
    x_axis = y_axis = z_axis = color_by = size_by = None

    if enable_scatter:
        if show_scatter_controls:
            scatter_mode = st.sidebar.radio("Scatter dimensionality", ["2D", "3D"], index=0)
            numeric_cols = pick_numeric_columns(df)
            categorical_cols = pick_categorical_columns(df)
            if not numeric_cols:
                numeric_cols = ["__index"]
            x_axis = st.sidebar.selectbox("X axis", numeric_cols, index=0)
            y_axis = st.sidebar.selectbox(
                "Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1)
            )
            if scatter_mode == "3D":
                z_axis = st.sidebar.selectbox(
                    "Z axis", numeric_cols, index=min(2, len(numeric_cols) - 1)
                )
            color_choice = st.sidebar.selectbox(
                "Color by",
                ["None", *numeric_cols, *categorical_cols],
                index=0,
            )
            color_by = None if color_choice == "None" else color_choice
            size_choice = st.sidebar.selectbox("Size by", ["Uniform", *numeric_cols], index=0)
            size_by = None if size_choice == "Uniform" else size_choice
        else:
            numeric_cols = pick_numeric_columns(df)
            if not numeric_cols:
                numeric_cols = ["__index"]
            x_axis = numeric_cols[0]
            y_axis = numeric_cols[min(1, len(numeric_cols) - 1)]
            if len(numeric_cols) > 2:
                z_axis = numeric_cols[min(2, len(numeric_cols) - 1)]
            st.sidebar.caption("Configure plotting in the main panel below the chart.")
    elif show_scatter_controls:
        st.sidebar.info("Add numeric properties to enable scatter plotting.")

    with st.sidebar.expander("3D Viewer", expanded=False):
        mode_label_to_key = {
            "Rotate / navigate": "rotate",
            "Select atom (inspect properties)": "select",
            "Measurement (auto)": "measurement",
        }
        viewer_mode_label = st.selectbox(
            "Mouse mode",
            list(mode_label_to_key.keys()),
            index=0,
            help="Choose how mouse clicks interact with the 3D viewer.",
        )
        viewer_mode = mode_label_to_key[viewer_mode_label]
        sphere_radius = st.slider("Atom radius", min_value=0.1, max_value=0.8, value=0.3, step=0.05)
        bond_radius = st.slider("Bond radius", min_value=0.05, max_value=0.4, value=0.12, step=0.01)
        representation_style = st.selectbox(
            "Rendering style",
            [
                "Ball + Stick",
                "Licorice",
                "Spacefilling",
                "Hyperball",
                "Line",
                "Point Cloud",
                "Surface",
            ],
            index=0,
            help="Choose how the molecule is drawn in the 3D viewer.",
        )
        atom_label = st.selectbox(
            "Atom label",
            ["None", "Symbol", "Atomic number", "Atom index"],
            index=0,
            help="Choose a single annotation to display in the 3D viewer",
        )
        show_hydrogens = st.checkbox(
            "Show hydrogen atoms",
            value=True,
            help="Uncheck to hide hydrogens across the viewer, measurements, and metadata tables.",
        )
        snapshot_transparent = st.checkbox(
            "Transparent snapshot background",
            value=False,
            help="When enabled, PNG exports use a transparent backdrop instead of the theme color.",
        )
        snapshot_quality_labels = [label for label, _ in SNAPSHOT_QUALITY_OPTIONS]
        snapshot_quality = st.selectbox(
            "Snapshot quality",
            snapshot_quality_labels,
            index=0,
            help="Choose a resolution multiplier for PNG exports.",
        )

    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "color_by": color_by,
        "size_by": size_by,
        "sphere_radius": sphere_radius,
        "bond_radius": bond_radius,
        "representation_style": representation_style,
        "atom_label": None if atom_label == "None" else atom_label,
        "viewer_mode": viewer_mode,
        "show_hydrogens": show_hydrogens,
        "snapshot_transparent": snapshot_transparent,
        "snapshot_quality": snapshot_quality,
    }


def default_plot_config(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = pick_numeric_columns(df)
    if not numeric_cols:
        numeric_cols = ["__index"]
    mode = "3D" if len(numeric_cols) >= 3 else "2D"
    x_axis = numeric_cols[0]
    y_axis = numeric_cols[min(1, len(numeric_cols) - 1)]
    z_axis = numeric_cols[min(2, len(numeric_cols) - 1)] if len(numeric_cols) >= 3 else None
    return {
        "mode": mode,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "color_by": None,
        "size_by": None,
        "grid": True,
        "text_size": 14,
        "axis_labels": {
            "x": x_axis,
            "y": y_axis,
            "z": z_axis or "",
        },
    }


def sanitize_plot_config(config: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    base = default_plot_config(df)
    if not config:
        return base

    numeric_cols = pick_numeric_columns(df)
    if not numeric_cols:
        numeric_cols = ["__index"]
    categorical_cols = pick_categorical_columns(df)

    sanitized: Dict[str, Any] = {}
    mode = str(config.get("mode", base["mode"]))
    if mode not in {"2D", "3D"}:
        mode = base["mode"]
    if mode == "3D" and len(numeric_cols) < 3:
        mode = "2D"
    sanitized["mode"] = mode

    def pick_numeric(value: Optional[str], default: str) -> str:
        return value if value in numeric_cols else default

    x_axis = pick_numeric(config.get("x_axis"), base["x_axis"])
    y_axis = pick_numeric(config.get("y_axis"), base["y_axis"])
    if mode == "3D":
        default_z = base["z_axis"] if base["z_axis"] in numeric_cols else (
            numeric_cols[min(2, len(numeric_cols) - 1)] if len(numeric_cols) >= 3 else None
        )
        z_candidate = config.get("z_axis")
        z_axis = z_candidate if z_candidate in numeric_cols else default_z
    else:
        z_axis = None
    sanitized.update({
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
    })

    color_options = set(numeric_cols) | set(categorical_cols)
    color_choice = config.get("color_by")
    sanitized["color_by"] = color_choice if color_choice in color_options else None

    size_choice = config.get("size_by")
    sanitized["size_by"] = size_choice if size_choice in numeric_cols else None

    sanitized["grid"] = bool(config.get("grid", base["grid"]))

    try:
        text_size = int(config.get("text_size", base["text_size"]))
    except (TypeError, ValueError):
        text_size = base["text_size"]
    sanitized["text_size"] = max(8, min(48, text_size))

    labels = config.get("axis_labels", {}) or {}
    sanitized["axis_labels"] = {
        "x": str(labels.get("x") or x_axis),
        "y": str(labels.get("y") or y_axis),
        "z": str(labels.get("z") or (z_axis or "")),
    }

    return sanitized


def plot_controls_panel(
    df: pd.DataFrame,
    *,
    key_prefix: str = "plot",
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    defaults = sanitize_plot_config(defaults or default_plot_config(df), df)
    numeric_cols = pick_numeric_columns(df)
    if not numeric_cols:
        numeric_cols = ["__index"]
    categorical_cols = pick_categorical_columns(df)

    if len(numeric_cols) < 3:
        scatter_options = ["2D"]
    else:
        scatter_options = ["2D", "3D"]

    prefixed = lambda name: f"{key_prefix}_{name}"

    def ensure_state(key: str, value: Any) -> None:
        state_key = prefixed(key)
        if state_key not in st.session_state:
            st.session_state[state_key] = value

    ensure_state("dim", defaults["mode"] if defaults["mode"] in scatter_options else scatter_options[0])
    ensure_state("x", defaults["x_axis"])
    ensure_state("y", defaults["y_axis"])
    ensure_state("z", defaults.get("z_axis"))
    ensure_state("color", defaults.get("color_by"))
    ensure_state("size", defaults.get("size_by"))
    ensure_state("grid", defaults.get("grid", True))
    ensure_state("text_size", defaults.get("text_size", 14))
    axis_labels = defaults.get("axis_labels", {})
    ensure_state("label_x", axis_labels.get("x", defaults["x_axis"]))
    ensure_state("label_y", axis_labels.get("y", defaults["y_axis"]))
    ensure_state("label_z", axis_labels.get("z", defaults.get("z_axis") or ""))

    with st.container():
        st.markdown(
            "<div style='padding:10px 12px;border:1px solid rgba(15,23,42,0.15);"
            "border-radius:10px;background:rgba(241,243,249,0.6);margin-bottom:8px;'>"
            "<strong>Plot settings</strong></div>",
            unsafe_allow_html=True,
        )
        scatter_mode = st.radio(
            "Scatter dimensionality",
            scatter_options,
            index=scatter_options.index(st.session_state[prefixed("dim")]),
            horizontal=True,
            key=prefixed("dim"),
        )
        axis_cols = st.columns(2)
        with axis_cols[0]:
            x_axis = st.selectbox(
                "X axis",
                numeric_cols,
                index=numeric_cols.index(
                    st.session_state[prefixed("x")] if st.session_state[prefixed("x")] in numeric_cols else defaults["x_axis"]
                ),
                key=prefixed("x"),
            )
        with axis_cols[1]:
            y_axis = st.selectbox(
                "Y axis",
                numeric_cols,
                index=numeric_cols.index(
                    st.session_state[prefixed("y")] if st.session_state[prefixed("y")] in numeric_cols else defaults["y_axis"]
                ),
                key=prefixed("y"),
            )

        z_axis = None
        if scatter_mode == "3D":
            z_default = (
                st.session_state[prefixed("z")]
                if st.session_state[prefixed("z")] in numeric_cols
                else defaults.get("z_axis")
            )
            if z_default is None and len(numeric_cols) >= 3:
                z_default = numeric_cols[min(2, len(numeric_cols) - 1)]
            z_index = numeric_cols.index(z_default) if z_default in numeric_cols else min(2, len(numeric_cols) - 1)
            z_axis = st.selectbox(
                "Z axis",
                numeric_cols,
                index=z_index,
                key=prefixed("z"),
            )

        color_options = ["None", *numeric_cols, *categorical_cols]
        current_color = st.session_state[prefixed("color")]
        color_index = color_options.index(current_color) if current_color in color_options else 0
        color_choice = st.selectbox(
            "Color column",
            color_options,
            index=color_index,
            key=prefixed("color"),
        )
        color_by = None if color_choice == "None" else color_choice

        size_options = ["Uniform", *numeric_cols]
        current_size = st.session_state[prefixed("size")]
        size_index = size_options.index(current_size) if current_size in size_options else 0
        size_choice = st.selectbox(
            "Size column",
            size_options,
            index=size_index,
            key=prefixed("size"),
        )
        size_by = None if size_choice == "Uniform" else size_choice

        grid = st.toggle(
            "Show gridlines",
            value=bool(st.session_state[prefixed("grid")]),
            key=prefixed("grid"),
        )
        text_size = st.slider(
            "Text size",
            min_value=8,
            max_value=36,
            value=int(st.session_state[prefixed("text_size")]),
            key=prefixed("text_size"),
        )

        label_cols = st.columns(2)
        with label_cols[0]:
            x_label = st.text_input(
                "X label",
                value=st.session_state[prefixed("label_x")],
                key=prefixed("label_x"),
            )
        with label_cols[1]:
            y_label = st.text_input(
                "Y label",
                value=st.session_state[prefixed("label_y")],
                key=prefixed("label_y"),
            )
        if scatter_mode == "3D":
            z_label = st.text_input(
                "Z label",
                value=st.session_state[prefixed("label_z")],
                key=prefixed("label_z"),
            )
        else:
            z_label = st.session_state.get(prefixed("label_z"), defaults.get("axis_labels", {}).get("z", ""))

    config = sanitize_plot_config(
        {
            "mode": scatter_mode,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z_axis": z_axis,
            "color_by": color_by,
            "size_by": size_by,
            "grid": grid,
            "text_size": text_size,
            "axis_labels": {
                "x": (x_label or "").strip() or x_axis,
                "y": (y_label or "").strip() or y_axis,
                "z": (z_label or "").strip() or (z_axis or ""),
            },
        },
        df,
    )

    st.session_state[prefixed("config")] = config
    return config


def plot_and_select(
    df: pd.DataFrame,
    fig: Any,
    *,
    downloads: Optional[List[Dict[str, Any]]] = None,
    download_error: Optional[str] = None,
) -> Optional[str]:
    downloads = downloads or []
    st.subheader("Data Exploration")
    if downloads:
        cols = st.columns(len(downloads))
        for idx, item in enumerate(downloads):
            with cols[idx]:
                st.download_button(
                    label=item.get("label", "Download"),
                    data=item["data"],
                    file_name=item.get("filename", "plot"),
                    mime=item.get("mime", "application/octet-stream"),
                    key=f"plot-download-{item.get('key', idx)}",
                    use_container_width=True,
                )
    elif download_error:
        st.caption(download_error)
    selected_id: Optional[str] = st.session_state.get("selected_id")
    if plotly_events is None:
        st.warning(
            "Install `streamlit-plotly-events` to enable point selection by clicking. Using dropdown fallback."
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
        events = plotly_events(fig, click_event=True, select_event=False, override_height=600, key="scatter_ngl")
        if events:
            event = events[0]
            candidate = None
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


def _format_atom_option(idx: int, symbols: Iterable[str], numbers: Iterable[int]) -> str:
    return f"{idx + 1} - {symbols[idx]} (Z={int(numbers[idx])})"


def _compute_distance(coords: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(coords[a] - coords[b]))


def _compute_angle(coords: np.ndarray, a: int, b: int, c: int) -> float:
    vec1 = coords[a] - coords[b]
    vec2 = coords[c] - coords[b]
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if not norm1 or not norm2:
        return float("nan")
    cos_theta = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _compute_dihedral(coords: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    b0 = coords[b] - coords[a]
    b1 = coords[c] - coords[b]
    b2 = coords[d] - coords[c]
    b1_norm = np.linalg.norm(b1)
    if not b1_norm:
        return float("nan")
    b1_unit = b1 / b1_norm
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if not n1_norm or not n2_norm:
        return float("nan")
    n1_unit = n1 / n1_norm
    n2_unit = n2 / n2_norm
    m1 = np.cross(n1_unit, b1_unit)
    x = np.dot(n1_unit, n2_unit)
    y = np.dot(m1, n2_unit)
    return float(np.degrees(np.arctan2(y, x)))


def show_measurement_panel(atoms: Atoms, mode: str, *, key_prefix: str) -> None:
    num_atoms = len(atoms)
    if num_atoms == 0:
        st.info("No atoms available for measurements.")
        return
    symbols = atoms.get_chemical_symbols()
    numbers = atoms.get_atomic_numbers()
    coords = atoms.get_positions()
    masses = atoms.get_masses()
    options = list(range(num_atoms))
    fmt = lambda idx: _format_atom_option(idx, symbols, numbers)
    st.markdown("**Measurement tools**")
    if mode == "select":
        idx = st.selectbox("Atom", options, format_func=fmt, key=f"{key_prefix}_sel")
        st.markdown(f"- Symbol: `{symbols[idx]}`  Z={int(numbers[idx])}")
        st.markdown("- Coordinates (Angstrom): ({:.4f}, {:.4f}, {:.4f})".format(*coords[idx]))
        st.markdown(f"- Mass (amu): {float(masses[idx]):.4f}")
        return
    if mode == "measurement":
        st.caption(
            "Select 2–4 atoms in the 3D view to measure distances, angles, or dihedrals automatically. "
            "Use the pickers below for manual calculations."
        )
        if num_atoms >= 2:
            col1, col2 = st.columns(2)
            a = col1.selectbox("Atom 1", options, format_func=fmt, key=f"{key_prefix}_dist_a")
            b = col2.selectbox(
                "Atom 2",
                options,
                format_func=fmt,
                key=f"{key_prefix}_dist_b",
                index=1 if num_atoms > 1 else 0,
            )
            dist = _compute_distance(coords, a, b)
            st.markdown(f"**Distance:** {dist:.4f} Angstrom")
        else:
            st.info("At least two atoms are required for distance measurements.")

        if num_atoms >= 3:
            col1, col2, col3 = st.columns(3)
            a = col1.selectbox("Atom 1", options, format_func=fmt, key=f"{key_prefix}_angle_a")
            b = col2.selectbox(
                "Atom 2 (vertex)",
                options,
                format_func=fmt,
                key=f"{key_prefix}_angle_b",
                index=1 if num_atoms > 1 else 0,
            )
            c = col3.selectbox(
                "Atom 3",
                options,
                format_func=fmt,
                key=f"{key_prefix}_angle_c",
                index=2 if num_atoms > 2 else 0,
            )
            angle = _compute_angle(coords, a, b, c)
            st.markdown(f"**Angle (deg):** {angle:.4f}")
            st.caption(
                f"d(1-2) = {_compute_distance(coords, a, b):.4f} Angstrom · "
                f"d(2-3) = {_compute_distance(coords, b, c):.4f} Angstrom"
            )
        else:
            st.info("At least three atoms are required for angle measurements.")

        if num_atoms >= 4:
            cols = st.columns(4)
            a = cols[0].selectbox("Atom 1", options, format_func=fmt, key=f"{key_prefix}_dih_a")
            b = cols[1].selectbox(
                "Atom 2",
                options,
                format_func=fmt,
                key=f"{key_prefix}_dih_b",
                index=1 if num_atoms > 1 else 0,
            )
            c = cols[2].selectbox(
                "Atom 3",
                options,
                format_func=fmt,
                key=f"{key_prefix}_dih_c",
                index=2 if num_atoms > 2 else 0,
            )
            d = cols[3].selectbox(
                "Atom 4",
                options,
                format_func=fmt,
                key=f"{key_prefix}_dih_d",
                index=3 if num_atoms > 3 else 0,
            )
            dihedral = _compute_dihedral(coords, a, b, c, d)
            st.markdown(f"**Dihedral (deg):** {dihedral:.4f}")
        else:
            st.info("At least four atoms are required for dihedral measurements.")
        return

    st.info("Switch mouse mode to enable backend measurements.")


def summarize_atoms(atoms: Optional[Atoms]) -> Dict[str, Any]:
    if atoms is None or len(atoms) == 0:
        return {
            "formula": "—",
            "num_atoms": 0,
            "mass_amu": None,
            "center_x": None,
            "center_y": None,
            "center_z": None,
            "cell_volume": None,
        }
    cell_volume = None
    if atoms.cell is not None and atoms.cell.volume > 0:
        cell_volume = float(atoms.cell.volume)
    com = atoms.get_center_of_mass()
    return {
        "formula": atoms.get_chemical_formula(),
        "num_atoms": int(len(atoms)),
        "mass_amu": float(atoms.get_masses().sum()),
        "center_x": float(com[0]),
        "center_y": float(com[1]),
        "center_z": float(com[2]),
        "cell_volume": cell_volume,
    }


def _render_dataframe(df: pd.DataFrame, theme: ThemeConfig) -> None:
    if theme.background.lower() == "#fafbff":
        styled = (
            df.style.set_properties(
                **{
                    "background-color": "#FFFFFF",
                    "color": "#0F172A",
                    "border-color": "rgba(15, 23, 42, 0.12)",
                }
            )
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", "#0F172A"),
                            ("border-color", "rgba(15, 23, 42, 0.12)"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [("border-color", "rgba(15, 23, 42, 0.12)")],
                    },
                ]
            )
        )
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)


def show_details(record: pd.Series, atoms: Optional[Atoms], theme: ThemeConfig) -> None:
    st.markdown(
        f"<div class='molviewer-card'><strong>Selected structure:</strong> `{record.label}`</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='molviewer-card'><strong>Basic information</strong></div>", unsafe_allow_html=True)
    _render_dataframe(pd.DataFrame([summarize_atoms(atoms)]), theme)
    metadata = {k: v for k, v in record.items() if k not in BASE_COLUMNS}
    if metadata:
        st.markdown("<div class='molviewer-card'><strong>Metadata</strong></div>", unsafe_allow_html=True)
        _render_dataframe(pd.DataFrame([metadata]), theme)


def navigation_controls(df: pd.DataFrame, selected_id: Optional[str]) -> Optional[str]:
    options = df["selection_id"].tolist()
    if not options:
        return selected_id
    if selected_id not in options:
        selected_id = options[0]
    idx = options.index(selected_id)
    prev_col, center_col, next_col = st.columns([1, 3, 1])
    if prev_col.button("◀ Previous", use_container_width=True):
        idx = (idx - 1) % len(options)
    if next_col.button("Next ▶", use_container_width=True):
        idx = (idx + 1) % len(options)
    new_id = options[idx]
    with center_col:
        st.markdown(
            f"**{df.loc[df['selection_id'] == new_id, 'label'].iloc[0]} ({idx + 1}/{len(options)})**"
        )
    st.session_state["selected_id"] = new_id
    return new_id


@st.cache_data(show_spinner=True)
def compute_dataset_statistics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    summary_rows = []
    element_counter = Counter()
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
        {"element": elem, "count": count} for elem, count in element_counter.most_common()
    )
    return summary_df, elements_df, failures


def render_distribution_charts(
    summary_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    theme: ThemeConfig,
    df: pd.DataFrame,
) -> None:
    numeric_cols: list[str] = []
    if df is not None:
        for col in pick_numeric_columns(df):
            series = df[col]
            if series.notna().any():
                numeric_cols.append(col)

    options: list[str] = []
    if not summary_df.empty:
        options.extend(["Number of atoms", "Total mass"])
    if not elements_df.empty:
        options.append("Atom types")
    if numeric_cols:
        options.extend(numeric_cols)

    if not options:
        st.info("Unable to compute distributions for the current dataset.")
        return

    container = st.container()
    with container:
        st.markdown(
            "<div style='padding:10px 12px;border:1px solid rgba(15,23,42,0.15);"
            "border-radius:10px;background:rgba(241,243,249,0.6);margin-bottom:12px;'>"
            "<strong>Distribution settings</strong></div>",
            unsafe_allow_html=True,
        )

        choice = st.selectbox(
            "Distribution",
            options,
            index=min(options.index(st.session_state.get("distribution_choice", options[0])) if "distribution_choice" in st.session_state and st.session_state["distribution_choice"] in options else 0, len(options) - 1),
            key="distribution_choice",
        )

        mode_options = ["Histogram"] if choice == "Atom types" else ["Histogram", "KDE"]
        if "distribution_mode" not in st.session_state or st.session_state["distribution_mode"] not in mode_options:
            st.session_state["distribution_mode"] = mode_options[0]
        mode = st.radio(
            "Plot type",
            mode_options,
            index=mode_options.index(st.session_state["distribution_mode"]),
            key="distribution_mode",
            horizontal=True,
        )

        cols = st.columns(2)
        with cols[0]:
            grid = st.toggle(
                "Show gridlines",
                value=st.session_state.get("distribution_grid", True),
                key="distribution_grid",
            )
        with cols[1]:
            text_size = st.slider(
                "Text size",
                min_value=8,
                max_value=36,
                value=int(st.session_state.get("distribution_text_size", 14)),
                key="distribution_text_size",
            )

        label_key = f"distribution_label_{choice}"
        default_label = st.session_state.get(label_key, choice)
        label = st.text_input(
            "Axis label",
            value=default_label,
            key=label_key,
        )
        if not label.strip():
            label = choice

    color_sequence = [theme.highlight]
    fig = None
    plot_mode = "plotly"

    if choice == "Number of atoms" and not summary_df.empty:
        fig = px.histogram(
            summary_df,
            x="num_atoms",
            nbins=min(30, max(5, summary_df["num_atoms"].nunique())),
            template=theme.plot_template,
            color_discrete_sequence=color_sequence,
        )
        fig.update_xaxes(title_text=label, showgrid=grid)
        fig.update_yaxes(title_text="Count", showgrid=grid)
    elif choice == "Total mass" and not summary_df.empty:
        fig = px.histogram(
            summary_df,
            x="mass_amu",
            nbins=30,
            template=theme.plot_template,
            color_discrete_sequence=color_sequence,
        )
        fig.update_xaxes(title_text=label, showgrid=grid)
        fig.update_yaxes(title_text="Count", showgrid=grid)
    elif choice == "Atom types":
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
        fig.update_xaxes(title_text=label, showgrid=grid)
        fig.update_yaxes(title_text="Frequency", showgrid=grid)
    else:
        if choice not in df.columns:
            st.info("Selected property unavailable for distribution plotting.")
            return
        col_data = df[choice].dropna()
        if col_data.empty:
            st.info("No values available for this property.")
            return
        values = pd.to_numeric(col_data, errors="coerce").dropna().astype(float)
        if values.empty:
            st.info("No numeric values available for this property.")
            return
        plot_mode = "plotly"
        if mode == "Histogram":
            fig = px.histogram(
                values.to_frame(name=choice),
                x=choice,
                nbins=min(40, max(5, values.nunique() or 1)),
                template=theme.plot_template,
                color_discrete_sequence=color_sequence,
            )
        else:  # KDE via matplotlib/seaborn
            plot_mode = "matplotlib"
            sns.set_theme(style="white")
            fig, ax = plt.subplots(figsize=(12, 8))
            kde = sns.kdeplot(
                values.to_numpy(),
                label=label,
                color=theme.highlight,
                fill=False,
                linewidth=3,
                ax=ax,
            )
            if kde.lines:
                line = kde.lines[-1]
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                ax.fill_between(
                    x_data,
                    0,
                    y_data,
                    color=to_rgba(theme.highlight, alpha=0.35),
                    zorder=1,
                )
            _, y_max = ax.get_ylim()
            if y_max:
                ax.set_ylim(0, y_max * 1.05)
            ax.set_title(f"{label} Distribution", fontsize=max(14, text_size + 4), color=theme.text_color)
            ax.set_xlabel(label, fontsize=max(12, text_size), color=theme.text_color)
            ax.set_ylabel("Density", fontsize=max(12, text_size), color=theme.text_color)
            ax.tick_params(axis="both", labelsize=max(10, text_size - 2), colors=theme.text_color)
            if grid:
                ax.grid(color="#CBD5F5" if theme.background.lower() == "#fafbff" else "#334155", alpha=0.4)
            else:
                ax.grid(False)
            ax.set_facecolor(theme.background)
            fig.patch.set_facecolor(theme.background)
            for spine in ax.spines.values():
                spine.set_color(theme.text_color)
            fig.tight_layout()

    if fig is None:
        st.info("Unable to render the requested distribution.")
        return

    if plot_mode == "plotly":
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor=theme.plot_bg,
            paper_bgcolor=theme.background,
            font=dict(size=max(8, int(text_size)), color=theme.text_color),
            height=600,
            legend=dict(font=dict(color=theme.text_color)),
        )
        fig.update_traces(marker_color=theme.highlight, selector={"type": "histogram"})
        fig.update_xaxes(
            showgrid=grid,
            title_font=dict(color=theme.text_color, size=max(8, int(text_size))),
            tickfont=dict(color=theme.text_color, size=max(8, int(text_size) - 2)),
            gridcolor="rgba(15,23,42,0.12)" if grid else "rgba(0,0,0,0)",
            linecolor="rgba(15,23,42,0.2)",
            zeroline=False,
        )
        fig.update_yaxes(
            showgrid=grid,
            title_font=dict(color=theme.text_color, size=max(8, int(text_size))),
            tickfont=dict(color=theme.text_color, size=max(8, int(text_size) - 2)),
            gridcolor="rgba(15,23,42,0.12)" if grid else "rgba(0,0,0,0)",
            linecolor="rgba(15,23,42,0.2)",
            zeroline=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        plt.close(fig)


def _rank_xyz_matches(df: pd.DataFrame, query: str, limit: int = 200) -> pd.DataFrame:
    """
    Return up to `limit` best rows whose identifier/label match `query`.
    Ranking rule (simple, dependency-free):
      - Case-insensitive
      - prefix match > substring
      - shorter match position is better
    """
    if not query:
        return df.head(limit)

    q = str(query).strip().lower()
    if not q:
        return df.head(limit)

    id_series = df["identifier"].astype(str)
    lbl_series = df["label"].astype(str)

    # Compute simple scores
    scores = []
    for i, (fid, lbl) in enumerate(zip(id_series, lbl_series)):
        s_id = fid.lower()
        s_lbl = lbl.lower()
        best_pos = 10**9
        best_kind = 0  # 2 = prefix, 1 = substring, 0 = no match

        # identifier
        if s_id.startswith(q):
            best_kind = max(best_kind, 2)
            best_pos = 0
        elif q in s_id:
            best_kind = max(best_kind, 1)
            best_pos = min(best_pos, s_id.index(q))

        # label
        if s_lbl.startswith(q):
            best_kind = max(best_kind, 2)
            best_pos = min(best_pos, 0)
        elif q in s_lbl:
            if best_kind < 2:
                best_kind = max(best_kind, 1)
            best_pos = min(best_pos, s_lbl.index(q))

        if best_kind > 0:
            # Higher kind, lower position = better; use negative for descending
            score = (best_kind, -best_pos)
            scores.append((i, score))

    if not scores:
        return df.iloc[0:0]  # empty

    # Sort by best_kind desc, best_pos asc (achieved via -best_pos above)
    scores.sort(key=lambda t: (t[1][0], t[1][1]), reverse=True)
    idx = [i for i, _ in scores[:limit]]
    return df.iloc[idx]


def xyz_navbar(df: pd.DataFrame, selected_id: Optional[str]) -> Optional[str]:
    """
    Top sticky navigation bar for XYZ mode:
      - ◀ Prev | type-to-filter selectbox | ▶ Next
      - Search over filename (identifier) and label with ranking.
    Returns possibly updated selected_id.
    """
    st.markdown('<div class="molviewer-navbar">', unsafe_allow_html=True)

    # Current position for prev/next logic
    all_ids = df["selection_id"].tolist()
    if not all_ids:
        st.markdown("</div>", unsafe_allow_html=True)
        return selected_id
    if selected_id not in all_ids:
        selected_id = all_ids[0]
    cur_idx = all_ids.index(selected_id)

    c_prev, c_search, c_next = st.columns([1, 8, 1])

    # Prev
    with c_prev:
        st.markdown('<span class="small-label">Navigate</span>', unsafe_allow_html=True)
        if st.button("◀", use_container_width=True, key="xyz_nav_prev"):
            cur_idx = (cur_idx - 1) % len(all_ids)
            selected_id = all_ids[cur_idx]

    # Search + autocomplete
    with c_search:
        # Persist user query between reruns
        query_key = "xyz_search_query"
        query = st.text_input("Find file (type to filter)", key=query_key, placeholder="e.g., 000123.xyz or 'benzene'")
        # Rank + filter candidates
        sub = _rank_xyz_matches(df, query, limit=500) if query is not None else df.head(200)

        # Build pretty labels and a mapping for selectbox
        label_map = {
            row["selection_id"]: (
                f"{row['identifier']} — {row['label']}"
                if str(row["identifier"]) != str(row["label"]) else str(row["identifier"])
            )
            for _, row in sub.iterrows()
        }
        opts = list(label_map.keys())
        # Keep current selection in filtered list if possible
        idx = opts.index(selected_id) if selected_id in opts else (0 if opts else -1)

        chosen = st.selectbox(
            "Matches (autocomplete)",
            options=opts,
            index=idx if idx >= 0 else 0,
            format_func=lambda sid: label_map.get(sid, sid),
            help="Type to autocomplete; matches filename and label.",
            label_visibility="collapsed",
            key="xyz_autocomplete_select",
        )
        if chosen:
            selected_id = chosen

    # Next
    with c_next:
        st.markdown('<span class="small-label">Navigate</span>', unsafe_allow_html=True)
        if st.button("▶", use_container_width=True, key="xyz_nav_next"):
            # Recompute in case selected_id changed via search
            cur_idx = all_ids.index(selected_id)
            cur_idx = (cur_idx + 1) % len(all_ids)
            selected_id = all_ids[cur_idx]
    st.caption("Tip: Press **Backspace** for previous, **Enter** for next. (Also works with ◀ / ▶ buttons.)")

    st.markdown("</div>", unsafe_allow_html=True)

    # Persist selection
    st.session_state["selected_id"] = selected_id
    return selected_id


def arrow_key_listener() -> Optional[str]:
    """
    Waits for Streamlit to be ready, then attaches a robust keyboard listener.
    """
    from streamlit.components.v1 import html
    return html(
        """
        <script>
        (function(){
            // Use a unique name for the guard flag to ensure this runs only once.
            if (window.parent.molViewerGlobalKeyListenerAttached) {
                return;
            }

            function initializeListener(Streamlit) {
                // Set the guard flag only after we successfully get the Streamlit object.
                if (window.parent.molViewerGlobalKeyListenerAttached) return;
                window.parent.molViewerGlobalKeyListenerAttached = true;

                console.log("3DMolViewer: Streamlit is ready. Attaching key listener.");

                let lastSend = 0;
                function sendValueToStreamlit(val) {
                    const now = Date.now();
                    if (now - lastSend < 150) return; // Throttle events
                    lastSend = now;
                    Streamlit.setComponentValue(val);
                }

                function isUserTyping() {
                    const el = window.parent.document.activeElement;
                    if (!el) return false;
                    const tagName = el.tagName.toUpperCase();
                    return tagName === 'INPUT' || tagName === 'TEXTAREA' || el.isContentEditable;
                }

                window.parent.document.addEventListener("keydown", (e) => {
                    if (isUserTyping()) return;

                    let intent = null;
                    switch (e.key) {
                        case "Enter":
                        case "ArrowRight":
                        case ">":
                            intent = "next";
                            break;
                        case "Backspace":
                        case "ArrowLeft":
                        case "<":
                            intent = "prev";
                            break;
                    }

                    if (intent) {
                        e.preventDefault();
                        e.stopPropagation();
                        sendValueToStreamlit(intent);
                    }
                });

                // Finally, signal that the component is ready.
                Streamlit.setComponentReady();
            }

            // This function waits for the Streamlit object to be available on the parent window.
            function waitForStreamlit() {
                const streamlitParent = window.parent;
                if (streamlitParent && streamlitParent.Streamlit) {
                    initializeListener(streamlitParent.Streamlit);
                } else {
                    // If not found, check again in 50 milliseconds.
                    setTimeout(waitForStreamlit, 50);
                }
            }

            // Start the waiting process.
            waitForStreamlit();
        })();
        </script>
        """,
        height=0,
    )

def main() -> None:
    st.set_page_config(page_title="3DMolViewer (NGL)", layout="wide")
    theme_name = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=1)
    theme = THEMES[theme_name]
    inject_theme_css(theme)
    st.title("3DMolViewer × NGL: Structure-Property Explorer")

    st.sidebar.header("Data Source")
    source_type = st.sidebar.radio("Source", ["XYZ Directory", "ASE Database"], index=0)
    try:
        if source_type == "XYZ Directory":
            default_xyz_dir = DEFAULT_XYZ_DIR if Path(DEFAULT_XYZ_DIR).exists() else ""
            xyz_dir = st.sidebar.text_input("XYZ directory", value=default_xyz_dir)
            default_csv = PACKAGE_ROOT / "test_xyz" / "xyz_properties.csv"
            csv_default_value = str(default_csv) if default_csv.exists() else ""
            csv_path = st.sidebar.text_input(
                "Optional CSV with properties", value=csv_default_value
            )
            if not xyz_dir:
                st.info("Provide a directory containing .xyz files to begin.")
                return
            df = load_xyz_metadata(xyz_dir, csv_path or None)
            if df.attrs.get("csv_properties_ignored"):
                ignored_path = df.attrs.get("csv_properties_path") or csv_path or "property CSV"
                st.warning(
                    f"Ignored properties from '{ignored_path}' because none of its filenames matched the XYZ files."
                )
            if df.attrs.get("csv_xyz_filtered"):
                skipped = df.attrs["csv_xyz_filtered"]
                st.info(
                    f"Skipped {skipped} structure{'s' if skipped != 1 else ''} without matching properties from the CSV."
                )
            if df.attrs.get("csv_only_count"):
                csv_only = df.attrs["csv_only_count"]
                st.info(
                    f"Loaded {csv_only} CSV-only entr{'y' if csv_only == 1 else 'ies'} without geometry; 3D view will be unavailable."
                )
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

    if "__index" not in df.columns:
        df["__index"] = np.arange(len(df))

    has_numeric = bool(pick_numeric_columns(df))
    viewer_config = sidebar_controls(
        df, enable_scatter=has_numeric, show_scatter_controls=False
    )

    selected_id: Optional[str] = None

    if has_numeric:
        plot_config_state_key = "plot_config_state"
        current_defaults = sanitize_plot_config(
            st.session_state.get(plot_config_state_key), df
        )
        st.session_state[plot_config_state_key] = current_defaults

        settings_visible_key = "plot_settings_visible"
        if settings_visible_key not in st.session_state:
            st.session_state[settings_visible_key] = False

        left_col, right_col = st.columns((1, 1))
        with left_col:
            toggle_label = (
                "Hide plot settings"
                if st.session_state[settings_visible_key]
                else "Show plot settings"
            )
            if st.button(
                f"🎛️ {toggle_label}",
                key="plot_settings_toggle",
                type="primary",
            ):
                st.session_state[settings_visible_key] = (
                    not st.session_state[settings_visible_key]
                )

            if st.session_state[settings_visible_key]:
                plot_config = plot_controls_panel(
                    df, defaults=st.session_state[plot_config_state_key]
                )
            else:
                plot_config = st.session_state[plot_config_state_key]
                summary_bits = [
                    plot_config["mode"],
                    f"X={plot_config['x_axis']}",
                    f"Y={plot_config['y_axis']}",
                ]
                if plot_config["mode"] == "3D" and plot_config.get("z_axis"):
                    summary_bits.append(f"Z={plot_config['z_axis']}")
                if plot_config.get("color_by"):
                    summary_bits.append(f"color={plot_config['color_by']}")
                if plot_config.get("size_by"):
                    summary_bits.append(f"size={plot_config['size_by']}")
                st.caption(" · ".join(summary_bits))
                quick_grid = st.checkbox(
                    "Show gridlines",
                    value=plot_config.get("grid", True),
                    key="plot_quick_grid",
                )
                plot_config["grid"] = bool(quick_grid)

            plot_config = sanitize_plot_config(plot_config, df)
            st.session_state[plot_config_state_key] = plot_config

            fig = build_scatter_figure(
                df,
                x_axis=plot_config["x_axis"],
                y_axis=plot_config["y_axis"],
                z_axis=plot_config.get("z_axis"),
                color_by=plot_config.get("color_by"),
                size_by=plot_config.get("size_by"),
                theme=theme,
                grid=plot_config.get("grid", True),
                text_size=plot_config.get("text_size", 14),
                axis_labels=plot_config.get("axis_labels"),
            )
            downloads: List[Dict[str, Any]] = []
            download_error: Optional[str] = None
            filename_base = f"{plot_config['y_axis']}_vs_{plot_config['x_axis']}"
            for fmt, label, mime in (
                ("png", "Save PNG", "image/png"),
                ("pdf", "Save PDF", "application/pdf"),
            ):
                try:
                    data = fig.to_image(format=fmt, scale=2)
                except Exception as exc:  # pragma: no cover - depends on kaleido
                    if download_error is None and "kaleido" in str(exc).lower():
                        download_error = "Install `kaleido` to enable PNG/PDF downloads."
                    elif download_error is None:
                        download_error = f"Unable to export plot as {fmt.upper()}."
                    data = None
                if data:
                    downloads.append(
                        {
                            "key": fmt,
                            "label": label,
                            "data": data,
                            "filename": f"{filename_base}.{fmt}",
                            "mime": mime,
                        }
                    )
            if downloads:
                download_error = None

            selected_id = plot_and_select(
                df,
                fig,
                downloads=downloads,
                download_error=download_error,
            )
        st.sidebar.divider()
        selected_id = st.sidebar.selectbox(
            "Jump directly to a structure",
            df["selection_id"],
            index=df["selection_id"].tolist().index(selected_id)
            if selected_id in df["selection_id"].values
            else 0,
            format_func=lambda sid: df.loc[df["selection_id"] == sid, "label"].iloc[0],
        )
        st.session_state["selected_id"] = selected_id
        viewer_container = right_col
    else:
        st.sidebar.header("Structure Selection")
        selected_id = st.sidebar.selectbox(
            "Choose a structure",
            df["selection_id"],
            format_func=lambda sid: df.loc[df["selection_id"] == sid, "label"].iloc[0],
            index=df["selection_id"].tolist().index(st.session_state.get("selected_id"))
            if st.session_state.get("selected_id") in df["selection_id"].values
            else 0,
        )
        st.session_state["selected_id"] = selected_id
        # Center the standalone viewer when no numeric properties are available
        _, viewer_container, _ = st.columns([1, 2, 1])

    slots_key = "viewer_slots"
    if slots_key not in st.session_state:
        st.session_state[slots_key] = []
    if selected_id and selected_id in df["selection_id"].values:
        slots = st.session_state[slots_key]
        if not slots or slots[-1] != selected_id:
            if selected_id in slots:
                slots.remove(selected_id)
            slots.append(selected_id)
            if len(slots) > 4:
                slots[:] = slots[-4:]
            st.session_state[slots_key] = slots

    if not st.session_state[slots_key]:
        st.info("Select a structure to view its 3D geometry.")
        return

    # Keyboard navigation (global)

    # Keyboard navigation via optional hotkeys component
    nav_evt = None
    if legacy_st_hotkeys is not None:
        hotkey_pressed = legacy_st_hotkeys([
            ("enter", "Next", ["enter", "ArrowRight", ">"]),
            ("backspace", "Previous", ["backspace", "ArrowLeft", "<"]),
        ])
        if hotkey_pressed == "Next":
            nav_evt = "next"
        elif hotkey_pressed == "Previous":
            nav_evt = "prev"
    elif hotkeys is not None:
        # streamlit-hotkeys >=0.5.0 API
        bindings = []
        for binding_id, help_label, keys in (
            ("next", "Next", ("Enter", "ArrowRight", ">")),
            ("prev", "Previous", ("Backspace", "ArrowLeft", "<")),
        ):
            for key_name in keys:
                bindings.append(
                    hotkeys.hk(binding_id, key=key_name, help=f"{help_label} ({key_name})")
                )
        manager_key = "molviewer-navigation"
        hotkeys.activate(bindings, key=manager_key)
        if hotkeys.pressed("next", key=manager_key):
            nav_evt = "next"
        elif hotkeys.pressed("prev", key=manager_key):
            nav_evt = "prev"
    # The rest of the logic to change the selection remains the same
    if nav_evt:
        should_rerun = False
        options = df["selection_id"].tolist()
        if options:
            current_id = st.session_state.get("selected_id", options[0])
            new_id = current_id
            try:
                current_index = options.index(current_id)
                if nav_evt == "prev":
                    new_index = (current_index - 1 + len(options)) % len(options)
                else:  # "next"
                    new_index = (current_index + 1) % len(options)
                new_id = options[new_index]
            except ValueError:
                new_id = options[0]
            if st.session_state.get("selected_id") != new_id:
                st.session_state["selected_id"] = new_id
                should_rerun = True

        if should_rerun:
            st.rerun()

    with viewer_container:

        if source_type == "XYZ Directory":
            selected_id = xyz_navbar(df, selected_id)
        else:
            selected_id = navigation_controls(df, selected_id)

        slots = list(st.session_state[slots_key])
        if not slots:
            st.info("No molecules queued for display. Select structures to populate the grid.")
            return

        tile_width = 420
        tile_height = 360
        rows = [slots[i : i + 2] for i in range(0, len(slots), 2)]
        for row in reversed(rows):
            cols = st.columns(len(row))
            for idx, slot_id in enumerate(row):
                with cols[idx]:
                    record = df.loc[df["selection_id"] == slot_id].iloc[0]
                    has_geometry = bool(record.get("has_geometry", True))
                    element_id = (
                        f"viewer-{slot_id.replace(':', '-').replace(' ', '_').replace('/', '_')}"
                    )

                    header_cols = st.columns([4, 1])
                    with header_cols[0]:
                        st.markdown(f"**{record.label}**")
                    with header_cols[1]:
                        if st.button("✕", key=f"close-{element_id}"):
                            updated = [sid for sid in st.session_state[slots_key] if sid != slot_id]
                            st.session_state[slots_key] = updated
                            if st.session_state.get("selected_id") == slot_id:
                                if updated:
                                    st.session_state["selected_id"] = updated[-1]
                                else:
                                    st.session_state.pop("selected_id", None)
                            st.experimental_rerun()

                    if not has_geometry:
                        st.info(
                            "No XYZ geometry available for this CSV entry. Nothing to show in the 3D viewer."
                        )
                        show_details(record, None, theme)
                        continue

                    atoms = get_atoms(record)
                    if atoms is None:
                        st.error("Unable to load atoms for the selected entry.")
                        continue
                    display_atoms = filter_hydrogens(
                        atoms, show_hydrogens=viewer_config["show_hydrogens"]
                    )
                    if len(display_atoms) == 0:
                        st.warning("No atoms remain after hiding hydrogens for this structure.")
                        continue

                    try:
                        quality_map = {label: factor for label, factor in SNAPSHOT_QUALITY_OPTIONS}
                        snapshot_factor = quality_map.get(
                            viewer_config.get("snapshot_quality", ""), 1
                        )
                        snapshot_params = {
                            "transparent": bool(viewer_config.get("snapshot_transparent", False)),
                            "factor": snapshot_factor,
                        }
                        html = render_ngl_view(
                            display_atoms,
                            record.label,
                            theme=theme,
                            sphere_radius=viewer_config["sphere_radius"],
                            bond_radius=viewer_config["bond_radius"],
                            interaction_mode=viewer_config["viewer_mode"],
                            height=tile_height,
                            width=tile_width,
                            representation_style=viewer_config["representation_style"],
                            label_mode=viewer_config["atom_label"],
                            snapshot=snapshot_params,
                            element_id=element_id,
                        )
                        st.components.v1.html(html, height=tile_height, width=tile_width)
                    except Exception as exc:  # pragma: no cover - defensive
                        st.error(str(exc))
                        continue

                    show_measurement_panel(
                        display_atoms,
                        viewer_config["viewer_mode"],
                        key_prefix=f"measure_{element_id}",
                    )
                    show_details(
                        record,
                        display_atoms if len(display_atoms) else atoms,
                        theme,
                    )

    with st.expander("Dataset distributions", expanded=False):
        with st.spinner("Computing dataset statistics..."):
            summary_df, elements_df, failures = compute_dataset_statistics(df)
        render_distribution_charts(summary_df, elements_df, theme, df)
        if failures:
            st.caption("Skipped structures without available geometries: " + ", ".join(failures))


if __name__ == "__main__":
    main()
