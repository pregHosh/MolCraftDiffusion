"""Chemiscop-style Streamlit app that renders molecules with NGL."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import re

from io import StringIO

import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ase import Atoms
from ase.db import connect
from ase.io import read as ase_read
from ase.io import write as ase_write

try:  # Optional dependency for scatter picking
    from streamlit_plotly_events import plotly_events  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency
    plotly_events = None


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

    return pd.DataFrame(records).convert_dtypes()


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


def _structure_search_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in ("label", "identifier", "path") if col in df.columns]


def find_structure_suggestions(
    df: pd.DataFrame, query: str, limit: int = 15
) -> tuple[pd.DataFrame, int, bool]:
    if df.empty:
        return df, 0, False
    columns = _structure_search_columns(df)
    if not columns:
        subset = df.head(limit)
        return subset, len(df), False

    cleaned_query = query.strip().lower() if query else ""
    if not cleaned_query:
        subset = df.head(min(limit, len(df)))
        return subset, len(df), False

    mask = pd.Series(False, index=df.index)
    for col in columns:
        series = df[col].fillna("").astype(str).str.lower()
        mask = mask | series.str.contains(cleaned_query)

    matches = df[mask]
    if not matches.empty:
        return matches.head(limit), len(matches), False

    scores: list[tuple[float, Any]] = []
    for idx, row in df.iterrows():
        text_parts = []
        for col in columns:
            value = row.get(col)
            if pd.notna(value):
                text_parts.append(str(value))
        if not text_parts:
            continue
        combined = " ".join(text_parts).lower()
        ratio = SequenceMatcher(None, cleaned_query, combined).ratio()
        if ratio >= 0.35:
            scores.append((ratio, idx))

    if not scores:
        return matches, 0, False

    scores.sort(key=lambda item: item[0], reverse=True)
    ranked_indices = [idx for _, idx in scores[:limit]]
    return df.loc[ranked_indices], 0, True


def structure_option_label(record: pd.Series) -> str:
    label = str(record.get("label", record.get("selection_id", "Unknown")))
    components = []
    identifier = record.get("identifier")
    if pd.notna(identifier) and str(identifier) not in {label, ""}:
        components.append(f"ID {identifier}")
    path_value = record.get("path")
    if pd.notna(path_value):
        try:
            components.append(Path(str(path_value)).name)
        except Exception:
            components.append(str(path_value))
    source = record.get("source")
    if pd.notna(source):
        components.append(str(source))
    return " · ".join([label, *components]) if components else label


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
    screenshot_factor: int = 2,
    screenshot_transparent: bool = False,
    label_mode: Optional[str] = None,
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
        "Line": "line",
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
    elif style_key == "line":
        style_params.update(
            {
                "linewidth": max(1, int(round(bond_radius * 30))),
                "multipleBond": "off",
            }
        )

    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "structure"
    screenshot_cfg = {
        "factor": max(int(screenshot_factor), 1),
        "transparent": bool(screenshot_transparent),
        "background": theme.background,
        "filename": safe_label,
    }

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
        "screenshot": screenshot_cfg,
        "zoomStep": 0.18,
    }

    return f"""
<div style=\"display:flex;justify-content:center;\">
  <div id=\"ngl-stage\" style=\"width:{width}px;height:{height}px;position:relative;background:{theme.background};\"></div>
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
    var controls = node.querySelector('.chemiscop-zoom-controls');
    if (controls) return controls;

    controls = document.createElement('div');
    controls.className = 'chemiscop-zoom-controls';
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

    function saveImage() {{
      var screenshot = cfg.screenshot || {{}};
      var params = {{
        factor: screenshot.factor || 1,
        antialias: true,
        trim: false,
        transparent: !!screenshot.transparent,
      }};
      if (!params.transparent) {{
        params.backgroundColor = screenshot.background || cfg.theme.background;
      }}
      stage.makeImage(params).then(function(blob) {{
        var filename = (screenshot.filename || 'structure') + '.png';
        var url = URL.createObjectURL(blob);
        var link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        setTimeout(function() {{
          URL.revokeObjectURL(url);
          link.remove();
        }}, 100);
      }}).catch(function(err) {{
        console.error('Screenshot failed', err);
      }});
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
    var snapshot = makeButton('PNG', 'Save PNG screenshot', function() {{
      try {{ saveImage(); }} catch (err) {{ console.warn('Screenshot action failed', err); }}
    }});

    controls.appendChild(zoomIn);
    controls.appendChild(zoomOut);
    controls.appendChild(reset);
    controls.appendChild(snapshot);
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
    if (!canvas || canvas.dataset.chemiscopPinchBound) return;
    canvas.dataset.chemiscopPinchBound = '1';

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

  function overlayContainer() {{
    var node = document.getElementById('ngl-stage');
    if (!node) return null;
    var overlay = node.querySelector('.chemiscop-overlay');
    if (overlay) return overlay;
    overlay = document.createElement('div');
    overlay.className = 'chemiscop-overlay';
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


def build_scatter_figure(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    z_axis: Optional[str],
    color_by: Optional[str],
    size_by: Optional[str],
    theme: ThemeConfig,
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
    )
    return fig


def pick_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BASE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


def pick_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BASE_COLUMNS and not pd.api.types.is_numeric_dtype(df[c])]


def sidebar_controls(df: pd.DataFrame, *, enable_scatter: bool) -> Dict[str, Any]:
    st.sidebar.header("Data")
    x_axis = y_axis = z_axis = color_by = size_by = None

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
        color_choice = st.sidebar.selectbox(
            "Color by",
            ["None", *numeric_cols, *categorical_cols],
            index=0,
        )
        color_by = None if color_choice == "None" else color_choice
        size_choice = st.sidebar.selectbox("Size by", ["Uniform", *numeric_cols], index=0)
        size_by = None if size_choice == "Uniform" else size_choice
    else:
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
            ["Ball + Stick", "Licorice", "Spacefilling", "Line"],
            index=0,
            help="Choose how the molecule is drawn in the 3D viewer.",
        )
        screenshot_factor_label = st.selectbox(
            "Screenshot resolution",
            ["Screen (1×)", "HD (2×)", "4K (4×)"],
            index=1,
            help="Scale the saved PNG relative to the on-screen size.",
        )
        screenshot_factor_map = {
            "Screen (1×)": 1,
            "HD (2×)": 2,
            "4K (4×)": 4,
        }
        screenshot_factor = screenshot_factor_map[screenshot_factor_label]
        screenshot_transparent = st.checkbox(
            "Transparent screenshot background",
            value=False,
            help="If enabled, saved PNGs use an alpha channel instead of the themed background.",
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
            help="Uncheck to omit hydrogens from the viewer and measurements.",
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
        "screenshot_factor": screenshot_factor,
        "screenshot_transparent": screenshot_transparent,
        "atom_label": None if atom_label == "None" else atom_label,
        "viewer_mode": viewer_mode,
        "show_hydrogens": show_hydrogens,
    }


def plot_and_select(df: pd.DataFrame, fig: Any) -> Optional[str]:
    st.subheader("Data Exploration")
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
        st.plotly_chart(fig, use_container_width=True)
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


def summarize_atoms(atoms: Atoms) -> Dict[str, Any]:
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


def show_details(record: pd.Series, atoms: Atoms) -> None:
    st.markdown(f"**Selected structure:** `{record.label}`")
    st.markdown("**Basic information**")
    st.dataframe(pd.DataFrame([summarize_atoms(atoms)]), use_container_width=True)
    metadata = {k: v for k, v in record.items() if k not in BASE_COLUMNS}
    if metadata:
        st.markdown("**Metadata**")
        st.dataframe(pd.DataFrame([metadata]), use_container_width=True)


def navigation_controls(df: pd.DataFrame, selected_id: Optional[str]) -> Optional[str]:
    options = df["selection_id"].tolist()
    if not options:
        return selected_id
    if selected_id not in options:
        selected_id = options[0]
    search_box_key = "chemiscop_search_box"
    prefill_key = "chemiscop_search_prefill"
    if search_box_key not in st.session_state:
        st.session_state[search_box_key] = ""
    if prefill_key in st.session_state:
        st.session_state[search_box_key] = st.session_state.pop(prefill_key)
    search_query = st.text_input(
        "Search structures",
        key=search_box_key,
        placeholder="Type a label, identifier, or path to jump…",
        help="Live suggestions update as you type. Press enter to apply the top result.",
    )

    suggestions_df, match_count, used_fallback = find_structure_suggestions(df, search_query, limit=15)
    suggestion_ids = suggestions_df["selection_id"].tolist()
    labels_map = {row["selection_id"]: structure_option_label(row) for _, row in suggestions_df.iterrows()}

    if search_query:
        if match_count > 0:
            if match_count > len(suggestion_ids):
                st.caption(f"Showing first {len(suggestion_ids)} of {match_count} matches.")
            else:
                st.caption(f"{match_count} match{'es' if match_count != 1 else ''} found.")
        elif used_fallback and suggestion_ids:
            st.caption("No exact matches found — showing the closest suggestions instead.")
        elif not suggestion_ids:
            st.caption("No matching structures. Adjust your search to try again.")

    if search_query and match_count == 1 and suggestion_ids:
        only_id = suggestion_ids[0]
        if only_id != selected_id:
            selected_id = only_id
            selected_row = df.loc[df["selection_id"] == selected_id]
            if not selected_row.empty:
                label_value = selected_row.iloc[0].get("label")
                if pd.notna(label_value):
                    st.session_state[prefill_key] = str(label_value)
                    st.experimental_rerun()

    if selected_id not in suggestion_ids:
        suggestion_ids.insert(0, selected_id)
    suggestion_ids = list(dict.fromkeys(suggestion_ids))

    if selected_id not in labels_map:
        selected_row = df.loc[df["selection_id"] == selected_id]
        if not selected_row.empty:
            labels_map[selected_id] = structure_option_label(selected_row.iloc[0])
        else:
            labels_map[selected_id] = str(selected_id)

    if suggestion_ids:
        default_index = suggestion_ids.index(selected_id) if selected_id in suggestion_ids else 0
        suggestion_choice = st.selectbox(
            "Suggested matches",
            suggestion_ids,
            index=default_index,
            format_func=lambda sid: labels_map.get(sid, str(sid)),
            key="chemiscop_search_suggestions",
        )
        if suggestion_choice != selected_id:
            selected_id = suggestion_choice
            selected_row = df.loc[df["selection_id"] == selected_id]
            if not selected_row.empty:
                label_value = selected_row.iloc[0].get("label")
                if pd.notna(label_value):
                    st.session_state[prefill_key] = str(label_value)
                    st.experimental_rerun()

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


def render_distribution_charts(summary_df: pd.DataFrame, elements_df: pd.DataFrame, theme: ThemeConfig) -> None:
    if summary_df.empty and elements_df.empty:
        st.info("Unable to compute distributions for the current dataset.")
        return
    choice = st.selectbox(
        "Distribution",
        ["Number of atoms", "Total mass", "Atom types"],
        index=0,
        key="distribution_choice_ngl",
    )
    color_sequence = [theme.highlight]
    if choice == "Number of atoms" and not summary_df.empty:
        fig = px.histogram(
            summary_df,
            x="num_atoms",
            nbins=min(30, max(5, summary_df["num_atoms"].nunique())),
            template=theme.plot_template,
            color_discrete_sequence=color_sequence,
        )
        fig.update_layout(xaxis_title="Number of atoms", yaxis_title="Count")
    elif choice == "Total mass" and not summary_df.empty:
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


def main() -> None:
    st.set_page_config(page_title="Chemiscop (NGL)", layout="wide")
    theme_name = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=1)
    theme = THEMES[theme_name]
    inject_theme_css(theme)
    st.title("Chemiscop × NGL: Structure-Property Explorer")

    st.sidebar.header("Data Source")
    source_type = st.sidebar.radio("Source", ["XYZ Directory", "ASE Database"], index=0)
    try:
        if source_type == "XYZ Directory":
            xyz_dir = st.sidebar.text_input("XYZ directory", value="visualizer/test_xyz")
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

    if "__index" not in df.columns:
        df["__index"] = np.arange(len(df))

    has_numeric = bool(pick_numeric_columns(df))
    config = sidebar_controls(df, enable_scatter=has_numeric)

    if has_numeric:
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
        viewer_container = st.container()

    if selected_id is None:
        st.info("Select a structure to view its 3D geometry.")
        return

    with viewer_container:
        selected_id = navigation_controls(df, selected_id)
        record = df.loc[df["selection_id"] == selected_id].iloc[0]
        atoms = get_atoms(record)
        if atoms is None:
            st.error("Unable to load atoms for the selected entry.")
            return
        display_atoms = filter_hydrogens(atoms, show_hydrogens=config["show_hydrogens"])
        if len(display_atoms) == 0:
            st.warning(
                "No atoms remain after hiding hydrogens for this structure. Showing original metadata below."
            )
            if len(atoms) > 0:
                show_details(record, atoms)
        else:
            try:
                html = render_ngl_view(
                    display_atoms,
                    record.label,
                    theme=theme,
                    sphere_radius=config["sphere_radius"],
                    bond_radius=config["bond_radius"],
                    interaction_mode=config["viewer_mode"],
                    height=600,
                    width=700,
                    representation_style=config["representation_style"],
                    screenshot_factor=config["screenshot_factor"],
                    screenshot_transparent=config["screenshot_transparent"],
                    label_mode=config["atom_label"],
                )
                st.components.v1.html(html, height=600, width=700)
            except Exception as exc:  # pragma: no cover - defensive
                st.error(str(exc))
                return

            show_measurement_panel(
                display_atoms, config["viewer_mode"], key_prefix=f"measure_{selected_id}"
            )
            show_details(record, display_atoms)

    with st.expander("Dataset distributions", expanded=False):
        with st.spinner("Computing dataset statistics..."):
            summary_df, elements_df, failures = compute_dataset_statistics(df)
        render_distribution_charts(summary_df, elements_df, theme)
        if failures:
            st.caption("Skipped structures without available geometries: " + ", ".join(failures))


if __name__ == "__main__":
    main()
