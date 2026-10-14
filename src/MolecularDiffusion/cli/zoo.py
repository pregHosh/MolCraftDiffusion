"""Zoo CLI: discover, fetch and register pretrained weights and datasets.

`MolCraftDiff zoo list` shows what exists, `fetch` pulls only what you ask
for, and `add` registers a newly integrated model so the zoo stays current
instead of turning back into a pile of scattered files.
"""

import difflib
import json
import logging
import os
import shutil
from pathlib import Path

import click

from MolecularDiffusion import assets

logger = logging.getLogger(__name__)


def _human(size):
    value = float(size or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return (
                f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
            )
        value /= 1024
    return f"{value:.1f} TB"


def _models():
    return assets.manifest().get("models") or {}


def _asset_map():
    return assets.manifest().get("assets") or {}


def _assets_of(model_name, variant=None):
    """Every asset key a model (or one of its variants) needs."""
    model = _models().get(model_name)
    if model is None:
        close = ", ".join(sorted(_models())[:5])
        raise click.ClickException(
            f"Unknown model {model_name!r}. Try: MolCraftDiff zoo list"
            + (f"\nKnown: {close}" if close else "")
        )
    variants = model.get("variants") or {}
    if variant is not None:
        if variant not in variants:
            known = ", ".join(sorted(variants))
            raise click.ClickException(
                f"Unknown variant {variant!r} for {model_name!r}. "
                f"Known: {known}"
            )
        variants = {variant: variants[variant]}
    keys = []
    for spec in variants.values():
        for field in ("checkpoint", "data"):
            if spec.get(field):
                keys.append(spec[field])
        keys.extend((spec.get("extra") or {}).values())
    # dedupe, preserve order
    return list(dict.fromkeys(keys))


def _cached(key):
    return assets.local_path(key).exists()


@click.group()
def zoo():
    """Pretrained weights, datasets and example configs."""


def _users_of(key):
    """Who reads this asset: model variants, plus bundled example configs."""
    users = {
        name for name, model in _models().items() if key in _assets_of(name)
    }
    examples = Path(assets.__file__).parent / "configs" / "examples"
    for cfg in sorted(examples.glob("*.yaml")) if examples.is_dir() else []:
        try:
            refs = _config_assets(cfg)
        except click.ClickException:
            continue
        for ref in refs:
            # refs may name a file inside an asset -- resolve to its owner.
            try:
                owner, _ = assets._resolve_key(ref)  # noqa: SLF001
            except (KeyError, ValueError):
                continue
            if owner == key:
                users.add(cfg.stem.replace("_generate", ""))
    return sorted(users)


def _list_data(fetched, as_json):
    rows = []
    for key, info in sorted(_asset_map().items()):
        if not key.startswith(("data/", "inputs/")):
            continue
        have = _cached(key)
        if fetched and not have:
            continue
        rows.append(
            {
                "name": key,
                "size": assets.total_size(key),
                "license": info.get("license", ""),
                "redistribute": bool(info.get("redistribute")),
                "used_by": _users_of(key),
                "cached": have,
            }
        )

    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo(
            "No corpora match. Cache root: " + str(assets.assets_root())
        )
        return

    width = max(len(r["name"]) for r in rows)
    click.echo(f"{'':2}{'CORPUS':<{width}}  {'SIZE':>9}  USED BY")
    for row in rows:
        mark = "*" if row["cached"] else " "
        users = ", ".join(row["used_by"]) or "-"
        note = "" if row["redistribute"] else "  (build locally)"
        click.echo(
            f"{mark:2}{row['name']:<{width}}  {_human(row['size']):>9}  "
            f"{users}{note}"
        )
    click.echo(f"\n* = already in {assets.assets_root()}")


@zoo.command("list")
@click.option("--tag", "-t", multiple=True, help="Filter by tag (repeatable)")
@click.option("--family", "-f", default=None, help="Filter by model family")
@click.option(
    "--fetched", is_flag=True, help="Only models already in the cache"
)
@click.option(
    "--data",
    "as_data",
    is_flag=True,
    help="List data corpora instead of models",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Machine-readable output"
)
def list_cmd(tag, family, fetched, as_json, as_data):
    """List models in the zoo, or corpora with ``--data``."""
    if as_data:
        _list_data(fetched, as_json)
        return
    rows = []
    for name, model in sorted(_models().items()):
        tags = model.get("tags") or []
        if tag and not set(tag) <= set(tags):
            continue
        if family and model.get("family") != family:
            continue
        keys = _assets_of(name)
        have = all(_cached(k) for k in keys) if keys else False
        if fetched and not have:
            continue
        rows.append(
            {
                "name": name,
                "family": model.get("family", ""),
                "tags": tags,
                "variants": sorted(model.get("variants") or {}),
                "size": sum(assets.total_size(k) for k in keys),
                "cached": have,
            }
        )

    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo("No models match. Cache root: " + str(assets.assets_root()))
        return

    width = max(len(r["name"]) for r in rows)
    click.echo(f"{'':2}{'MODEL':<{width}}  {'SIZE':>9}  FAMILY / TAGS")
    for row in rows:
        mark = "*" if row["cached"] else " "
        tags = ", ".join(row["tags"])
        variants = row["variants"]
        extra = f"  [{len(variants)} variants]" if len(variants) > 1 else ""
        click.echo(
            f"{mark:2}{row['name']:<{width}}  {_human(row['size']):>9}  "
            f"{row['family']}{extra}"
            + (f"\n{'':{width + 15}}{tags}" if tags else "")
        )
    click.echo(f"\n* = already in {assets.assets_root()}")


@zoo.command("info")
@click.argument("target")
def info_cmd(target):
    """Show details for MODEL or MODEL/VARIANT."""
    name, _, variant = target.partition("/")
    model = _models().get(name)
    if model is None:
        raise click.ClickException(
            f"Unknown model {name!r}. Try: MolCraftDiff zoo list"
        )

    click.echo(f"{name}  ({model.get('family', 'unknown family')})")
    if model.get("summary"):
        click.echo(f"\n  {' '.join(model['summary'].split())}\n")
    if model.get("tags"):
        click.echo(f"  tags     : {', '.join(model['tags'])}")
    if model.get("task_type"):
        click.echo(f"  task_type: {model['task_type']}")

    for vname, spec in sorted((model.get("variants") or {}).items()):
        if variant and vname != variant:
            continue
        click.echo(f"\n  variant: {vname}")
        keys = []
        for field in ("checkpoint", "data"):
            if spec.get(field):
                keys.append((field, spec[field]))
        for label, key in (spec.get("extra") or {}).items():
            keys.append((label, key))
        for label, key in keys:
            info = assets.entry(key)
            state = "cached" if _cached(key) else "not fetched"
            lic = info.get("license", "licence unstated")
            note = (
                "" if info.get("redistribute", True) else "  (build locally)"
            )
            click.echo(
                f"    {label:<16} {key:<24} "
                f"{_human(assets.total_size(key)):>9}  {state}  {lic}{note}"
            )
        if spec.get("configs"):
            click.echo(f"    {'configs':<16} {', '.join(spec['configs'])}")

    click.echo(f"\n  Fetch:  MolCraftDiff zoo fetch --model {name}")


@zoo.command("fetch")
@click.argument("names", nargs=-1)
@click.option(
    "--model",
    "-m",
    "model_name",
    default=None,
    help="Fetch every asset this model needs",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Fetch exactly what this config references",
)
@click.option("--all", "fetch_all", is_flag=True, help="Fetch everything")
@click.option("--force", is_flag=True, help="Redownload even if already valid")
@click.option("--dry-run", is_flag=True, help="Report size and licences only")
def fetch_cmd(names, model_name, config_path, fetch_all, force, dry_run):
    """Download assets into the cache.

    Nothing is downloaded unless you ask for it. Pick a granularity:
    a single asset, one model, everything one config needs, or --all.
    """
    keys = list(names)
    if model_name:
        keys += (
            _assets_of(*model_name.split("/", 1))
            if "/" in model_name
            else _assets_of(model_name)
        )
    if config_path:
        keys += _config_assets(Path(config_path))
    if fetch_all:
        keys += [k for k, v in _asset_map().items() if "alias" not in v]
    if not keys:
        raise click.ClickException(
            "Nothing selected. Give an asset name, or use --model / "
            "--config / --all."
        )

    # Resolve aliases so a shared corpus is counted and fetched once.
    resolved = []
    for key in keys:
        try:
            owner, _ = assets._resolve_key(key)  # noqa: SLF001
        except (KeyError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        if owner not in resolved:
            resolved.append(owner)

    total = sum(assets.total_size(k) for k in resolved)
    blocked = [
        k for k in resolved if assets.entry(k).get("redistribute") is False
    ]

    for key in resolved:
        info = assets.entry(key)
        state = "cached" if _cached(key) else "fetch"
        if info.get("redistribute") is False:
            state = "BUILD LOCALLY"
        click.echo(
            f"  {state:<14} {key:<26} {_human(assets.total_size(key)):>9}  "
            f"{info.get('license', 'licence unstated')}"
        )
    click.echo(f"\n  total: {_human(total)} across {len(resolved)} assets")

    if dry_run:
        click.echo("\nDry run. Re-run without --dry-run to download.")
        return
    if blocked:
        click.echo(
            "\nNot redistributable, so not downloaded: "
            + ", ".join(blocked)
            + "\nBuild each locally:  MolCraftDiff zoo recipe <asset>"
        )

    for key in resolved:
        if key in blocked:
            continue
        click.echo(f"fetching {key} ...")
        try:
            assets.fetch(key, force=force)
        except Exception as exc:
            raise click.ClickException(f"{key}: {exc}") from exc
    click.echo(f"done -> {assets.assets_root()}")


def _config_assets(path):
    """Every ${asset:...} reference in a config file."""
    import re  # noqa: PLC0415

    text = Path(path).read_text(encoding="utf-8")
    found = re.findall(r"\$\{asset:([^}]+)\}", text)
    if not found:
        raise click.ClickException(
            f"{path} contains no ${{asset:...}} references. "
            "It may be an unmigrated config using literal paths."
        )
    return found


@zoo.command("verify")
@click.argument("names", nargs=-1)
@click.option(
    "--all", "verify_all", is_flag=True, help="Verify everything cached"
)
def verify_cmd(names, verify_all):
    """Re-hash cached assets, and check the manifest is complete.

    Exits non-zero on any mismatch or any unaudited `redistribute: TODO`,
    so a half-registered model fails `just check` instead of being
    discovered a year later.
    """
    failures = 0

    for key, info in sorted(_asset_map().items()):
        if "alias" in info:
            continue
        if str(info.get("redistribute", "TODO")).upper() == "TODO":
            click.echo(f"  UNAUDITED  {key}: redistribute is still TODO")
            failures += 1
        if not info.get("license"):
            click.echo(f"  UNAUDITED  {key}: no license field")
            failures += 1

    targets = list(names)
    if verify_all or not targets:
        targets = [
            k
            for k, v in _asset_map().items()
            if "alias" not in v and assets.local_path(k).exists()
        ]

    for key in targets:
        for rel, status in assets.verify(key):
            if status == "ok":
                click.echo(f"  ok         {key}/{rel}")
            else:
                click.echo(f"  {status.upper():<10} {key}/{rel}")
                if status != "missing" or names:
                    failures += 1

    if failures:
        raise click.ClickException(f"{failures} problem(s) found")
    click.echo("all good")


@zoo.command("path")
@click.argument("name")
def path_cmd(name):
    """Print an asset's local path, for $(...) interop."""
    try:
        click.echo(assets.resolve(name))
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@zoo.command("config")
@click.argument("name", required=False)
@click.argument("dest", type=click.Path(), default=".")
def config_cmd(name, dest):
    """Copy a bundled example config out so you can edit it.

    With no NAME, lists what is available. The bundled configs live inside
    the installed package, so copying one by hand means finding that
    directory first -- this does it for you. A config of the same name in
    your working directory takes precedence over the bundled one.
    """
    examples = Path(assets.__file__).parent / "configs" / "examples"
    if not examples.is_dir():
        raise click.ClickException(f"No bundled examples at {examples}")

    if not name:
        for f in sorted(examples.glob("*.yaml")):
            click.echo(f"  {f.name}")
        click.echo(f"\nCopy one:  MolCraftDiff zoo config <name> [dest]")
        return

    src = examples / (name if name.endswith(".yaml") else f"{name}.yaml")
    if not src.exists():
        near = difflib.get_close_matches(
            src.name, [f.name for f in examples.glob("*.yaml")], n=3
        )
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        raise click.ClickException(f"No bundled config {src.name!r}.{hint}")

    target = Path(dest)
    if target.is_dir():
        target = target / src.name
    if target.exists():
        raise click.ClickException(
            f"{target} already exists -- not overwriting"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    click.echo(f"copied -> {target}")
    click.echo(f"edit it, then:  MolCraftDiff generate {target}")


@zoo.command("bundle")
@click.option(
    "--out",
    type=click.Path(),
    default="zoo_bundle",
    help="Directory to build the archive in",
)
@click.option(
    "--archive/--no-archive",
    default=True,
    help="Also write a .tar.gz next to the tree",
)
def bundle_cmd(out, archive):
    """Build a self-contained, citable snapshot of the zoo.

    Lays assets out in the CACHE layout, not the HuggingFace one, so the
    unpacked archive is usable with a single `export MOLCRAFT_ASSETS=$(pwd)`
    -- no unpacking logic and no special format. Only `redistribute: true`
    assets are included; the manifest travels with them so the archive
    records exactly what it holds.
    """
    import shutil  # noqa: PLC0415
    import tarfile  # noqa: PLC0415

    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)

    staged = withheld = 0
    total = 0
    for key, info in sorted(_asset_map().items()):
        if "alias" in info:
            continue
        if not info.get("redistribute"):
            withheld += 1
            continue
        src_dir = assets.local_path(key)
        if not src_dir.exists():
            raise click.ClickException(
                f"{key} is not in the local cache. Fetch everything first:\n"
                f"  MolCraftDiff zoo fetch --all"
            )
        for spec in info.get("files") or []:
            src = src_dir / spec["path"]
            dest = root / key / spec["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                try:
                    os.link(src, dest)
                except OSError:
                    shutil.copy2(src, dest)
            total += spec["size"]
        staged += 1

    shutil.copy2(assets.MANIFEST, root / "zoo.yaml")
    examples = Path(assets.__file__).parent / "configs" / "examples"
    if examples.is_dir():
        shutil.copytree(examples, root / "cfg_examples", dirs_exist_ok=True)

    (root / "README.md").write_text(
        "# MolCraftDiffusion — zoo snapshot\n\n"
        "Unpack, point the platform at this directory, and run:\n\n"
        "```bash\nexport MOLCRAFT_ASSETS=$(pwd)\n"
        "MolCraftDiff generate examples/kgdiff_generate.yaml\n```\n\n"
        "This layout IS the cache layout, so no import step is needed.\n\n"
        f"Contains {staged} assets ({_human(total)}). `zoo.yaml` is the "
        "manifest, with a sha256 for every file; `MolCraftDiff zoo verify "
        "--all` checks them.\n\n"
        f"{withheld} asset(s) are not included because their upstream licence "
        "does not permit redistribution; `MolCraftDiff zoo recipe <asset>` "
        "prints how to build each one locally.\n",
        encoding="utf-8",
    )

    click.echo(f"  {staged} assets ({_human(total)}), {withheld} withheld")
    click.echo(f"  -> {root}")
    if archive:
        tgz = root.with_suffix(".tar.gz")
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(root, arcname=root.name)
        click.echo(f"  -> {tgz} ({_human(tgz.stat().st_size)})")


@zoo.command("recipe")
@click.argument("name")
def recipe_cmd(name):
    """Show how to build a non-redistributable asset locally."""
    try:
        info = assets.entry(name)
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    up = info.get("upstream") or {}
    click.echo(f"{name}")
    if info.get("reason"):
        click.echo(f"  not bundled: {info['reason']}")
    for label, field in (
        ("upstream repo", "repo"),
        ("commit", "commit"),
        ("download", "download"),
        ("source", "source"),
        ("convert with", "recipe"),
        ("convert with", "script"),
    ):
        if up.get(field):
            click.echo(f"  {label:<14} {up[field]}")
    click.echo(f"  place result in {assets.local_path(name)}")
    for spec in info.get("files") or []:
        click.echo(f"    {spec['path']}  sha256 {spec.get('sha256', '?')}")
    click.echo(f"  then: MolCraftDiff zoo verify {name}")


@zoo.command("add")
@click.argument("name")
@click.option(
    "--ckpt",
    type=click.Path(exists=True),
    multiple=True,
    help="Checkpoint file or directory (repeatable)",
)
@click.option(
    "--data",
    "data_paths",
    type=click.Path(exists=True),
    multiple=True,
    help="Dataset file or directory (repeatable)",
)
@click.option(
    "--config",
    "config_paths",
    type=click.Path(exists=True),
    multiple=True,
    help="Example config (repeatable)",
)
@click.option("--variant", default="default", show_default=True)
@click.option("--task-type", default=None, help="The task_type string")
@click.option("--family", default=None)
@click.option("--tag", multiple=True)
def add_cmd(
    name, ckpt, data_paths, config_paths, variant, task_type, family, tag
):
    """Register a newly integrated model's assets in zoo.yaml.

    Hashes every file and writes the entry, leaving `license` and
    `redistribute` as TODO for you to fill in -- `zoo verify` then fails
    until they are, which is what stops the zoo drifting.
    """
    import yaml  # noqa: PLC0415

    if not ckpt and not data_paths:
        raise click.ClickException("Give at least one --ckpt or --data.")

    doc = assets.manifest()
    doc.setdefault("assets", {})
    doc.setdefault("models", {})

    def _entry(paths, kind, host):
        files = []
        for item in paths:
            p = Path(item)
            members = (
                sorted(f for f in p.rglob("*") if f.is_file())
                if p.is_dir()
                else [p]
            )
            for f in members:
                rel = f.relative_to(p) if p.is_dir() else Path(f.name)
                files.append(
                    {
                        "path": str(rel),
                        "sha256": assets._sha256(f),  # noqa: SLF001
                        "size": f.stat().st_size,
                    }
                )
        return {
            "kind": kind,
            "host": host,
            "rev": "main",
            "files": files,
            "license": "TODO",
            "redistribute": "TODO",
        }

    spec = {}
    if ckpt:
        key = f"{name}/pretrained"
        doc["assets"][key] = _entry(ckpt, "checkpoint", "models")
        spec["checkpoint"] = key
    if data_paths:
        key = f"{name}/data"
        doc["assets"][key] = _entry(data_paths, "dataset", "datasets")
        spec["data"] = key
    if config_paths:
        spec["configs"] = [Path(c).name for c in config_paths]

    model = doc["models"].setdefault(name, {})
    if family:
        model["family"] = family
    if task_type:
        model["task_type"] = task_type
    if tag:
        model["tags"] = list(tag)
    model.setdefault("variants", {})[variant] = spec

    # safe_dump cannot round-trip comments, so preserve the file's leading
    # header block (the "how to use this manifest" docs) by hand. Per-entry
    # inline comments are lost -- that prose belongs in the model card.
    header = ""
    if assets.MANIFEST.exists():
        lines = assets.MANIFEST.read_text(encoding="utf-8").splitlines(True)
        keep = []
        for line in lines:
            if line.strip() and not line.lstrip().startswith("#"):
                break
            keep.append(line)
        header = "".join(keep)

    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    assets.MANIFEST.write_text(header + body, encoding="utf-8")
    assets.manifest.cache_clear()

    click.echo(f"registered {name} in {assets.MANIFEST}")
    click.echo("Now fill in license / redistribute / upstream, then run:")
    click.echo("  MolCraftDiff zoo verify --all")


def _publish_source(key, spec, roots):
    """Find the on-disk file backing one manifest entry, sha256-confirmed.

    The manifest stores no absolute paths -- they would be machine-specific
    -- so the source is re-derived: the local cache first, then any
    ``--source`` root. The hash check makes a wrong guess impossible rather
    than merely unlikely.
    """
    rel = spec["path"]
    probes = [assets.local_path(key) / rel]
    for root in roots:
        base = Path(root)
        probes += [base / key / rel, base / rel, base / Path(rel).name]
    candidates = list(dict.fromkeys(probes))  # a flat `rel` repeats itself

    stale = None
    for cand in candidates:
        if not cand.is_file():
            continue
        if spec.get("size") and cand.stat().st_size != spec["size"]:
            stale = stale or cand  # wrong size, so skip hashing 12 GB
            continue
        if assets._sha256(cand) == spec["sha256"]:  # noqa: SLF001
            return cand
        stale = stale or cand

    tried = "\n    ".join(str(c) for c in candidates)
    drift = (
        f"\n  {stale} exists but does not match the manifest -- it changed "
        "since `zoo add`."
        if stale
        else ""
    )
    raise click.ClickException(
        f"{key}/{rel}: no matching source file.{drift}\n"
        "  Pass --source <dir> pointing at the files you registered.\n"
        f"  Looked in:\n    {tried}"
    )


@zoo.command("publish")
@click.argument("names", nargs=-1)
@click.option(
    "--all",
    "publish_all",
    is_flag=True,
    help="Publish every redistributable asset",
)
@click.option(
    "--source",
    "sources",
    type=click.Path(exists=True, file_okay=False),
    multiple=True,
    help="Directory to look for the asset's files in (repeatable)",
)
@click.option(
    "--out",
    type=click.Path(),
    default=".zoo-staging",
    show_default=True,
    help="Staging tree to build",
)
@click.option(
    "--stage-only", is_flag=True, help="Build the staging tree and stop"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report what would be staged, write nothing",
)
@click.option(
    "--message",
    default="Add MolCraftDiffusion zoo assets",
    show_default=True,
    help="Upload commit message",
)
@click.option("--yes", is_flag=True, help="Skip the upload confirmation")
def publish_cmd(
    names, publish_all, sources, out, stage_only, dry_run, message, yes
):
    """Stage registered assets and push them to HuggingFace.

    NAMES are asset keys or model names. Only `redistribute: true` assets are
    staged, an unaudited `TODO` aborts before anything is written, and the
    push asks first -- it is outward-facing and not undoable. `--dry-run` and
    `--stage-only` never touch the remote.
    """
    keys = []
    for name in names:
        if name in _asset_map():
            keys.append(name)
        elif name in _models():
            keys += _assets_of(name)
        else:
            close = difflib.get_close_matches(name, list(_asset_map()), 3)
            raise click.ClickException(
                f"Unknown asset or model {name!r}."
                + (f" Close: {', '.join(close)}" if close else "")
                + "\nTry: MolCraftDiff zoo list"
            )
    if publish_all:
        keys += [k for k, v in _asset_map().items() if "alias" not in v]
    if not keys:
        raise click.ClickException(
            "Nothing selected. Give an asset or model name, or use --all."
        )

    # Resolve aliases so a shared corpus is staged and uploaded once.
    resolved = []
    for key in keys:
        try:
            owner, _ = assets._resolve_key(key)  # noqa: SLF001
        except (KeyError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        if owner not in resolved:
            resolved.append(owner)

    todo = [
        k
        for k in resolved
        if str(assets.entry(k).get("redistribute", "TODO")).upper() == "TODO"
    ]
    if todo:
        raise click.ClickException(
            "Unaudited, refusing to stage:\n  "
            + "\n  ".join(todo)
            + "\nSettle `license` and `redistribute: true|false` on each "
            "first, then: MolCraftDiff zoo verify --all"
        )

    selected = [
        k for k in resolved if assets.entry(k).get("redistribute") is True
    ]
    for key in resolved:
        if key not in selected:
            click.echo(f"  withheld {key:<30} (redistribute: no)")
    if not selected:
        raise click.ClickException(
            "Nothing to publish -- every asset selected is withheld."
        )

    root = Path(out)
    hosting = assets.manifest().get("hosting") or {}
    hosts = {}
    total = 0
    for key in selected:
        info = assets.entry(key)
        host = info.get("host", "models")
        for spec in info.get("files") or []:
            src = _publish_source(key, spec, sources)
            total += spec.get("size") or src.stat().st_size
            if dry_run:
                continue
            dest = root / host / key / spec["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            try:
                os.link(src, dest)  # instant, no multi-GB copy
            except OSError:
                shutil.copy2(src, dest)  # different filesystem
        hosts.setdefault(host, []).append(key)
        click.echo(
            f"  {'would stage' if dry_run else 'staged':<11} {key:<30} "
            f"{_human(assets.total_size(key)):>9} -> {host}/"
        )

    click.echo(f"\n  {len(selected)} asset(s), {_human(total)} -> {root}")
    if dry_run:
        click.echo("Dry run. Nothing written, nothing uploaded.")
        return
    if stage_only:
        click.echo("Staged only. Re-run without --stage-only to push.")
        return

    repos = ", ".join(
        (hosting.get(h) or {}).get("repo", f"<no hosting.{h}.repo>")
        for h in hosts
    )
    if not yes:
        click.confirm(
            f"Push {len(selected)} asset(s) ({_human(total)}) to {repos}?",
            abort=True,
        )

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
    except ImportError as exc:  # maintainer-only path, not a runtime dep
        raise click.ClickException(
            "Uploading needs huggingface_hub: pip install huggingface_hub"
        ) from exc

    token = assets.hf_token()
    if token is None:
        raise click.ClickException(
            "No HuggingFace token found. Set HF_TOKEN, or run `hf auth login`."
        )
    api = HfApi(token=token)
    for host in hosts:
        cfg = hosting.get(host) or {}
        repo = cfg.get("repo")
        if not repo:
            raise click.ClickException(f"zoo.yaml has no hosting.{host}.repo")
        kind = cfg.get("kind", "model")
        click.echo(f"uploading {host}/ -> {repo} ({kind})")
        # ponytail: pushes whatever sits in the host folder, leftovers from an
        # earlier run included -- use a fresh --out to scope a single push.
        api.upload_folder(
            folder_path=str(root / host),
            repo_id=repo,
            repo_type=kind,
            commit_message=message,
        )
        prefix = "datasets/" if kind == "dataset" else ""
        click.echo(f"  done: https://huggingface.co/{prefix}{repo}")
