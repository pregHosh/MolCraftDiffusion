"""Self-check for the `MolCraftDiff zoo` CLI.

The two assertions that matter operationally:

* `verify` exits non-zero while any asset still says ``redistribute: TODO``
  -- that is what makes `just check` catch a half-registered model instead
  of it being discovered a year later.
* `add` hashes a checkpoint and its sidecar into ONE asset directory,
  preserving the co-location contract `cli/generate.py` depends on.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml
from click.testing import CliRunner

from MolecularDiffusion import assets
from MolecularDiffusion.cli.zoo import zoo

HEADER = "# MolCraftDiffusion model zoo manifest.\n# Second header line.\n\n"

STUB = HEADER + textwrap.dedent(
    """
    schema: 1
    hosting:
      models: {repo: org/zoo, kind: model}
      datasets: {repo: org/zoo-data, kind: dataset}
      url: "https://example.invalid/{repo}/resolve/{rev}/{path}"
    assets:
      demo/pretrained:
        kind: checkpoint
        host: models
        license: MIT
        redistribute: yes
        files:
          - {path: model.ckpt, sha256: "%(sha)s", size: 5}
      demo/data:
        kind: dataset
        host: datasets
        license: CC0
        redistribute: yes
        files:
          - {path: demo.db, sha256: "%(sha)s", size: 5}
      other/data:
        alias: demo/data
      data/corpus/variant:
        kind: dataset
        host: datasets
        license: CC0
        redistribute: yes
        files:
          - {path: corpus.db, sha256: "%(sha)s", size: 5}
    models:
      demo:
        family: test-family
        task_type: diffusion_demo
        tags: [alpha, beta]
        variants:
          default:
            checkpoint: demo/pretrained
            data: demo/data
            configs: [demo_generate.yaml]
      shares:
        family: test-family
        variants:
          default: {checkpoint: demo/pretrained, data: other/data}
      corpususer:
        family: test-family
        variants:
          default: {checkpoint: demo/pretrained, data: data/corpus/variant}
    """
)

_SHA = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


@pytest.fixture
def env(tmp_path, monkeypatch):
    manifest = tmp_path / "zoo.yaml"
    manifest.write_text(STUB % {"sha": _SHA}, encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(assets, "MANIFEST", manifest)
    monkeypatch.setenv("MOLCRAFT_ASSETS", str(cache))
    assets.manifest.cache_clear()
    yield tmp_path, cache, manifest
    assets.manifest.cache_clear()


def _run(*args):
    return CliRunner().invoke(zoo, list(args))


def test_list_shows_models(env):
    result = _run("list")
    assert result.exit_code == 0
    assert "demo" in result.output
    assert "test-family" in result.output


def test_list_data_shows_corpora_and_their_readers(env):
    """`--data` is the only way to discover a corpus -- it is not a model."""
    result = _run("list", "--data")
    assert result.exit_code == 0
    assert "data/corpus/variant" in result.output
    assert "corpususer" in result.output  # USED BY column
    assert "demo/pretrained" not in result.output  # weights are not corpora


def test_info_reports_licences_and_cache_state(env):
    result = _run("info", "demo")
    assert result.exit_code == 0
    assert "MIT" in result.output
    assert "not fetched" in result.output
    assert "demo_generate.yaml" in result.output


def test_fetch_dry_run_dedupes_aliases(env):
    """`shares` uses other/data, an alias of demo/data -- count it once."""
    result = _run("fetch", "--model", "shares", "--dry-run")
    assert result.exit_code == 0
    assert "across 2 assets" in result.output
    assert "other/data" not in result.output
    assert "demo/data" in result.output


def test_fetch_requires_a_selection(env):
    result = _run("fetch")
    assert result.exit_code != 0
    assert "Nothing selected" in result.output


def test_fetch_from_config_reads_asset_refs(env):
    tmp_path, _, _ = env
    cfg = tmp_path / "demo_generate.yaml"
    cfg.write_text(
        "chkpt_directory: ${asset:demo/pretrained}\n"
        "pocket_db: ${asset:demo/data/demo.db}\n",
        encoding="utf-8",
    )
    result = _run("fetch", "--config", str(cfg), "--dry-run")
    assert result.exit_code == 0
    assert "demo/pretrained" in result.output
    assert "demo/data" in result.output


def test_fetch_from_unmigrated_config_says_so(env):
    tmp_path, _, _ = env
    cfg = tmp_path / "old.yaml"
    cfg.write_text("chkpt_directory: docs/model_integrations/x\n", "utf-8")
    result = _run("fetch", "--config", str(cfg))
    assert result.exit_code != 0
    assert "unmigrated" in result.output


def test_verify_fails_while_redistribute_is_todo(env):
    """The gate that keeps the zoo honest as it grows."""
    _, _, manifest = env
    doc = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    doc["assets"]["demo/pretrained"]["redistribute"] = "TODO"
    manifest.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assets.manifest.cache_clear()

    result = _run("verify", "--all")
    assert result.exit_code != 0
    assert "UNAUDITED" in result.output


def test_add_registers_sidecar_into_one_asset(env):
    tmp_path, _, manifest = env
    ckpt = tmp_path / "midi_pretrained.ckpt"
    stat = tmp_path / "edm_stat.pkl"
    db = tmp_path / "midi_smoke.db"
    for f in (ckpt, stat, db):
        f.write_bytes(b"hello")

    result = _run(
        "add",
        "midi",
        "--ckpt",
        str(ckpt),
        "--ckpt",
        str(stat),
        "--data",
        str(db),
        "--task-type",
        "diffusion_midi",
        "--family",
        "bond-generating-diffusion",
    )
    assert result.exit_code == 0, result.output

    text = manifest.read_text(encoding="utf-8")
    assert text.startswith(
        "# MolCraftDiffusion model zoo manifest."
    )  # header kept
    doc = yaml.safe_load(text)

    entry = doc["assets"]["midi/pretrained"]
    names = [f["path"] for f in entry["files"]]
    assert names == ["midi_pretrained.ckpt", "edm_stat.pkl"]  # one directory
    assert all(f["sha256"] == _SHA for f in entry["files"])
    assert entry["redistribute"] == "TODO"

    assert doc["models"]["midi"]["task_type"] == "diffusion_midi"
    assert doc["models"]["midi"]["variants"]["default"]["data"] == "midi/data"

    # and the freshly added model must fail the audit gate
    assets.manifest.cache_clear()
    assert _run("verify", "--all").exit_code != 0


def test_add_requires_something_to_register(env):
    result = _run("add", "empty")
    assert result.exit_code != 0
    assert "at least one" in result.output


def _audit(manifest, key, value):
    """Rewrite one asset's `redistribute` field in the stub manifest."""
    doc = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    doc["assets"][key]["redistribute"] = value
    manifest.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assets.manifest.cache_clear()


def _source_tree(root, rel):
    """A source directory holding one file whose sha256 is `_SHA`."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("hello", encoding="utf-8")
    return root


def test_publish_stages_by_host_and_stops_at_stage_only(env):
    tmp_path, _cache, _manifest = env
    src = _source_tree(tmp_path / "src", "demo/pretrained/model.ckpt")
    _source_tree(src, "demo/data/demo.db")
    out = tmp_path / "staging"

    result = _run(
        "publish",
        "demo",
        "--source",
        str(src),
        "--out",
        str(out),
        "--stage-only",
    )
    assert result.exit_code == 0, result.output
    # the HuggingFace layout: <host>/<key>/<path>, not the cache layout
    ckpt = out / "models" / "demo" / "pretrained" / "model.ckpt"
    data = out / "datasets" / "demo" / "data" / "demo.db"
    assert ckpt.read_text(encoding="utf-8") == "hello"
    assert data.read_text(encoding="utf-8") == "hello"
    assert "Staged only" in result.output


def test_publish_dry_run_writes_nothing(env):
    tmp_path, _cache, _manifest = env
    src = _source_tree(tmp_path / "src", "demo/pretrained/model.ckpt")
    out = tmp_path / "staging"

    result = _run(
        "publish",
        "demo/pretrained",
        "--source",
        str(src),
        "--out",
        str(out),
        "--dry-run",
    )
    assert result.exit_code == 0, result.output
    assert not out.exists()


def test_publish_never_pushes_without_confirmation(env):
    tmp_path, _cache, _manifest = env
    src = _source_tree(tmp_path / "src", "demo/pretrained/model.ckpt")
    out = tmp_path / "staging"

    result = CliRunner().invoke(
        zoo,
        [
            "publish",
            "demo/pretrained",
            "--source",
            str(src),
            "--out",
            str(out),
        ],
        input="n\n",
    )
    assert result.exit_code != 0
    assert "org/zoo" in result.output  # the repo it would have pushed to


def test_publish_refuses_while_redistribute_is_todo(env):
    tmp_path, _cache, manifest = env
    src = _source_tree(tmp_path / "src", "demo/pretrained/model.ckpt")
    _audit(manifest, "demo/pretrained", "TODO")
    out = tmp_path / "staging"

    result = _run(
        "publish",
        "demo/pretrained",
        "--source",
        str(src),
        "--out",
        str(out),
        "--stage-only",
    )
    assert result.exit_code != 0
    assert "Unaudited" in result.output
    assert not out.exists()  # aborts before writing anything


def test_publish_withholds_non_redistributable(env):
    tmp_path, _cache, manifest = env
    src = _source_tree(tmp_path / "src", "demo/pretrained/model.ckpt")
    _audit(manifest, "demo/pretrained", False)
    out = tmp_path / "staging"

    result = _run(
        "publish",
        "demo/pretrained",
        "--source",
        str(src),
        "--out",
        str(out),
        "--stage-only",
    )
    assert result.exit_code != 0
    assert "withheld" in result.output
    assert not out.exists()


def test_publish_without_a_source_says_how(env):
    tmp_path, _cache, _manifest = env
    result = _run(
        "publish",
        "demo/pretrained",
        "--out",
        str(tmp_path / "staging"),
        "--stage-only",
    )
    assert result.exit_code != 0
    assert "--source" in result.output


def test_publish_rejects_a_source_whose_hash_drifted(env):
    tmp_path, _cache, _manifest = env
    src = tmp_path / "src"
    (src / "demo" / "pretrained").mkdir(parents=True)
    (src / "demo" / "pretrained" / "model.ckpt").write_text(
        "world", encoding="utf-8"
    )

    result = _run(
        "publish",
        "demo/pretrained",
        "--source",
        str(src),
        "--out",
        str(tmp_path / "staging"),
        "--stage-only",
    )
    assert result.exit_code != 0
    assert "does not match the manifest" in result.output


def test_publish_dedupes_aliases(env):
    tmp_path, _cache, _manifest = env
    src = _source_tree(tmp_path / "src", "demo/data/demo.db")
    _source_tree(src, "demo/pretrained/model.ckpt")

    result = _run(
        "publish",
        "shares",  # reaches demo/data through the `other/data` alias
        "--source",
        str(src),
        "--out",
        str(tmp_path / "staging"),
        "--stage-only",
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("demo/data") == 1
    assert "other/data" not in result.output
