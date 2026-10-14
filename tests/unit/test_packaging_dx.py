"""Packaging and help text must describe the installed product accurately."""

from __future__ import annotations

import importlib.metadata
import runpy
import tomllib
from pathlib import Path

import pytest

from mutiny_cli.main import main

REPO = Path(__file__).resolve().parents[2]


def test_workspace_has_no_product_version_or_entrypoint() -> None:
    root = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert "project" not in root
    assert "mutiny-ai" in root["dependency-groups"]["dev"]


def test_api_version_reads_its_distribution_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def version(name: str) -> str:
        assert name == "mutiny-api"
        return "9.8.7"

    monkeypatch.setattr(importlib.metadata, "version", version)
    namespace = runpy.run_path(str(REPO / "apps/api/src/mutiny_api/__init__.py"))
    assert namespace["__version__"] == "9.8.7"


def test_api_without_metadata_is_explicitly_unreleased(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    namespace = runpy.run_path(str(REPO / "apps/api/src/mutiny_api/__init__.py"))
    assert namespace["__version__"] == "0.0.0+unreleased"


def test_cli_python_classifiers_match_ci_matrix() -> None:
    cli = tomllib.loads((REPO / "packages/mutiny_cli/pyproject.toml").read_text(encoding="utf-8"))
    classifiers = cli["project"]["classifiers"]
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers
    assert "Programming Language :: Python :: 3.13" not in classifiers
    assert "Programming Language :: Python :: 3.14" not in classifiers
    assert not any(dep.startswith("mutiny-api") for dep in cli["project"]["dependencies"])


@pytest.mark.parametrize("args", [["--help"], ["db", "--help"]])
def test_db_help_explains_api_requirement(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "requires mutiny-api" in output.lower()
    assert "source/workspace" in output


def test_db_help_works_without_api(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import builtins

    original = builtins.__import__

    def without_api(name: str, *args, **kwargs):
        if name == "mutiny_api" or name.startswith("mutiny_api."):
            raise ImportError("mutiny-api is not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_api)
    with pytest.raises(SystemExit) as exc:
        main(["db", "--help"])
    assert exc.value.code == 0
    assert "does not include mutiny-api" in capsys.readouterr().out


def test_web_readme_describes_hosted_development() -> None:
    readme = (REPO / "apps/web/README.md").read_text(encoding="utf-8")
    assert "Mutiny Hosted UI" in readme
    assert "python -m mutiny_api" in readme
    assert "MUTINY_API_URL" in readme
    assert "create-next-app" not in readme
