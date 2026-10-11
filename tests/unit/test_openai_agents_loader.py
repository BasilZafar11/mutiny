"""Project-local imports must follow the currently selected project."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from mutiny_openai_agents import loader


@pytest.fixture(autouse=True)
def isolate_project_imports(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(loader, "_project_root", None, raising=False)
    names = {"agent", "project_helpers", "project_helpers.values", "mutiny_project_adapter"}
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    for name in names:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def make_project(root: Path, marker: str, *, deferred: bool = False) -> Path:
    (root / ".mutiny").mkdir(parents=True)
    (root / "agent.py").write_text(f"VALUE = {marker!r}\n", encoding="utf-8")
    source = (
        "def create_adapter():\n    from agent import VALUE\n    return VALUE\n"
        if deferred
        else "from agent import VALUE\ndef create_adapter():\n    return VALUE\n"
    )
    (root / ".mutiny" / "adapter.py").write_text(source, encoding="utf-8")
    return root


@pytest.mark.parametrize("deferred", [False, True])
def test_sequential_projects_load_their_own_agent(tmp_path: Path, deferred: bool) -> None:
    first = make_project(tmp_path / "first", "A", deferred=deferred)
    second = make_project(tmp_path / "second", "B", deferred=deferred)
    assert loader.load_adapter_factory(first)() == "A"
    assert loader.load_adapter_factory(second)() == "B"
    assert loader.load_adapter_factory(first)() == "A"
    assert sys.path[0] == str(first.resolve())
    assert sys.modules["pytest"] is pytest


@pytest.mark.parametrize("namespace", [False, True])
def test_project_packages_are_not_reused(tmp_path: Path, namespace: bool) -> None:
    for marker in ("A", "B"):
        project = make_project(tmp_path / marker, marker)
        package = project / "project_helpers"
        package.mkdir()
        if not namespace:
            (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "values.py").write_text(f"VALUE = {marker!r}\n", encoding="utf-8")
        (project / "agent.py").write_text(
            "from project_helpers.values import VALUE\n", encoding="utf-8"
        )
        assert loader.load_adapter_factory(project)() == marker


def test_same_project_preserves_cached_imports(tmp_path: Path) -> None:
    project = make_project(tmp_path, "A")
    assert loader.load_adapter_factory(project)() == "A"
    agent = sys.modules["agent"]
    sys.path.insert(0, str(project / "other"))
    loader.ensure_project_on_path(project)
    assert sys.modules["agent"] is agent
    assert sys.modules["pytest"] is pytest
    assert sys.path[0] == str(project.resolve())
    assert sys.path.count(str(project.resolve())) == 1


def test_missing_agent_does_not_fall_back_to_previous_project(tmp_path: Path) -> None:
    first = make_project(tmp_path / "first", "A")
    second = make_project(tmp_path / "second", "B")
    (second / "agent.py").unlink()
    assert loader.load_adapter_factory(first)() == "A"
    # Restrict lookup so the checkout's sample agent cannot mask the missing file.
    sys.path[:] = [str(first.resolve())]
    with pytest.raises(ModuleNotFoundError, match="agent"):
        loader.load_adapter_factory(second)


def test_module_attribute_refs_follow_selected_project(tmp_path: Path) -> None:
    for marker in ("A", "B", "A"):
        project = tmp_path / marker
        if not project.exists():
            make_project(project, marker)
            (project / "agent.py").write_text(
                f"from types import SimpleNamespace\n"
                f"def support_agent():\n"
                f"    return SimpleNamespace(name={marker!r}, tools=[])\n",
                encoding="utf-8",
            )
        loader.ensure_project_on_path(project)
        assert loader.load_agent_from_ref("agent:support_agent").name == marker
