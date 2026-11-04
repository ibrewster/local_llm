import asyncio
import json

import pytest

from MLXOllama import config
from MLXOllama import local_tools


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_FILES_ROOT", str(tmp_path))
    policy_path = tmp_path / "policy.json"
    monkeypatch.setattr(
        config, "LOCAL_FILES_POLICY_PATH", str(policy_path), raising=False
    )
    policy_path.write_text(json.dumps({
        "include": [str(tmp_path)],
        "deny": [str(policy_path)],
        "sensitive": [],
    }), encoding="utf-8")
    return tmp_path


def test_list_local_directory(workspace):
    (workspace / "folder").mkdir()
    (workspace / "note.txt").write_text("hello", encoding="utf-8")

    result = json.loads(asyncio.run(local_tools.list_local_directory(".")))

    assert result["path"] == "."
    assert {entry["name"]: entry["type"] for entry in result["entries"]} == {
        "folder": "directory",
        "note.txt": "file",
    }
    assert result["truncated"] is False


def test_list_local_directory_caps_results(workspace):
    for index in range(3):
        (workspace / f"{index}.txt").touch()

    result = json.loads(asyncio.run(local_tools.list_local_directory(".", max_entries=2)))

    assert len(result["entries"]) == 2
    assert result["truncated"] is True


def test_find_local_files_matches_names_and_limits_results(workspace):
    (workspace / "one.py").touch()
    nested = workspace / "nested"
    nested.mkdir()
    (nested / "two.py").touch()
    (nested / "skip.txt").touch()

    result = json.loads(asyncio.run(local_tools.find_local_files("*.py")))
    limited = json.loads(asyncio.run(
        local_tools.find_local_files("*.py", max_results=1)
    ))

    assert {match["path"] for match in result["matches"]} == {
        "one.py",
        "nested/two.py",
    }
    assert result["truncated"] is False
    assert len(limited["matches"]) == 1
    assert limited["truncated"] is True


def test_find_local_files_does_not_follow_symlink_directories(workspace):
    outside = workspace.parent / "outside"
    outside.mkdir()
    (outside / "secret.py").touch()
    (workspace / "external").symlink_to(outside, target_is_directory=True)

    result = json.loads(asyncio.run(local_tools.find_local_files("*.py")))

    assert result["matches"] == []


def test_read_local_file_and_reject_large_file(workspace):
    (workspace / "note.txt").write_text("hello", encoding="utf-8")

    assert asyncio.run(local_tools.read_local_file("note.txt")) == "hello"
    with pytest.raises(ValueError, match="exceeds"):
        asyncio.run(local_tools.read_local_file("note.txt", max_bytes=4))


@pytest.mark.parametrize("tool", ["list_local_directory", "read_local_file"])
def test_read_tools_reject_paths_outside_workspace(workspace, tool):
    outside = workspace.parent / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    args = {"path": "../outside.txt"}
    if tool == "list_local_directory":
        args["path"] = ".."

    with pytest.raises(PermissionError, match="not included"):
        asyncio.run(getattr(local_tools, tool)(**args))


def test_read_file_rejects_symlink_outside_workspace(workspace):
    outside = workspace.parent / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside)

    with pytest.raises(PermissionError, match="not included"):
        asyncio.run(local_tools.read_local_file("link.txt"))


def test_write_local_file_is_create_only(workspace):
    (workspace / "new").mkdir()

    result = asyncio.run(local_tools.write_local_file("new/note.txt", "hello"))

    assert result == "Created new/note.txt (5 bytes)"
    assert (workspace / "new/note.txt").read_text(encoding="utf-8") == "hello"
    with pytest.raises(FileExistsError):
        asyncio.run(local_tools.write_local_file("new/note.txt", "replacement"))
    assert (workspace / "new/note.txt").read_text(encoding="utf-8") == "hello"


def test_write_local_file_rejects_escaping_and_missing_directories(workspace):
    with pytest.raises(PermissionError, match="not included"):
        asyncio.run(local_tools.write_local_file("../outside.txt", "no"))
    with pytest.raises(FileNotFoundError):
        asyncio.run(local_tools.write_local_file("missing/note.txt", "no"))


def test_file_policy_supports_multiple_roots_and_runtime_edits(workspace):
    external = workspace.parent / "external"
    external.mkdir()
    (external / "note.txt").write_text("outside default root", encoding="utf-8")
    policy_path = workspace / "policy.json"
    policy = {
        "include": [str(workspace), str(external)],
        "deny": [],
    }
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    assert asyncio.run(local_tools.read_local_file(str(external / "note.txt"))) == (
        "outside default root"
    )
    listed = json.loads(asyncio.run(local_tools.list_local_directory(str(external))))
    assert {entry["name"] for entry in listed["entries"]} == {"note.txt"}
    asyncio.run(local_tools.write_local_file(str(external / "created.txt"), "created"))
    assert (external / "created.txt").read_text(encoding="utf-8") == "created"
    results = json.loads(asyncio.run(local_tools.find_local_files("note.txt")))
    assert str((external / "note.txt").resolve()) in {
        item["path"] for item in results["matches"]
    }

    policy["deny"] = ["*.txt"]
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(PermissionError, match="denied"):
        asyncio.run(local_tools.read_local_file(str(external / "note.txt")))
    results = json.loads(asyncio.run(local_tools.find_local_files("note.txt")))
    assert str((external / "note.txt").resolve()) not in {
        item["path"] for item in results["matches"]
    }


def test_sensitive_files_are_blocked_and_hidden(workspace):
    (workspace / ".env").write_text("TOKEN=value", encoding="utf-8")
    (workspace / "id_rsa").write_text("private key", encoding="utf-8")
    (workspace / "ordinary.txt").write_text("visible", encoding="utf-8")

    with pytest.raises(PermissionError, match="sensitive"):
        asyncio.run(local_tools.read_local_file(".env"))

    listing = json.loads(asyncio.run(local_tools.list_local_directory(".")))
    assert {entry["name"] for entry in listing["entries"]} == {"ordinary.txt"}
    search = json.loads(asyncio.run(local_tools.find_local_files("*")))
    assert {item["path"] for item in search["matches"]} == {"ordinary.txt"}


def test_local_file_tools_are_registered():
    assert {
        "list_local_directory",
        "find_local_files",
        "read_local_file",
        "write_local_file",
    } <= local_tools.LOCAL_TOOLS.keys()
