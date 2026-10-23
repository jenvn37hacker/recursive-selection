"""Exercise real version calculation/local Git, artifact coverage and draft recovery."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
from pathlib import Path
import subprocess
import tomllib
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("release_pipeline", ROOT / ".github/scripts/release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def command(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def commit(root, message):
    path = root / "changes.txt"
    path.write_text((path.read_text() if path.exists() else "") + message + "\n", encoding="utf-8")
    command(root, "add", ".")
    command(root, "commit", "-m", message)
    return command(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    def create(version="0.5.0", tagged="0.5.0"):
        root = tmp_path / "repo"
        root.mkdir()
        command(root, "init", "--initial-branch=master")
        command(root, "config", "user.name", "Release test")
        command(root, "config", "user.email", "test@example.invalid")
        command(root, "remote", "add", "origin", "https://github.com/example/release-test.git")
        config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        checked_out_version = tomllib.loads(config)["project"]["version"]
        config = config.replace(f'version = "{checked_out_version}"', f'version = "{tagged}"', 1)
        (root / "pyproject.toml").write_text(config, encoding="utf-8")
        package = root / "src/recursive_selection"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f'__version__ = "{tagged}"\n', encoding="utf-8")
        (package / "selector.py").write_text('"""Fixture implementation module."""\n', encoding="utf-8")
        commit(root, "chore: initial source")
        command(root, "tag", "v" + tagged)
        if version != tagged:
            (root / "pyproject.toml").write_text(config.replace(f'version = "{tagged}"', f'version = "{version}"', 1), encoding="utf-8")
            (package / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
            commit(root, "update sensor-features")
        return root
    return create


@pytest.mark.parametrize("message,expected,new_release", [
    ("fix: correct a calculation", "0.5.1", True),
    ("perf: reduce inference overhead", "0.5.1", True),
    ("feat: add a ranking method", "0.6.0", True),
    ("feat!: change the selector API", "1.0.0", True),
    ("docs: update examples", "0.5.0", False),
    ("ci: update the runner", "0.5.0", False),
    ("update code", "0.5.0", False),
])
def test_real_semantic_version_plans_do_not_change_git(repository, message, expected, new_release):
    root = repository()
    head = commit(root, message)
    before = command(root, "status", "--porcelain")
    plan = release.plan_release(root)
    assert (plan["version"], plan["release"]) == (expected, new_release)
    assert command(root, "rev-parse", "HEAD") == head
    assert command(root, "status", "--porcelain") == before == ""
    assert command(root, "tag", "--list") == "v0.5.0"


def test_highest_change_wins(repository):
    root = repository()
    commit(root, "fix: repair an edge case")
    commit(root, "feat: support another adapter")
    assert release.plan_release(root)["version"] == "0.6.0"


def test_fixtures_still_work_after_ci_updates_the_checkout_version(repository, tmp_path, monkeypatch):
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    current = tomllib.loads(config)["project"]["version"]
    shadow = tmp_path / "future-checkout"
    shadow.mkdir()
    (shadow / "pyproject.toml").write_text(
        config.replace(f'version = "{current}"', 'version = "1.9.0"', 1), encoding="utf-8")
    monkeypatch.setitem(globals(), "ROOT", shadow)
    root = repository()
    commit(root, "fix: verify releases after another version bump")
    assert release.plan_release(root)["version"] == "0.5.1"


def test_bootstrap_stamps_current_source_and_recovers_without_another_bump(repository):
    root = repository(version="0.5.0", tagged="0.4.0")
    original = command(root, "rev-parse", "HEAD")
    assert release.plan_release(root)["version"] == "0.5.0"
    prepared = release.prepare(root, event="push", trigger_sha=original, push=False)
    assert prepared["version"] == "0.5.0" and prepared["release"]
    assert release.project_version(root) == "0.5.0"
    assert release.tag_commit(root, "v0.5.0") == prepared["ref"] != original
    assert release.tag_commit(root, "v0.4.0") is not None
    assert (root / "CHANGELOG.md").is_file()
    retried = release.prepare(root, event="push", trigger_sha=original, push=False)
    assert retried == prepared
    assert command(root, "status", "--porcelain") == ""


def test_new_patch_updates_both_versions_and_tag(repository):
    root = repository()
    original = commit(root, "fix: repair selection history")
    result = release.prepare(root, event="push", trigger_sha=original, push=False)
    assert result["version"] == release.project_version(root) == "0.5.1"
    assert release.tag_commit(root, "v0.5.1") == result["ref"]


def test_superseded_push_does_not_release(repository):
    root = repository()
    old = commit(root, "fix: first correction")
    commit(root, "fix: a newer correction")
    result = release.prepare(root, event="push", trigger_sha=old, push=False)
    assert result["skip"] and not result["release"]
    assert command(root, "tag", "--list") == "v0.5.0"


def test_mismatched_source_versions_fail(repository):
    root = repository()
    (root / "src/recursive_selection/__init__.py").write_text('__version__ = "0.5.9"\n')
    with pytest.raises(ValueError, match="must agree"):
        release.plan_release(root)


def wheel_matrix(root, directory, *, leak_source=False):
    directory.mkdir()
    for py in ["cp311", "cp312", "cp313"]:
        for platform in ["win_amd64", "manylinux_2_28_x86_64", "macosx_13_0_x86_64", "macosx_11_0_arm64"]:
            path = directory / f"recursive_selection-0.5.0-{py}-{py}-{platform}.whl"
            suffix = ".pyd" if platform == "win_amd64" else ".so"
            with ZipFile(path, "w") as archive:
                archive.writestr("recursive_selection-0.5.0.dist-info/METADATA",
                    "Name: recursive-selection\nVersion: 0.5.0\n")
                archive.writestr("recursive_selection/__init__.py", '__version__ = "0.5.0"')
                archive.writestr("recursive_selection/selector." + py + suffix, b"native fixture")
                if leak_source:
                    archive.writestr("recursive_selection/selector.py", b"unexpected source")
    return directory


def test_complete_wheel_matrix_writes_checksums(repository, tmp_path):
    root = repository()
    directory = wheel_matrix(root, tmp_path / "wheels")
    files = release.verify_wheels(root, directory, "0.5.0")
    assert len(files) == 13
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert sha256((directory / name).read_bytes()).hexdigest() == digest


def test_linux_compatibility_tags_share_one_platform_slot():
    for tag in ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64"]:
        assert release.platform_slot(tag) == "linux"
    with pytest.raises(ValueError, match="Unsupported"):
        release.platform_slot("manylinux_2_28_aarch64")


@pytest.mark.parametrize("problem", ["missing", "source", "version", "abi", "module"])
def test_bad_or_incomplete_artifacts_never_pass(repository, tmp_path, problem):
    root = repository()
    directory = wheel_matrix(root, tmp_path / "wheels", leak_source=problem == "source")
    first = next(directory.glob("*.whl"))
    if problem == "missing":
        first.unlink()
    elif problem == "version":
        first.rename(first.with_name(first.name.replace("0.5.0", "0.5.1")))
    elif problem == "abi":
        # Directory iteration order differs between filesystems. Select the
        # interpreter we intend to corrupt instead of assuming it comes first.
        first = next(directory.glob("*-cp311-cp311-*.whl"))
        first.rename(first.with_name(first.name.replace("-cp311-cp311-", "-cp311-abi3-")))
    elif problem == "module":
        (root / "src/recursive_selection/extra.py").write_text("# additional implementation\n")
    with pytest.raises(ValueError):
        release.verify_wheels(root, directory, "0.5.0")
    assert not (directory / "SHA256SUMS").exists()


class FakeGitHub:
    def __init__(self):
        self.release = None
        self.assets = []
        self.calls = []
        self.fail_upload = None

    def request(self, method, path, data=None, **kwargs):
        self.calls.append((method, path, deepcopy(data)))
        if method == "GET" and path.startswith("/releases/tags/"):
            return deepcopy(self.release)
        if method == "POST" and path == "/releases":
            self.release = {"id": 1, "draft": True, "upload_url": "https://uploads.github.com/repos/test/repo/releases/1/assets{?name}",
                            "html_url": "https://github.com/test/repo/releases/tag/v0.5.0"}
            return deepcopy(self.release)
        if method == "GET" and "/assets?" in path:
            return deepcopy(self.assets)
        if method == "POST" and path.startswith("https://uploads.github.com/"):
            from urllib.parse import parse_qs, urlparse
            name = parse_qs(urlparse(path).query)["name"][0]
            if name == self.fail_upload:
                raise RuntimeError("Temporary upload failure")
            self.assets.append({"id": len(self.assets)+1, "name": name,
                                "digest": "sha256:" + sha256(data).hexdigest(), "state": "uploaded"})
            return self.assets[-1]
        if method == "DELETE":
            self.assets = [a for a in self.assets if a["id"] != int(path.rsplit("/", 1)[1])]
            return None
        if method == "PATCH":
            self.release["draft"] = data["draft"]
            return deepcopy(self.release)
        raise AssertionError((method, path))


def test_draft_failure_recovers_and_public_release_is_not_overwritten(tmp_path):
    files = [tmp_path / "first.whl", tmp_path / "second.whl"]
    for path in files:
        path.write_bytes(path.name.encode())
    client = FakeGitHub()
    client.fail_upload = "second.whl"
    with pytest.raises(RuntimeError, match="upload failure"):
        release.publish_assets(client, "v0.5.0", files)
    assert client.release["draft"]
    assert not any(method == "PATCH" for method, _, _ in client.calls)
    client.fail_upload = None
    release.publish_assets(client, "v0.5.0", files)
    assert not client.release["draft"]
    assert sum(method == "POST" and path == "/releases" for method, path, _ in client.calls) == 1
    client.calls.clear()
    release.publish_assets(client, "v0.5.0", files)
    assert all(method == "GET" for method, _, _ in client.calls)


def test_incomplete_public_release_fails_without_mutation(tmp_path):
    path = tmp_path / "missing.whl"
    path.write_bytes(b"wheel")
    client = FakeGitHub()
    client.release = {"id": 1, "draft": False, "html_url": "existing"}
    with pytest.raises(ValueError, match="incomplete public release"):
        release.publish_assets(client, "v0.5.0", [path])
    assert all(method == "GET" for method, _, _ in client.calls)


def test_authentication_is_never_sent_to_an_unexpected_host():
    with pytest.raises(ValueError, match="host"):
        release.GitHub("test/repo", "not-a-real-token").request("POST", "https://example.org/upload", b"file", binary=True)


def test_bad_upload_digest_prevents_publication(tmp_path):
    class CorruptedUpload(FakeGitHub):
        def request(self, method, path, data=None, **kwargs):
            result = super().request(method, path, data, **kwargs)
            if method == "POST" and path.startswith("https://uploads.github.com/"):
                self.assets[-1]["digest"] = "sha256:wrong"
            return result
    path = tmp_path / "wheel.whl"
    path.write_bytes(b"wheel")
    client = CorruptedUpload()
    with pytest.raises(ValueError, match="hash differs"):
        release.publish_assets(client, "v0.5.0", [path])
    assert client.release["draft"]
    assert not any(method == "PATCH" for method, _, _ in client.calls)
