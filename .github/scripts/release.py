"""Version planning and verified GitHub releases; never publish to PyPI."""
import argparse
import ast
from email.parser import Parser
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

VERSION = re.compile(r"\d+\.\d+\.\d+\Z")
PROJECT = "recursive-group-selection"


def run(root, *args):
    result = subprocess.run(args, cwd=root, text=True, capture_output=True, timeout=180)
    if result.returncode:
        message = result.stderr or result.stdout
        token = os.environ.get("GH_TOKEN")
        if token:
            message = message.replace(token, "***")
        raise RuntimeError(f"{args[0]} failed: {message.strip()}")
    return result.stdout.strip()


def git(root, *args):
    return run(root, "git", *args)


def project_version(root):
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = config["project"]["version"]
    if not VERSION.fullmatch(version):
        raise ValueError("Only stable major.minor.patch versions are released.")
    tree = ast.parse((root / "src/recursive_group_selection/__init__.py").read_text(encoding="utf-8"))
    declared = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)]
    if declared != [version]:
        raise ValueError("pyproject.toml and __version__ must agree before releasing.")
    return version


def tag_commit(root, tag):
    result = subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
                            cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def psr(root, *options):
    return run(root, sys.executable, "-m", "semantic_release", "version", *options)


def predicted(root, *options):
    # Printing never stamps files, pushes commits/tags or contacts a release API.
    value = psr(root, "--print", "--no-push", "--no-vcs-release", *options).splitlines()[-1]
    if not VERSION.fullmatch(value):
        raise ValueError(f"Unexpected semantic-release version output: {value!r}")
    return value


def plan_release(root):
    current = project_version(root)
    head = git(root, "rev-parse", "HEAD")
    existing = tag_commit(root, f"v{current}")
    if existing == head:
        # Retrying a failed matrix/publication must reuse its existing version.
        return dict(version=current, tag=f"v{current}", ref=head, release=True,
                    force=None, existing=True)
    last = psr(root, "--print-last-released", "--no-push", "--no-vcs-release")
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    bootstrap = config.get("tool", {}).get("release_pipeline", {}).get("bootstrap_version")
    force = None
    if current == bootstrap and existing is None and last == "0.4.0":
        force = "--minor"
        next_version = predicted(root, force)
        if next_version != current:
            raise ValueError("Bootstrap must resolve source 0.5.0 from the existing v0.4.0 tag.")
    else:
        if last != current:
            raise ValueError("Source version differs from the last tag; pull CI's version commit first.")
        next_version = predicted(root)
    return dict(version=next_version, tag=f"v{next_version}", ref=head,
                release=force is not None or Version(next_version) > Version(current),
                force=force, existing=False)


def prepare(root, *, event=None, trigger_sha=None, push=True):
    event = event or os.environ.get("RELEASE_EVENT", "workflow_dispatch")
    trigger_sha = trigger_sha or os.environ.get("RELEASE_TRIGGER_SHA")
    head = git(root, "rev-parse", "HEAD")
    current = project_version(root)
    if event == "push" and trigger_sha and head != trigger_sha:
        parent = git(root, "rev-parse", "HEAD^")
        message = git(root, "log", "-1", "--format=%s")
        recovery = (parent == trigger_sha and tag_commit(root, f"v{current}") == head
                    and message == f"chore(release): {current} [skip ci]")
        if not recovery:
            return dict(version=current, tag=f"v{current}", ref=head, release=False, skip=True)
    result = plan_release(root)
    if result["release"] and not result["existing"]:
        options = ["--skip-build", "--no-vcs-release"]
        if result["force"]:
            options.append(result["force"])
        if not push:
            options.append("--no-push")
        psr(root, *options)
        result["ref"] = git(root, "rev-parse", "HEAD")
        if (project_version(root) != result["version"]
                or tag_commit(root, result["tag"]) != result["ref"]):
            raise ValueError("Stamped source, release commit and tag do not match.")
    return {key: result[key] for key in ("version", "tag", "ref", "release")} | {"skip": False}


def write_outputs(result):
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            for key, value in result.items():
                stream.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")
    print(json.dumps(result, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(f"Version: **{result['version']}** | release: **{result['release']}**\n\n")
            stream.write(f"Pinned source commit: `{result['ref']}`\n")


def platform_slot(platform):
    if platform == "win_amd64":
        return "windows"
    if re.fullmatch(r"(?:manylinux_\d+_\d+|manylinux1|manylinux2010|manylinux2014)_x86_64", platform):
        return "linux"
    if platform.startswith("macosx_") and platform.endswith("_x86_64"):
        return "macos-intel"
    if platform.startswith("macosx_") and platform.endswith("_arm64"):
        return "macos-arm64"
    raise ValueError(f"Unsupported wheel platform: {platform}")


def verify_wheels(root, directory, version):
    if not VERSION.fullmatch(version) or project_version(root) != version:
        raise ValueError("Release version must match the checked-out source.")
    wheels = sorted(directory.glob("*.whl"))
    expected = {(py, platform) for py in ("cp311", "cp312", "cp313")
                for platform in ("windows", "linux", "macos-intel", "macos-arm64")}
    slots = set()
    modules = {p.relative_to(root / "src").with_suffix("").as_posix()
               for p in (root / "src/recursive_group_selection").rglob("*.py") if p.name != "__init__.py"}
    for wheel in wheels:
        name, actual_version, build, tags = parse_wheel_filename(wheel.name)
        if canonicalize_name(name) != PROJECT or str(actual_version) != version or build:
            raise ValueError(f"Unexpected project/version/build tag: {wheel.name}")
        wheel_slots = {(t.interpreter, platform_slot(t.platform)) for t in tags}
        if len(wheel_slots) != 1 or not wheel_slots.issubset(expected) or slots.intersection(wheel_slots):
            raise ValueError(f"Unexpected or duplicate Python/platform wheel: {wheel.name}")
        if any(t.abi != t.interpreter for t in tags):
            raise ValueError(f"Wheel ABI must match its CPython version: {wheel.name}")
        slots.update(wheel_slots)
        with ZipFile(wheel) as archive:
            names = archive.namelist()
            metadata_paths = [n for n in names if n.endswith(".dist-info/METADATA")]
            if len(metadata_paths) != 1:
                raise ValueError(f"Wheel must contain exactly one METADATA: {wheel.name}")
            metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode("utf-8"))
            if canonicalize_name(metadata["Name"]) != PROJECT or metadata["Version"] != version:
                raise ValueError(f"Wheel METADATA differs from the release: {wheel.name}")
            if any(n.endswith(".py") and not n.endswith("/__init__.py") for n in names):
                raise ValueError(f"Implementation source leaked into native wheel: {wheel.name}")
            for module in modules:
                if not any(n.startswith(module + ".") and n.endswith((".pyd", ".so")) for n in names):
                    raise ValueError(f"Missing compiled module {module} in {wheel.name}")
    if len(wheels) != 12 or slots != expected:
        raise ValueError(f"Need all 12 Python/platform wheels; found {len(wheels)} covering {len(slots)} slots.")
    checksums = directory / "SHA256SUMS"
    checksums.write_text("".join(f"{sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in wheels), encoding="utf-8")
    return wheels + [checksums]


class GitHub:
    def __init__(self, repository, token):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or not token:
            raise ValueError("A repository and GH_TOKEN are required for publication.")
        self.base = f"https://api.github.com/repos/{repository}"
        self.token = token

    def request(self, method, path, data=None, *, binary=False, missing_ok=False):
        url = path if path.startswith("https://") else self.base + path
        # Never send the token to an unexpected upload/download host.
        if not url.startswith(("https://api.github.com/", "https://uploads.github.com/")):
            raise ValueError("Unexpected GitHub API host.")
        body = data if binary else json.dumps(data).encode() if data is not None else None
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "recursive-group-selection-release"}
        if body is not None:
            headers["Content-Type"] = "application/octet-stream" if binary else "application/json"
        try:
            with urlopen(Request(url, data=body, headers=headers, method=method), timeout=120) as response:
                content = response.read()
                return json.loads(content) if content else None
        except HTTPError as error:
            if error.code == 404 and missing_ok:
                return None
            raise RuntimeError(f"GitHub API {method} failed with HTTP {error.code}.") from None


def publish_assets(client, tag, files):
    route = "/releases/tags/" + quote(tag, safe="")
    release = client.request("GET", route, missing_ok=True)
    expected_names = {path.name for path in files}
    if release and not release["draft"]:
        assets = client.request("GET", f"/releases/{release['id']}/assets?per_page=100")
        names = {asset["name"] for asset in assets if asset.get("state") == "uploaded"}
        if not expected_names.issubset(names):
            raise ValueError("An incomplete public release already exists; it will not be overwritten.")
        print("Release is already published; leaving its assets unchanged:", release["html_url"])
        return release["html_url"]
    if release is None:
        release = client.request("POST", "/releases", {
            "tag_name": tag, "name": tag, "draft": True, "generate_release_notes": True})
    assets = {a["name"]: a for a in client.request("GET", f"/releases/{release['id']}/assets?per_page=100")}
    upload_url = release["upload_url"].split("{", 1)[0]
    for path in files:
        content = path.read_bytes()
        old = assets.get(path.name)
        if old and old.get("digest") == "sha256:" + sha256(content).hexdigest():
            continue
        if old:
            # Replacement is restricted to a still-unpublished draft.
            client.request("DELETE", f"/releases/assets/{old['id']}")
        client.request("POST", upload_url + "?" + urlencode({"name": path.name}), content, binary=True)
    uploaded = client.request("GET", f"/releases/{release['id']}/assets?per_page=100")
    if not expected_names.issubset({a["name"] for a in uploaded}):
        raise ValueError("Release is missing assets; it remains a draft.")
    for asset in uploaded:
        if asset["name"] in expected_names and asset.get("state") != "uploaded":
            raise ValueError("An asset is not fully uploaded; release remains a draft.")
        if asset["name"] in expected_names:
            path = next(path for path in files if path.name == asset["name"])
            if asset.get("digest") not in (None, "sha256:" + sha256(path.read_bytes()).hexdigest()):
                raise ValueError("Uploaded asset hash differs; release remains a draft.")
            if asset.get("size", path.stat().st_size) != path.stat().st_size:
                raise ValueError("Uploaded asset size differs; release remains a draft.")
    published = client.request("PATCH", f"/releases/{release['id']}", {"draft": False, "make_latest": "true"})
    print("Published:", published["html_url"])
    return published["html_url"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "verify-wheels", "publish"))
    parser.add_argument("--directory", type=Path, default=Path("release-wheels"))
    args = parser.parse_args()
    root = Path.cwd()
    if args.command == "plan":
        print(json.dumps(plan_release(root), indent=2))
    elif args.command == "prepare":
        write_outputs(prepare(root))
    else:
        version = os.environ["RELEASE_VERSION"]
        files = verify_wheels(root, args.directory, version)
        print(f"Verified 12 native wheels and SHA256SUMS for {version}.")
        if args.command == "publish":
            tag = os.environ["RELEASE_TAG"]
            if tag != f"v{version}" or tag_commit(root, tag) != git(root, "rev-parse", "HEAD"):
                raise ValueError("Release tag must identify the exact tested source commit.")
            client = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"])
            publish_assets(client, tag, files)


if __name__ == "__main__":
    main()
