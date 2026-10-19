"""Build native extensions. Importing the source package never runs this file."""
from pathlib import Path
from hashlib import sha256
import tomllib

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_py import build_py


source_root = Path("src")
sources = sorted(source_root.rglob("*.py"))
compiled_names = {
    ".".join(path.relative_to(source_root).with_suffix("").parts)
    for path in sources if path.name != "__init__.py"
}
# A new version or module layout gets a fresh cache without deleting old builds.
# This prevents removed extensions from leaking into a later wheel.
package_version = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
layout_hash = sha256("|".join(sorted(compiled_names)).encode("utf-8")).hexdigest()[:12]
build_root = Path("build") / f"{package_version}-{layout_hash}"


class NativeBuildPy(build_py):
    """Keep package facades; do not ship duplicate Python implementations."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [item for item in modules if f"{item[0]}.{item[1]}" not in compiled_names]


extensions = [
    Extension(name, [str(source_root.joinpath(*name.split(".")).with_suffix(".py"))])
    for name in sorted(compiled_names)
]

setup(
    package_dir={"": "src"},
    ext_modules=cythonize(
        extensions,
        build_dir=str(build_root / "cython"),
        compiler_directives={
            "language_level": 3,
            "annotation_typing": False,
            "binding": True,
            "embedsignature": True,
        },
    ),
    cmdclass={"build_py": NativeBuildPy},
    options={"build": {"build_base": str(build_root)}},
    include_package_data=False,
)
