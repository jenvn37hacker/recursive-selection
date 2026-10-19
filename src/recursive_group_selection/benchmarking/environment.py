"""Best-effort, read-only machine metadata outside all measurement intervals."""
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
import subprocess


@dataclass(frozen=True)
class EnvironmentReport:
    information: dict

    @property
    def table(self):
        import pandas as pd
        return pd.DataFrame([{"Information": key, "Value": str(value)}
                             for key, value in self.information.items()])

    @property
    def caption(self):
        info = self.information
        return (f"CPU: {info['CPU']} | {info['Physical cores']} cores / "
                f"{info['Logical processors']} logical processors | RAM: {info['RAM (GB)']} GB\n"
                f"Installed GPU: {info['Installed GPU']} | Inference: {info['Inference device']} | "
                f"thread_count: {info['Inference thread_count']}\nOS: {info['Operating system']}\n"
                + " | ".join(f"{key}: {value}" for key, value in info.items()
                             if key == "Python" or key.startswith("Package: ")))


def collect_environment(*, packages=("numpy", "scipy", "recursive-group-selection"),
                        inference_device="Unspecified", inference_threads="Model default", extra=None):
    info = {"CPU": platform.processor() or "Unavailable", "Physical cores": "Unavailable",
            "Logical processors": os.cpu_count() or "Unavailable", "RAM (GB)": "Unavailable",
            "Installed GPU": "Unavailable", "Operating system": platform.platform(),
            "Inference device": inference_device, "Inference thread_count": inference_threads,
            "Python": platform.python_version()}
    # psutil is optional and is never installed or required by this function.
    try:
        import psutil
        info["Physical cores"] = psutil.cpu_count(logical=False) or "Unavailable"
        info["RAM (GB)"] = f"{psutil.virtual_memory().total / 1024**3:.2f}"
    except ImportError:
        pass
    if platform.system() == "Windows":
        command = r"""
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$reportCpus = @(Get-CimInstance Win32_Processor -ErrorAction Stop)
$reportComputer = Get-CimInstance Win32_ComputerSystem -ErrorAction Stop
$reportGpus = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue)
[PSCustomObject]@{
 cpu = ($reportCpus | ForEach-Object { $_.Name.Trim() }) -join ", "
 physical_cores = [int](($reportCpus | Measure-Object NumberOfCores -Sum).Sum)
 logical_processors = [int](($reportCpus | Measure-Object NumberOfLogicalProcessors -Sum).Sum)
 ram_gb = [double]($reportComputer.TotalPhysicalMemory / 1GB)
 gpu = ($reportGpus | ForEach-Object { $_.Name }) -join ", "
} | ConvertTo-Json -Compress
"""
        try:
            completed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
                timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
            data = json.loads(completed.stdout)
            info.update({"CPU": data["cpu"], "Physical cores": data["physical_cores"],
                "Logical processors": data["logical_processors"], "RAM (GB)": f"{data['ram_gb']:.2f}",
                "Installed GPU": data["gpu"] or "None detected"})
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
            pass
    for package in packages:
        try:
            info["Package: " + package] = version(package)
        except PackageNotFoundError:
            info["Package: " + package] = "Not installed"
    if set(extra or {}).intersection(info):
        raise ValueError("Extra environment metadata cannot override collected fields.")
    info.update(extra or {})
    return EnvironmentReport(info)
