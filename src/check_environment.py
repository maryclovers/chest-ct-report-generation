# check_environment.py
# prints what is installed and whether a GPU is visible, then saves the same
# thing to outputs/environment_report.txt so I have a file to show, not just talk.
# runs fine on CPU, so I can run it before the GPU question is even settled.

import sys
import platform
from datetime import datetime

# config sets up the outputs folder for me
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config import OUTPUTS_DIR


def check_package(name, import_name=None):
    # try to import a package and grab its version, report cleanly if missing
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        return f"OK    {name:<16} {version}"
    except Exception as error:
        return f"MISSING {name:<14} ({error.__class__.__name__})"


def check_gpu():
    # report CUDA status and VRAM if torch can see a GPU
    lines = []
    try:
        import torch
        lines.append(f"torch version:    {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        lines.append(f"CUDA available:   {cuda_ok}")
        if cuda_ok:
            device_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            lines.append(f"GPU:              {device_name}")
            lines.append(f"VRAM total:       {total_vram:.1f} GB")
        else:
            lines.append("GPU:              none visible (running on CPU)")
    except Exception as error:
        lines.append(f"torch not usable: {error}")
    return lines


def main():
    report = []
    report.append("=" * 60)
    report.append("ENVIRONMENT REPORT")
    report.append("=" * 60)
    report.append(f"generated:        {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"python version:   {sys.version.split()[0]}")
    report.append(f"platform:         {platform.platform()}")
    report.append("")

    report.append("-" * 60)
    report.append("GPU")
    report.append("-" * 60)
    report.extend(check_gpu())
    report.append("")

    report.append("-" * 60)
    report.append("CORE PACKAGES")
    report.append("-" * 60)
    # the packages this project actually needs, checked one by one
    packages = [
        ("monai", "monai"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
        ("huggingface_hub", "huggingface_hub"),
        ("nibabel", "nibabel"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("scikit-learn", "sklearn"),
    ]
    for name, import_name in packages:
        report.append(check_package(name, import_name))

    # print to screen and save to file
    text = "\n".join(report)
    print(text)

    report_path = OUTPUTS_DIR / "environment_report.txt"
    with open(report_path, "w") as f:
        f.write(text + "\n")
    print("")
    print(f"saved report to: {report_path}")


if __name__ == "__main__":
    main()
