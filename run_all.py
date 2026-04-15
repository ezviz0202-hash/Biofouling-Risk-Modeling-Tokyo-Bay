import subprocess
import sys
import time
from pathlib import Path


MODULES = [
    ("Module 1 — DEB Model + Sensitivity Analysis",
     Path("biofouling_deb_sensitivity.py"),
     Path("results") / "deb"),

    ("Module 2 — Tokyo Bay Risk Prediction",
     Path("tokyo_bay_biofouling_risk.py"),
     Path("results") / "tokyobay"),
]

SEP = "=" * 65


def run_module(title, script_path, output_dir):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

    if not script_path.exists():
        print(f"  ERROR: {script_path} not found.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  Module failed (exit code {result.returncode})")
        return False

    outputs = sorted(output_dir.iterdir())
    print(f"\n  Completed in {elapsed:.1f}s  |  {len(outputs)} file(s) in {output_dir}/")
    for f in outputs:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<45} {size_kb:>7.1f} KB")
    return True


def main():
    print(f"\n{SEP}")
    print("  Biofouling Dynamics — Numerical Modeling Prototype")
    print(SEP)

    all_ok = True
    for title, script, out_dir in MODULES:
        ok = run_module(title, script, out_dir)
        all_ok = all_ok and ok

    print(f"\n{SEP}")
    if all_ok:
        print("  ✓  All modules completed successfully.")
        print("     Results saved to results/deb/ and results/tokyobay/")
    else:
        print("  ✗  One or more modules failed. Check output above.")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
