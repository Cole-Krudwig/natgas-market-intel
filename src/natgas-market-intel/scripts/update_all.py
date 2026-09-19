import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"


SCRIPTS = [
    "update_yfinance.py",
    "update_eia.py",
    "update_noaa.py",
]


def run_script(script_name):
    script_path = SCRIPTS_DIR / script_name

    print()
    print("=" * 70)
    print(f"Running {script_path}")
    print("=" * 70)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_DIR),
        check=True,
    )


def main():
    for script in SCRIPTS:
        run_script(script)

    print()
    print("All daily updates completed successfully.")


if __name__ == "__main__":
    main()