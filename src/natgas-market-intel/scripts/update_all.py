import subprocess
import sys


SCRIPTS = [
    "scripts/update_yfinance.py",
    "scripts/update_eia.py",
    "scripts/update_noaa.py",
    # "scripts/update_cftc.py",
]


def run_script(script):
    print(f"\n{'=' * 70}")
    print(f"Running {script}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, script],
        check=True,
    )

    return result.returncode


def main():
    for script in SCRIPTS:
        run_script(script)

    print("\nAll daily updates completed successfully.")


if __name__ == "__main__":
    main()