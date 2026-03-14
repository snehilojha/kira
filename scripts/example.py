"""Example script for testing telegram-runner execution."""

import time
import sys


def main():
    """Print numbered lines with a small delay to test streaming output."""
    for i in range(1, 11):
        print(f"Step {i}/10 — processing...")
        time.sleep(1)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
