"""Command-line entry point: python scripts/run_pipeline.py --out outputs"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoore.pipeline import run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="GeoOre prospectivity + grade pipeline")
    p.add_argument("--out", default="outputs")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--size", type=int, default=200)
    p.add_argument("--faults", help="Optional fault lines file (GeoJSON/shapefile)")
    p.add_argument("--samples", help="Optional assay CSV with x,y,grade and element columns")
    a = p.parse_args()
    *_, summary = run(a.out, a.seed, a.size, a.faults, a.samples)
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2))
    print(f"\nOutputs written to {a.out}/")


if __name__ == "__main__":
    main()
