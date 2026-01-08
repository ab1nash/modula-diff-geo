#!/usr/bin/env python3
"""
Download geometric datasets for benchmarking.

Run: python download_datasets.py

This will download:
1. PhysioNet EEG (3 subjects, ~50MB) - SPD manifold
2. GHCN-Daily (30 stations, ~30MB) - Spherical manifold
3. CMU Mocap (3 subjects, ~5MB) - Shape/SO(3) manifold

Progress is cached - safe to restart if interrupted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.download import (
    download_physionet_eeg,
    download_ghcn_sample,
    download_cmu_mocap_sample,
    check_dataset_status,
)


def main():
    print("=" * 60)
    print("Downloading Geometric Datasets")
    print("=" * 60)
    
    # Check current status
    print("\nCurrent status:")
    status = check_dataset_status()
    for name in ['physionet_eeg', 'ghcn_daily', 'cmu_mocap']:
        info = status[name]
        marker = "✓" if info['exists'] and info['n_files'] > 0 else "○"
        print(f"  {marker} {name}: {info['n_files']} files, {info['size_mb']:.1f} MB")
    
    # Download PhysioNet EEG
    print("\n" + "-" * 60)
    print("[1/3] PhysioNet EEG (SPD Manifold)")
    print("-" * 60)
    try:
        download_physionet_eeg(n_subjects=3, force=False)
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Download GHCN-Daily
    print("\n" + "-" * 60)
    print("[2/3] GHCN-Daily (Spherical Manifold)")
    print("-" * 60)
    try:
        download_ghcn_sample(n_stations=30, force=False)
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Download CMU Mocap
    print("\n" + "-" * 60)
    print("[3/3] CMU Mocap (Shape/SO(3) Manifold)")
    print("-" * 60)
    try:
        download_cmu_mocap_sample(n_subjects=3, force=False)
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Final status
    print("\n" + "=" * 60)
    print("Final Status")
    print("=" * 60)
    status = check_dataset_status()
    for name in ['physionet_eeg', 'ghcn_daily', 'cmu_mocap']:
        info = status[name]
        marker = "✓" if info['exists'] and info['n_files'] > 0 else "✗"
        print(f"  {marker} {name}: {info['n_files']} files, {info['size_mb']:.1f} MB")
    
    print("\nManual download required for:")
    print("  - stanford_hardi: https://purl.stanford.edu/ng782rw8378")
    print("  - redwood_3d: http://redwood-data.org/3dscan/")
    print("\nPlace in: modula-diff-geo/data/<dataset_name>/")


if __name__ == '__main__':
    main()

