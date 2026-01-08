"""
Dataset Download Utilities

Downloads publicly available datasets for geometric imputation benchmarks.
Some datasets require manual download due to size or authentication.

Features:
- Caching: Won't re-download if files exist
- Partial download recovery: Uses .partial files
- Progress reporting
- Graceful failure handling
"""
import os
import urllib.request
import gzip
import shutil
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Callable
from datetime import datetime


DATA_DIR = Path(__file__).parent
CACHE_FILE = DATA_DIR / ".download_cache.json"


def _ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _load_cache() -> Dict:
    """Load download cache."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def _save_cache(cache: Dict):
    """Save download cache."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def _download_with_progress(url: str, output_path: Path, desc: str = "") -> bool:
    """Download file with progress reporting and partial file recovery."""
    partial_path = output_path.with_suffix(output_path.suffix + '.partial')
    
    try:
        # Download to partial file first
        print(f"    Downloading: {desc or url.split('/')[-1]}", end='', flush=True)
        urllib.request.urlretrieve(url, partial_path)
        
        # Move to final location
        shutil.move(str(partial_path), str(output_path))
        print(" ✓")
        return True
        
    except Exception as e:
        print(f" ✗ ({e})")
        if partial_path.exists():
            partial_path.unlink()
        return False


def _is_cached(dataset: str, key: str, cache: Dict) -> bool:
    """Check if a file is already cached and valid."""
    if dataset not in cache:
        return False
    return key in cache.get(dataset, {})


def download_physionet_eeg(n_subjects: int = 5, force: bool = False) -> Path:
    """
    Download PhysioNet EEG Motor Movement/Imagery Dataset.
    
    This dataset contains 64-channel EEG from 109 subjects performing
    motor/imagery tasks. Useful for SPD manifold (covariance) experiments.
    
    Reference: https://physionet.org/content/eegmmidb/1.0.0/
    
    Args:
        n_subjects: Number of subjects to download (max 109)
        force: Re-download even if files exist
        
    Returns:
        Path to downloaded data directory
    """
    output_dir = DATA_DIR / "physionet_eeg"
    _ensure_dir(output_dir)
    
    cache = _load_cache()
    if 'physionet_eeg' not in cache:
        cache['physionet_eeg'] = {}
    
    base_url = "https://physionet.org/files/eegmmidb/1.0.0/"
    
    # Download subject files (S001 through S{n_subjects})
    downloaded = []
    failed = []
    
    for i in range(1, min(n_subjects + 1, 110)):
        subject_id = f"S{i:03d}"
        subject_dir = output_dir / subject_id
        
        # Check cache
        if _is_cached('physionet_eeg', subject_id, cache) and not force:
            if subject_dir.exists():
                print(f"  {subject_id}: cached ✓")
                downloaded.append(subject_dir)
                continue
            
        _ensure_dir(subject_dir)
        
        # Download runs (R01-R14)
        # Start with just R01, R02, R03, R04 for faster testing
        runs_to_download = [1, 2, 3, 4] if n_subjects > 3 else range(1, 15)
        subject_success = True
        
        for run in runs_to_download:
            run_id = f"R{run:02d}"
            filename = f"{subject_id}{run_id}.edf"
            url = f"{base_url}{subject_id}/{filename}"
            output_path = subject_dir / filename
            
            if output_path.exists() and not force:
                continue
            
            if not _download_with_progress(url, output_path, f"{subject_id}/{filename}"):
                subject_success = False
                break
        
        if subject_success:
            downloaded.append(subject_dir)
            cache['physionet_eeg'][subject_id] = {
                'timestamp': datetime.now().isoformat(),
                'runs': len(runs_to_download)
            }
        else:
            failed.append(subject_id)
    
    # Save cache
    _save_cache(cache)
    
    print(f"\nPhysioNet EEG: {len(downloaded)} subjects downloaded, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed}")
    
    return output_dir


def download_ghcn_sample(n_stations: int = 100, force: bool = False) -> Path:
    """
    Download GHCN-Daily sample data.
    
    Global Historical Climatology Network daily data from weather stations.
    Useful for spherical (S²) manifold experiments.
    
    Reference: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
    
    Args:
        n_stations: Number of stations to include in sample
        force: Re-download even if files exist
        
    Returns:
        Path to downloaded data directory
    """
    output_dir = DATA_DIR / "ghcn_daily"
    _ensure_dir(output_dir)
    
    cache = _load_cache()
    if 'ghcn_daily' not in cache:
        cache['ghcn_daily'] = {}
    
    base_url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
    
    # Download station metadata first
    stations_path = output_dir / "ghcnd-stations.txt"
    if not stations_path.exists() or force:
        _download_with_progress(
            f"{base_url}ghcnd-stations.txt",
            stations_path,
            "station metadata"
        )
    
    # Parse stations file to get station IDs with good US coverage
    station_ids = []
    if stations_path.exists():
        with open(stations_path, 'r') as f:
            for line in f:
                if len(station_ids) >= n_stations * 2:  # Get extra for filtering
                    break
                station_id = line[:11].strip()
                # Prefer US stations (start with US) for reliability
                if station_id and station_id.startswith('US'):
                    station_ids.append(station_id)
        
        # If not enough US stations, add others
        if len(station_ids) < n_stations:
            with open(stations_path, 'r') as f:
                for line in f:
                    station_id = line[:11].strip()
                    if station_id and station_id not in station_ids:
                        station_ids.append(station_id)
                    if len(station_ids) >= n_stations:
                        break
    
    # Download station data files
    data_subdir = output_dir / "by_station"
    _ensure_dir(data_subdir)
    
    downloaded = 0
    failed = 0
    
    for station_id in station_ids[:n_stations]:
        dly_path = data_subdir / f"{station_id}.dly"
        
        if dly_path.exists() and _is_cached('ghcn_daily', station_id, cache) and not force:
            downloaded += 1
            continue
        
        gz_url = f"{base_url}all/{station_id}.dly.gz"
        gz_path = data_subdir / f"{station_id}.dly.gz"
        
        try:
            if _download_with_progress(gz_url, gz_path, station_id):
                # Decompress
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(dly_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                gz_path.unlink()
                
                cache['ghcn_daily'][station_id] = datetime.now().isoformat()
                downloaded += 1
            else:
                failed += 1
        except Exception as e:
            print(f"    Failed {station_id}: {e}")
            failed += 1
            if gz_path.exists():
                gz_path.unlink()
    
    _save_cache(cache)
    print(f"\nGHCN-Daily: {downloaded} stations downloaded, {failed} failed")
    return output_dir


def download_cmu_mocap_sample(n_subjects: int = 5, force: bool = False) -> Path:
    """
    Download CMU Motion Capture sample data.
    
    Motion capture data with marker positions. Useful for shape manifold
    and SO(3)^k (articulated body) experiments.
    
    Reference: http://mocap.cs.cmu.edu/
    
    Args:
        n_subjects: Number of subjects to download
        force: Re-download even if files exist
        
    Returns:
        Path to downloaded data directory
    """
    output_dir = DATA_DIR / "cmu_mocap"
    _ensure_dir(output_dir)
    
    cache = _load_cache()
    if 'cmu_mocap' not in cache:
        cache['cmu_mocap'] = {}
    
    # CMU Mocap base URL (ASF/AMC format - simpler than C3D)
    base_url = "http://mocap.cs.cmu.edu/subjects/"
    
    # Subject numbers that have good walking/running data
    sample_subjects = [
        ("01", ["01", "02"]),   # Walking
        ("02", ["01", "02"]),   # Walking variations
        ("07", ["01", "02"]),   # Walking
        ("08", ["01", "02"]),   # Running
        ("09", ["01", "02"]),   # Walking/running
    ]
    
    downloaded = 0
    failed = 0
    
    for subj_id, motion_ids in sample_subjects[:n_subjects]:
        subj_dir = output_dir / f"subject_{subj_id}"
        
        # Check cache
        if _is_cached('cmu_mocap', subj_id, cache) and subj_dir.exists() and not force:
            print(f"  Subject {subj_id}: cached ✓")
            downloaded += 1
            continue
        
        _ensure_dir(subj_dir)
        
        # Download ASF skeleton file
        asf_path = subj_dir / f"{subj_id}.asf"
        if not asf_path.exists() or force:
            if not _download_with_progress(
                f"{base_url}{subj_id}/{subj_id}.asf",
                asf_path,
                f"subject {subj_id} skeleton"
            ):
                failed += 1
                continue
        
        # Download AMC motion files
        subject_success = True
        for motion_id in motion_ids:
            amc_path = subj_dir / f"{subj_id}_{motion_id}.amc"
            
            if amc_path.exists() and not force:
                continue
            
            if not _download_with_progress(
                f"{base_url}{subj_id}/{subj_id}_{motion_id}.amc",
                amc_path,
                f"motion {subj_id}_{motion_id}"
            ):
                subject_success = False
        
        if subject_success:
            cache['cmu_mocap'][subj_id] = {
                'timestamp': datetime.now().isoformat(),
                'motions': motion_ids
            }
            downloaded += 1
        else:
            failed += 1
    
    _save_cache(cache)
    print(f"\nCMU Mocap: {downloaded} subjects downloaded, {failed} failed")
    return output_dir


def download_all_available(force: bool = False) -> Dict[str, Path]:
    """
    Download all available datasets.
    
    Returns:
        Dict mapping dataset name to download path
    """
    print("=" * 60)
    print("Downloading Real-World Geometric Datasets")
    print("=" * 60)
    
    results = {}
    
    print("\n[1/3] PhysioNet EEG (SPD Manifold - Covariance)")
    print("-" * 40)
    try:
        results['physionet_eeg'] = download_physionet_eeg(n_subjects=5, force=force)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['physionet_eeg'] = None
    
    print("\n[2/3] GHCN-Daily (Spherical S² - Climate Stations)")
    print("-" * 40)
    try:
        results['ghcn_daily'] = download_ghcn_sample(n_stations=50, force=force)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['ghcn_daily'] = None
    
    print("\n[3/3] CMU Mocap (Shape Manifold - Motion Capture)")
    print("-" * 40)
    try:
        results['cmu_mocap'] = download_cmu_mocap_sample(n_subjects=5, force=force)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['cmu_mocap'] = None
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for name, path in results.items():
        status = "✓" if path else "✗"
        print(f"  {status} {name}: {path or 'FAILED'}")
    
    return results


def check_dataset_status() -> Dict[str, dict]:
    """
    Check status of all datasets.
    
    Returns:
        Dict with dataset info including existence and file counts
    """
    datasets = {
        'physionet_eeg': {
            'path': DATA_DIR / "physionet_eeg",
            'manifold': 'SPD (covariance)',
            'downloadable': True,
            'manual_url': None,
        },
        'ghcn_daily': {
            'path': DATA_DIR / "ghcn_daily",
            'manifold': 'Sphere S²',
            'downloadable': True,
            'manual_url': None,
        },
        'cmu_mocap': {
            'path': DATA_DIR / "cmu_mocap",
            'manifold': 'Shape / SO(3)^k',
            'downloadable': True,
            'manual_url': None,
        },
        'tum_vi': {
            'path': DATA_DIR / "tum_vi",
            'manifold': 'Lie Group SE(3)',
            'downloadable': False,
            'manual_url': 'https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset',
        },
        'uzh_fpv': {
            'path': DATA_DIR / "uzh_fpv",
            'manifold': 'Lie Group SE(3)',
            'downloadable': False,
            'manual_url': 'https://fpv.ifi.uzh.ch/',
        },
        'google_smartphone': {
            'path': DATA_DIR / "google_smartphone",
            'manifold': 'SE(3) / Signal Space',
            'downloadable': False,
            'manual_url': 'https://www.kaggle.com/competitions/google-smartphone-decimeter-challenge/data',
        },
        'stanford_hardi': {
            'path': DATA_DIR / "stanford_hardi",
            'manifold': 'SPD (diffusion tensor)',
            'downloadable': False,
            'manual_url': 'https://purl.stanford.edu/ng782rw8378',
        },
        'redwood_3d': {
            'path': DATA_DIR / "redwood_3d",
            'manifold': '2-Manifold (Mesh)',
            'downloadable': False,
            'manual_url': 'http://redwood-data.org/3dscan/',
        },
    }
    
    for name, info in datasets.items():
        path = info['path']
        if path.exists():
            # Count files
            files = list(path.rglob('*'))
            info['exists'] = True
            info['n_files'] = len([f for f in files if f.is_file()])
            info['size_mb'] = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
        else:
            info['exists'] = False
            info['n_files'] = 0
            info['size_mb'] = 0
    
    return datasets


if __name__ == "__main__":
    print("Checking dataset status...\n")
    status = check_dataset_status()
    
    print("=" * 80)
    print("Dataset Status Report")
    print("=" * 80)
    print(f"{'Dataset':<20} {'Manifold':<20} {'Status':<12} {'Files':<8} {'Size':<10}")
    print("-" * 80)
    
    for name, info in status.items():
        status_str = "✓ Ready" if info['exists'] else ("⬇ Auto" if info['downloadable'] else "✗ Manual")
        size_str = f"{info['size_mb']:.1f} MB" if info['exists'] else "-"
        print(f"{name:<20} {info['manifold']:<20} {status_str:<12} {info['n_files']:<8} {size_str:<10}")
    
    print("\nDatasets requiring manual download:")
    for name, info in status.items():
        if not info['downloadable'] and not info['exists']:
            print(f"  - {name}: {info['manual_url']}")

