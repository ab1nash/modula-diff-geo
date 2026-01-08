"""
Real-World Dataset Loaders for Geometric Imputation Benchmarks

Datasets organized by manifold type:
- SPD Manifold: PhysioNet EEG (covariance matrices)
- Sphere S²: GHCN-Daily (global climate stations)
- Shape/SO(3)^k: CMU Mocap (motion capture)

Datasets requiring manual download:
- TUM VI Benchmark (large video/IMU data)
- UZH-FPV Drone Racing (large video/IMU data)
- Google Smartphone Decimeter (Kaggle API needed)
- Stanford HARDI (large MRI data)
- Redwood 3D Scans (large 3D mesh data)
"""
from .loaders import (
    PhysioNetEEGLoader,
    GHCNDailyLoader,
    CMUMocapLoader,
    DatasetRegistry,
)

from .download import (
    download_physionet_eeg,
    download_ghcn_sample,
    download_cmu_mocap_sample,
    download_all_available,
    check_dataset_status,
)

__all__ = [
    'PhysioNetEEGLoader',
    'GHCNDailyLoader', 
    'CMUMocapLoader',
    'DatasetRegistry',
    'download_physionet_eeg',
    'download_ghcn_sample',
    'download_cmu_mocap_sample',
    'download_all_available',
    'check_dataset_status',
]

