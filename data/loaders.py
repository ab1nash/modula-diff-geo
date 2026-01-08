"""
Dataset Loaders for Geometric Imputation Benchmarks

Converts raw data into manifold representations:
- PhysioNet EEG → SPD covariance matrices
- GHCN-Daily → Spherical coordinates + scalar fields
- CMU Mocap → Joint angle trajectories on SO(3)^k
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import numpy as np

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = np
    HAS_JAX = False


DATA_DIR = Path(__file__).parent


@dataclass
class SPDDataset:
    """Container for SPD manifold data (covariance matrices)."""
    matrices: np.ndarray      # (n_samples, dim, dim) SPD matrices
    labels: np.ndarray        # (n_samples,) class labels
    timestamps: np.ndarray    # (n_samples,) time indices
    subject_ids: List[str]    # Subject identifiers
    manifold_dim: int         # Matrix dimension
    
    @property
    def n_samples(self) -> int:
        return len(self.matrices)


@dataclass
class SphericalDataset:
    """Container for spherical S² data."""
    coordinates: np.ndarray   # (n_points, 2) lat/lon in radians
    values: np.ndarray        # (n_points, n_features) scalar fields
    timestamps: np.ndarray    # (n_times,) time indices
    station_ids: List[str]    # Station identifiers
    feature_names: List[str]  # Names of scalar fields
    
    @property
    def n_points(self) -> int:
        return len(self.coordinates)


@dataclass 
class MocapDataset:
    """Container for motion capture data."""
    joint_angles: np.ndarray   # (n_frames, n_joints, 3) Euler angles
    marker_positions: np.ndarray  # (n_frames, n_markers, 3) XYZ positions
    skeleton: Dict              # Skeleton hierarchy
    timestamps: np.ndarray      # (n_frames,) time indices
    subject_id: str
    motion_id: str
    
    @property
    def n_frames(self) -> int:
        return len(self.timestamps)


class PhysioNetEEGLoader:
    """
    Load PhysioNet EEG data and convert to SPD covariance matrices.
    
    The 64-channel EEG signals are converted to spatial covariance matrices,
    which lie on the SPD manifold P_64.
    
    Tasks:
    - T0: Rest
    - T1: Left fist movement/imagery
    - T2: Right fist movement/imagery
    - T3: Both fists movement/imagery
    - T4: Both feet movement/imagery
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (DATA_DIR / "physionet_eeg")
        self._check_pyedflib()
    
    def _check_pyedflib(self):
        """Check if pyedflib is available for reading EDF files."""
        try:
            import pyedflib
            self.has_pyedflib = True
        except ImportError:
            self.has_pyedflib = False
            print("Warning: pyedflib not installed. Install with: pip install pyedflib")
    
    def load_subject(self, subject_id: str, 
                     runs: Optional[List[int]] = None,
                     window_size: int = 160,  # 1 second at 160 Hz
                     overlap: float = 0.5) -> Optional[SPDDataset]:
        """
        Load EEG data for a single subject and compute covariance matrices.
        
        Args:
            subject_id: e.g., "S001"
            runs: List of run numbers (1-14), None for all
            window_size: Samples per window for covariance estimation
            overlap: Overlap fraction between windows
            
        Returns:
            SPDDataset or None if loading fails
        """
        if not self.has_pyedflib:
            return self._load_synthetic_fallback(subject_id)
        
        import pyedflib
        
        subject_dir = self.data_dir / subject_id
        if not subject_dir.exists():
            print(f"Subject {subject_id} not found")
            return None
        
        runs = runs or list(range(1, 15))
        
        all_matrices = []
        all_labels = []
        all_timestamps = []
        
        for run in runs:
            run_file = subject_dir / f"{subject_id}R{run:02d}.edf"
            if not run_file.exists():
                continue
            
            try:
                f = pyedflib.EdfReader(str(run_file))
                n_channels = f.signals_in_file
                n_samples = f.getNSamples()[0]
                
                # Read all channels
                signals = np.zeros((n_channels, n_samples))
                for i in range(n_channels):
                    signals[i] = f.readSignal(i)
                f.close()
                
                # Normalize signals (z-score per channel) to ensure scale-invariance
                # Raw EEG is typically in microvolts, which can cause huge covariance values
                for i in range(n_channels):
                    mean_val = np.mean(signals[i])
                    std_val = np.std(signals[i])
                    if std_val > 1e-10:  # Avoid division by zero
                        signals[i] = (signals[i] - mean_val) / std_val
                
                # Determine label from run number
                # Runs 3,7,11 = T1 (left fist), Runs 4,8,12 = T2 (right fist)
                # Runs 5,9,13 = T3 (both fists), Runs 6,10,14 = T4 (feet)
                # Runs 1,2 = baseline
                if run in [1, 2]:
                    label = 0  # Rest/baseline
                elif run in [3, 7, 11]:
                    label = 1  # Left fist
                elif run in [4, 8, 12]:
                    label = 2  # Right fist
                elif run in [5, 9, 13]:
                    label = 3  # Both fists
                else:
                    label = 4  # Both feet
                
                # Compute windowed covariance matrices
                step = int(window_size * (1 - overlap))
                for start in range(0, n_samples - window_size, step):
                    window = signals[:, start:start + window_size]
                    
                    # Compute covariance matrix (ensure SPD)
                    cov = np.cov(window)
                    # Regularize to ensure positive definiteness
                    cov = cov + 1e-6 * np.eye(n_channels)
                    
                    all_matrices.append(cov)
                    all_labels.append(label)
                    all_timestamps.append(start / 160.0)  # Convert to seconds
                    
            except Exception as e:
                print(f"Error loading {run_file}: {e}")
                continue
        
        if not all_matrices:
            return self._load_synthetic_fallback(subject_id)
        
        return SPDDataset(
            matrices=np.array(all_matrices),
            labels=np.array(all_labels),
            timestamps=np.array(all_timestamps),
            subject_ids=[subject_id] * len(all_matrices),
            manifold_dim=all_matrices[0].shape[0]
        )
    
    def _load_synthetic_fallback(self, subject_id: str) -> SPDDataset:
        """Generate synthetic SPD data as fallback when real data unavailable."""
        print(f"Using synthetic SPD data for {subject_id}")
        
        np.random.seed(hash(subject_id) % (2**32))
        
        n_samples = 100
        dim = 64  # Match EEG channels
        n_classes = 5
        
        matrices = []
        labels = []
        
        # Generate class-specific base covariances
        class_bases = []
        for c in range(n_classes):
            # Random SPD matrix via Wishart-like construction
            L = np.random.randn(dim, dim) * 0.1
            base = L @ L.T + (1 + c * 0.5) * np.eye(dim)
            class_bases.append(base)
        
        for i in range(n_samples):
            c = i % n_classes
            base = class_bases[c]
            
            # Add noise while preserving SPD
            noise = np.random.randn(dim, dim) * 0.05
            noise = noise @ noise.T
            
            mat = base + noise
            matrices.append(mat)
            labels.append(c)
        
        return SPDDataset(
            matrices=np.array(matrices),
            labels=np.array(labels),
            timestamps=np.arange(n_samples, dtype=float),
            subject_ids=[subject_id] * n_samples,
            manifold_dim=dim
        )
    
    def load_multiple_subjects(self, subject_ids: Optional[List[str]] = None,
                               **kwargs) -> SPDDataset:
        """Load and concatenate data from multiple subjects."""
        if subject_ids is None:
            # Find available subjects
            subject_ids = [d.name for d in self.data_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('S')]
            if not subject_ids:
                subject_ids = ["S001"]  # Fallback to synthetic
        
        all_matrices = []
        all_labels = []
        all_timestamps = []
        all_subject_ids = []
        
        for sid in subject_ids[:5]:  # Limit to 5 subjects
            dataset = self.load_subject(sid, **kwargs)
            if dataset is not None:
                all_matrices.append(dataset.matrices)
                all_labels.append(dataset.labels)
                all_timestamps.append(dataset.timestamps)
                all_subject_ids.extend(dataset.subject_ids)
        
        if not all_matrices:
            return self._load_synthetic_fallback("combined")
        
        return SPDDataset(
            matrices=np.concatenate(all_matrices),
            labels=np.concatenate(all_labels),
            timestamps=np.concatenate(all_timestamps),
            subject_ids=all_subject_ids,
            manifold_dim=all_matrices[0].shape[1]
        )


class GHCNDailyLoader:
    """
    Load GHCN-Daily climate data for spherical experiments.
    
    Converts weather station data to spherical coordinates (lat/lon)
    with associated temperature/precipitation scalar fields.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (DATA_DIR / "ghcn_daily")
    
    def load_stations(self, n_stations: int = 50,
                      year: int = 2020) -> Optional[SphericalDataset]:
        """
        Load station data for a specific year.
        
        Args:
            n_stations: Maximum number of stations to load
            year: Year to extract data from
            
        Returns:
            SphericalDataset or None
        """
        stations_file = self.data_dir / "ghcnd-stations.txt"
        data_dir = self.data_dir / "by_station"
        
        if not stations_file.exists():
            return self._load_synthetic_fallback(n_stations)
        
        # Parse station metadata
        stations = []
        with open(stations_file, 'r') as f:
            for line in f:
                if len(stations) >= n_stations * 2:  # Get extra in case some fail
                    break
                try:
                    station_id = line[0:11].strip()
                    lat = float(line[12:20])
                    lon = float(line[21:30])
                    stations.append((station_id, lat, lon))
                except:
                    continue
        
        # Load station data files
        coords = []
        values = []
        station_ids = []
        
        for station_id, lat, lon in stations:
            if len(station_ids) >= n_stations:
                break
                
            dly_file = data_dir / f"{station_id}.dly"
            if not dly_file.exists():
                continue
            
            try:
                # Parse .dly file for specified year
                tmax_values, tmin_values, prcp_values = self._parse_dly_file(
                    dly_file, year
                )
                
                if tmax_values is None:
                    continue
                
                # Convert lat/lon to radians
                lat_rad = np.radians(lat)
                lon_rad = np.radians(lon)
                
                coords.append([lat_rad, lon_rad])
                values.append([
                    np.nanmean(tmax_values) if len(tmax_values) > 0 else np.nan,
                    np.nanmean(tmin_values) if len(tmin_values) > 0 else np.nan,
                    np.nanmean(prcp_values) if len(prcp_values) > 0 else np.nan,
                ])
                station_ids.append(station_id)
                
            except Exception as e:
                continue
        
        if not coords:
            return self._load_synthetic_fallback(n_stations)
        
        return SphericalDataset(
            coordinates=np.array(coords),
            values=np.array(values),
            timestamps=np.array([year]),
            station_ids=station_ids,
            feature_names=['TMAX', 'TMIN', 'PRCP']
        )
    
    def _parse_dly_file(self, filepath: Path, year: int) -> Tuple[List, List, List]:
        """Parse GHCN .dly file format."""
        tmax = []
        tmin = []
        prcp = []
        
        with open(filepath, 'r') as f:
            for line in f:
                if len(line) < 21:
                    continue
                    
                try:
                    line_year = int(line[11:15])
                    if line_year != year:
                        continue
                        
                    element = line[17:21]
                    
                    # Parse daily values (31 days, each value is 8 chars)
                    for day in range(31):
                        start = 21 + day * 8
                        value_str = line[start:start + 5].strip()
                        flag = line[start + 6:start + 7] if len(line) > start + 6 else ''
                        
                        if value_str == '-9999' or flag in ['I', 'G', 'X']:
                            continue
                            
                        try:
                            value = float(value_str)
                            if element == 'TMAX':
                                tmax.append(value / 10.0)  # Convert to Celsius
                            elif element == 'TMIN':
                                tmin.append(value / 10.0)
                            elif element == 'PRCP':
                                prcp.append(value / 10.0)  # Convert to mm
                        except:
                            continue
                            
                except Exception:
                    continue
        
        return tmax, tmin, prcp
    
    def _load_synthetic_fallback(self, n_stations: int) -> SphericalDataset:
        """Generate synthetic spherical data."""
        print(f"Using synthetic spherical data for {n_stations} stations")
        
        np.random.seed(42)
        
        # Random points on sphere (using uniform sampling)
        u = np.random.uniform(0, 1, n_stations)
        v = np.random.uniform(0, 1, n_stations)
        
        lat = np.arcsin(2 * u - 1)  # Uniform on sphere
        lon = 2 * np.pi * v - np.pi
        
        coords = np.stack([lat, lon], axis=1)
        
        # Generate temperature field (latitude-dependent + noise)
        tmax = 30 - 40 * np.abs(lat) / (np.pi/2) + np.random.randn(n_stations) * 5
        tmin = tmax - 10 + np.random.randn(n_stations) * 2
        prcp = np.abs(np.random.randn(n_stations) * 50)
        
        values = np.stack([tmax, tmin, prcp], axis=1)
        
        return SphericalDataset(
            coordinates=coords,
            values=values,
            timestamps=np.array([2020]),
            station_ids=[f"SYNTH{i:04d}" for i in range(n_stations)],
            feature_names=['TMAX', 'TMIN', 'PRCP']
        )


class CMUMocapLoader:
    """
    Load CMU Motion Capture data for shape/rotation manifold experiments.
    
    Converts ASF/AMC files to joint angle trajectories, which lie on
    the product manifold SO(3)^k (k joints).
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (DATA_DIR / "cmu_mocap")
    
    def load_motion(self, subject_id: str, motion_id: str) -> Optional[MocapDataset]:
        """
        Load a single motion sequence.
        
        Args:
            subject_id: e.g., "01"
            motion_id: e.g., "01"
            
        Returns:
            MocapDataset or None
        """
        subject_dir = self.data_dir / f"subject_{subject_id}"
        asf_file = subject_dir / f"{subject_id}.asf"
        amc_file = subject_dir / f"{subject_id}_{motion_id}.amc"
        
        if not asf_file.exists() or not amc_file.exists():
            return self._load_synthetic_fallback(subject_id, motion_id)
        
        try:
            skeleton = self._parse_asf(asf_file)
            joint_angles = self._parse_amc(amc_file, skeleton)
            
            n_frames = len(joint_angles)
            
            return MocapDataset(
                joint_angles=joint_angles,
                marker_positions=np.zeros((n_frames, 0, 3)),  # Not available in ASF/AMC
                skeleton=skeleton,
                timestamps=np.arange(n_frames) / 120.0,  # 120 fps default
                subject_id=subject_id,
                motion_id=motion_id
            )
        except Exception as e:
            print(f"Error loading motion {subject_id}_{motion_id}: {e}")
            return self._load_synthetic_fallback(subject_id, motion_id)
    
    def _parse_asf(self, filepath: Path) -> Dict:
        """Parse ASF skeleton file."""
        skeleton = {'joints': {}, 'hierarchy': {}}
        
        current_section = None
        current_bone = None
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if line.startswith(':'):
                    current_section = line[1:].lower()
                    continue
                
                if current_section == 'bonedata':
                    if line == 'begin':
                        current_bone = {}
                    elif line == 'end':
                        if 'name' in current_bone:
                            skeleton['joints'][current_bone['name']] = current_bone
                        current_bone = None
                    elif current_bone is not None:
                        parts = line.split()
                        if parts[0] == 'name':
                            current_bone['name'] = parts[1]
                        elif parts[0] == 'dof':
                            current_bone['dof'] = parts[1:]
                        elif parts[0] == 'direction':
                            current_bone['direction'] = [float(x) for x in parts[1:4]]
                        elif parts[0] == 'length':
                            current_bone['length'] = float(parts[1])
        
        return skeleton
    
    def _parse_amc(self, filepath: Path, skeleton: Dict) -> np.ndarray:
        """Parse AMC motion file."""
        frames = []
        current_frame = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith(':'):
                    continue
                
                parts = line.split()
                
                # Check if it's a frame number
                try:
                    frame_num = int(parts[0])
                    if current_frame:
                        frames.append(current_frame)
                    current_frame = {}
                    continue
                except ValueError:
                    pass
                
                # It's joint data
                if len(parts) >= 2:
                    joint_name = parts[0]
                    values = [float(x) for x in parts[1:]]
                    current_frame[joint_name] = values
        
        if current_frame:
            frames.append(current_frame)
        
        # Convert to array (n_frames, n_joints, 3)
        joint_names = list(skeleton['joints'].keys())
        n_frames = len(frames)
        n_joints = len(joint_names)
        
        joint_angles = np.zeros((n_frames, n_joints, 3))
        
        for i, frame in enumerate(frames):
            for j, name in enumerate(joint_names):
                if name in frame:
                    angles = frame[name]
                    # Take first 3 values (rx, ry, rz) if available
                    for k in range(min(3, len(angles))):
                        joint_angles[i, j, k] = np.radians(angles[k])
        
        return joint_angles
    
    def _load_synthetic_fallback(self, subject_id: str, motion_id: str) -> MocapDataset:
        """Generate synthetic motion data."""
        print(f"Using synthetic mocap data for {subject_id}_{motion_id}")
        
        np.random.seed(hash(f"{subject_id}_{motion_id}") % (2**32))
        
        n_frames = 500
        n_joints = 20
        
        # Generate smooth joint angle trajectories
        t = np.linspace(0, 4 * np.pi, n_frames)
        
        joint_angles = np.zeros((n_frames, n_joints, 3))
        
        for j in range(n_joints):
            freq = 1 + j * 0.1
            phase = j * np.pi / n_joints
            
            # Periodic motion with noise
            joint_angles[:, j, 0] = 0.3 * np.sin(freq * t + phase)
            joint_angles[:, j, 1] = 0.2 * np.cos(freq * t + phase + np.pi/4)
            joint_angles[:, j, 2] = 0.1 * np.sin(2 * freq * t + phase)
            
            # Add some noise
            joint_angles[:, j] += np.random.randn(n_frames, 3) * 0.02
        
        skeleton = {
            'joints': {f'joint_{i}': {'dof': ['rx', 'ry', 'rz']} 
                      for i in range(n_joints)},
            'hierarchy': {}
        }
        
        return MocapDataset(
            joint_angles=joint_angles,
            marker_positions=np.zeros((n_frames, 0, 3)),
            skeleton=skeleton,
            timestamps=np.arange(n_frames) / 120.0,
            subject_id=subject_id,
            motion_id=motion_id
        )
    
    def load_multiple_motions(self, n_motions: int = 5) -> List[MocapDataset]:
        """Load multiple motion sequences."""
        motions = []
        
        # Try to find available motions
        if self.data_dir.exists():
            for subject_dir in sorted(self.data_dir.iterdir())[:n_motions]:
                if not subject_dir.is_dir():
                    continue
                    
                subject_id = subject_dir.name.replace("subject_", "")
                
                # Find motion files
                for amc_file in sorted(subject_dir.glob("*.amc"))[:2]:
                    motion_id = amc_file.stem.split('_')[-1]
                    dataset = self.load_motion(subject_id, motion_id)
                    if dataset is not None:
                        motions.append(dataset)
                        
                if len(motions) >= n_motions:
                    break
        
        # Fill with synthetic if needed
        while len(motions) < n_motions:
            i = len(motions)
            motions.append(self._load_synthetic_fallback(f"synth_{i:02d}", "01"))
        
        return motions


class DatasetRegistry:
    """Central registry for all dataset loaders."""
    
    @staticmethod
    def get_loader(dataset_name: str):
        """Get loader for a dataset by name."""
        loaders = {
            'physionet_eeg': PhysioNetEEGLoader,
            'ghcn_daily': GHCNDailyLoader,
            'cmu_mocap': CMUMocapLoader,
        }
        
        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(loaders.keys())}")
        
        return loaders[dataset_name]()
    
    @staticmethod
    def list_available() -> List[str]:
        """List available dataset loaders."""
        return ['physionet_eeg', 'ghcn_daily', 'cmu_mocap']
    
    @staticmethod
    def load_for_benchmark(dataset_name: str, **kwargs):
        """Load dataset in format ready for benchmarking."""
        loader = DatasetRegistry.get_loader(dataset_name)
        
        if dataset_name == 'physionet_eeg':
            return loader.load_multiple_subjects(**kwargs)
        elif dataset_name == 'ghcn_daily':
            return loader.load_stations(**kwargs)
        elif dataset_name == 'cmu_mocap':
            return loader.load_multiple_motions(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

