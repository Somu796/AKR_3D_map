"""
CSV to TFCat Converter Module

Complete OOP implementation with type checking for converting
Wind AKR burst CSV data to TFCat JSON format.

Classes:
    - Config: Configuration management
    - BurstTimeSeries: Time series data model
    - TimeSeriesParser: Parse comma-separated arrays
    - GeometryBuilder: Create TFCat polygon geometries
    - TFCatMetadata: Build TFCat metadata structures
    - BurstConverter: Convert single burst to Feature
    - CSVToTFCatConverter: Main conversion orchestrator
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
import json
import logging
from tfcat import FeatureCollection, Feature, Polygon

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Configuration loaded from YAML file."""
    
    catalog: Dict[str, str]
    conversion: Dict[str, Any]
    csv: Dict[str, Any]
    tfcat: Dict[str, str]
    paths: Dict[str, str]
    logging: Dict[str, Any]
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'Config':
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        config_path : Path
            Path to config.yaml file
            
        Returns
        -------
        Config
            Configuration object
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def get_required_columns(self) -> List[str]:
        """Get list of required CSV columns."""
        return self.csv['required_columns']
    
    def get_datetime_format(self) -> str:
        """Get datetime parsing format."""
        return self.csv['datetime_format']


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class BurstTimeSeries:
    """
    Time-series data for a single AKR burst.
    
    All arrays must have the same length and are synchronized by timestamp.
    """
    
    timestamps: np.ndarray  # Unix timestamps
    x_gse: np.ndarray
    y_gse: np.ndarray
    z_gse: np.ndarray
    freq_min: np.ndarray
    freq_max: np.ndarray
    radius: np.ndarray
    lat_gse: np.ndarray
    lon_gse: np.ndarray
    lt_gse: np.ndarray
    
    def __post_init__(self) -> None:
        """Validate that all arrays have the same length."""
        lengths = {
            'timestamps': len(self.timestamps),
            'x_gse': len(self.x_gse),
            'y_gse': len(self.y_gse),
            'z_gse': len(self.z_gse),
            'freq_min': len(self.freq_min),
            'freq_max': len(self.freq_max),
            'radius': len(self.radius),
            'lat_gse': len(self.lat_gse),
            'lon_gse': len(self.lon_gse),
            'lt_gse': len(self.lt_gse)
        }
        
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"All time-series arrays must have same length. Got: {lengths}"
            )
    
    @property
    def n_points(self) -> int:
        """Number of data points in time series."""
        return len(self.timestamps)
    
    def get_valid_mask(self) -> np.ndarray:
        """
        Get boolean mask for valid (non-NaN) data points.
        
        Returns
        -------
        np.ndarray
            Boolean array, True where all critical fields are valid
        """
        return ~(
            np.isnan(self.timestamps) |
            np.isnan(self.x_gse) |
            np.isnan(self.y_gse) |
            np.isnan(self.z_gse) |
            np.isnan(self.freq_min) |
            np.isnan(self.freq_max)
        )
    
    def filter_valid(self) -> 'BurstTimeSeries':
        """
        Return new BurstTimeSeries with only valid data points.
        
        Returns
        -------
        BurstTimeSeries
            Filtered time series
        """
        mask = self.get_valid_mask()
        
        return BurstTimeSeries(
            timestamps=self.timestamps[mask],
            x_gse=self.x_gse[mask],
            y_gse=self.y_gse[mask],
            z_gse=self.z_gse[mask],
            freq_min=self.freq_min[mask],
            freq_max=self.freq_max[mask],
            radius=self.radius[mask],
            lat_gse=self.lat_gse[mask],
            lon_gse=self.lon_gse[mask],
            lt_gse=self.lt_gse[mask]
        )


@dataclass
class BurstMetadata:
    """Metadata for a single burst."""
    
    stime: datetime
    etime: datetime
    
    @property
    def duration_seconds(self) -> float:
        """Calculate burst duration in seconds."""
        return (self.etime - self.stime).total_seconds()
    
    @property
    def stime_iso(self) -> str:
        """Start time in ISO 8601 format."""
        return self.stime.isoformat() + 'Z'
    
    @property
    def etime_iso(self) -> str:
        """End time in ISO 8601 format."""
        return self.etime.isoformat() + 'Z'
    
    def validate(self) -> None:
        """
        Validate metadata.
        
        Raises
        ------
        ValueError
            If validation fails
        """
        if self.etime <= self.stime:
            raise ValueError(
                f"End time ({self.etime}) must be after start time ({self.stime})"
            )
        
        if self.duration_seconds <= 0:
            raise ValueError(
                f"Duration must be positive, got {self.duration_seconds}"
            )


# ============================================================================
# Parsers
# ============================================================================

class TimeSeriesParser:
    """Parse comma-separated string arrays into numpy arrays."""
    
    def __init__(self, datetime_format: str):
        """
        Initialize parser.
        
        Parameters
        ----------
        datetime_format : str
            strptime format string for datetime parsing
        """
        self.datetime_format = datetime_format
        self.logger = logging.getLogger(__name__)
    
    def parse_timestamp_array(self, timestamp_str: str) -> np.ndarray:
        """
        Parse comma-separated timestamp strings to Unix timestamps.
        
        Parameters
        ----------
        timestamp_str : str
            Comma-separated timestamp strings
            
        Returns
        -------
        np.ndarray
            Array of Unix timestamps (float64)
        """
        if pd.isna(timestamp_str) or not timestamp_str:
            return np.array([], dtype=np.float64)
        
        timestamps: List[float] = []
        
        for ts_str in str(timestamp_str).split(','):
            ts_str = ts_str.strip()
            
            # Handle NaT (Not a Time)
            if ts_str in ('NaT', 'nan', ''):
                timestamps.append(np.nan)
                continue
            
            try:
                # Parse datetime and convert to Unix timestamp
                dt = datetime.strptime(ts_str, self.datetime_format)
                unix_time = dt.timestamp()
                timestamps.append(unix_time)
            except ValueError as e:
                self.logger.debug(f"Failed to parse timestamp '{ts_str}': {e}")
                timestamps.append(np.nan)
        
        return np.array(timestamps, dtype=np.float64)
    
    def parse_float_array(self, value_str: str) -> np.ndarray:
        """
        Parse comma-separated float strings to numpy array.
        
        Parameters
        ----------
        value_str : str
            Comma-separated float strings
            
        Returns
        -------
        np.ndarray
            Array of floats (float64)
        """
        if pd.isna(value_str) or not value_str:
            return np.array([], dtype=np.float64)
        
        values: List[float] = []
        
        for val_str in str(value_str).split(','):
            val_str = val_str.strip()
            
            # Handle nan
            if val_str.lower() in ('nan', ''):
                values.append(np.nan)
            else:
                try:
                    values.append(float(val_str))
                except ValueError:
                    values.append(np.nan)
        
        return np.array(values, dtype=np.float64)


# ============================================================================
# Geometry Builder
# ============================================================================

class GeometryBuilder:
    """Build TFCat polygon geometries from time-series data."""
    
    def __init__(self, validate: bool = True):
        """
        Initialize geometry builder.
        
        Parameters
        ----------
        validate : bool
            Whether to validate polygon geometry
        """
        self.validate = validate
        self.logger = logging.getLogger(__name__)
    
    def create_polygon_coordinates(
        self,
        timeseries: BurstTimeSeries
    ) -> List[List[float]]:
        """
        Create polygon coordinates from time-series data.
        
        The polygon traces the burst boundary in time-frequency space:
        1. Top edge: forward in time at max frequency
        2. Right edge: down to min frequency at last time
        3. Bottom edge: backward in time at min frequency
        4. Left edge: up to max frequency at first time
        5. Close: return to starting point
        
        Parameters
        ----------
        timeseries : BurstTimeSeries
            Time-series data
            
        Returns
        -------
        List[List[float]]
            List of [time, frequency] coordinate pairs
            
        Raises
        ------
        ValueError
            If insufficient points for polygon
        """
        if timeseries.n_points < 2:
            raise ValueError(
                f"Insufficient points for polygon: {timeseries.n_points}"
            )
        
        times = timeseries.timestamps
        freq_min = timeseries.freq_min
        freq_max = timeseries.freq_max
        
        coords: List[List[float]] = []
        
        # Top edge: forward in time at max frequency
        for t, f in zip(times, freq_max):
            coords.append([float(t), float(f)])
        
        # Right edge: down to min frequency at last time
        if freq_min[-1] != freq_max[-1]:
            coords.append([float(times[-1]), float(freq_min[-1])])
        
        # Bottom edge: backward in time at min frequency
        for t, f in zip(reversed(times[:-1]), reversed(freq_min[:-1])):
            coords.append([float(t), float(f)])
        
        # Left edge: up to max frequency (close polygon)
        if freq_min[0] != freq_max[0]:
            coords.append([float(times[0]), float(freq_min[0])])
        
        # Close polygon (repeat first coordinate)
        coords.append(coords[0])
        
        # Validate if requested
        if self.validate:
            self._validate_polygon(coords)
        
        return coords
    
    def _validate_polygon(self, coords: List[List[float]]) -> None:
        """
        Validate polygon geometry.
        
        Parameters
        ----------
        coords : List[List[float]]
            Polygon coordinates
            
        Raises
        ------
        ValueError
            If validation fails
        """
        if len(coords) < 4:
            raise ValueError(
                f"Polygon must have at least 4 points, got {len(coords)}"
            )
        
        # Check if closed
        if coords[0] != coords[-1]:
            raise ValueError("Polygon is not closed")
        
        self.logger.debug(f"Polygon validated: {len(coords)} points")


# ============================================================================
# TFCat Metadata Builder
# ============================================================================

class TFCatMetadata:
    """Build TFCat metadata structures (CRS, fields, properties)."""
    
    def __init__(self, config: Config):
        """
        Initialize metadata builder.
        
        Parameters
        ----------
        config : Config
            Configuration object
        """
        self.config = config
    
    def build_crs(self) -> Dict[str, Any]:
        """
        Build Coordinate Reference System metadata.
        
        Returns
        -------
        dict
            CRS dictionary
        """
        return {
            "type": "local",
            "properties": {
                "name": "Time-Frequency",
                "time_coords_id": self.config.tfcat['time_coords'],
                "spectral_coords": {
                    "type": self.config.tfcat['spectral_type'],
                    "unit": self.config.tfcat['spectral_unit']
                },
                "ref_position_id": self.config.tfcat['ref_position']
            }
        }
    
    def build_fields(self) -> Dict[str, Dict[str, Any]]:
        """
        Build field definitions.
        
        Returns
        -------
        dict
            Field definitions dictionary
        """
        return {
            "stime": {
                "info": "Burst start time (ISO 8601 format)",
                "datatype": "str",
                "ucd": "time.start"
            },
            "etime": {
                "info": "Burst end time (ISO 8601 format)",
                "datatype": "str",
                "ucd": "time.end"
            },
            "duration_seconds": {
                "info": "Burst duration in seconds",
                "datatype": "float",
                "ucd": "time.duration",
                "unit": "s"
            },
            "n_points": {
                "info": "Number of time-frequency measurements",
                "datatype": "int",
                "ucd": "meta.number"
            },
            "x_gse_mean": {
                "info": "Mean spacecraft X position during burst (GSE coordinates)",
                "datatype": "float",
                "ucd": "pos.cartesian.x",
                "unit": "R_E"
            },
            "y_gse_mean": {
                "info": "Mean spacecraft Y position during burst (GSE coordinates)",
                "datatype": "float",
                "ucd": "pos.cartesian.y",
                "unit": "R_E"
            },
            "z_gse_mean": {
                "info": "Mean spacecraft Z position during burst (GSE coordinates)",
                "datatype": "float",
                "ucd": "pos.cartesian.z",
                "unit": "R_E"
            },
            "radius_mean": {
                "info": "Mean radial distance from Earth center",
                "datatype": "float",
                "ucd": "pos.distance",
                "unit": "R_E"
            },
            "lat_gse_mean": {
                "info": "Mean latitude in GSE coordinates",
                "datatype": "float",
                "ucd": "pos.galactic.lat",
                "unit": "deg"
            },
            "lon_gse_mean": {
                "info": "Mean longitude in GSE coordinates",
                "datatype": "float",
                "ucd": "pos.galactic.lon",
                "unit": "deg"
            },
            "lt_gse_mean": {
                "info": "Mean local time in GSE frame",
                "datatype": "float",
                "ucd": "time.phase",
                "unit": "h"
            },
            "timestamps": {
                "info": "Unix timestamps during burst window",
                "datatype": "float",
                "ucd": "time.epoch",
                "unit": "s"
            },
            "x_gse_timeseries": {
                "info": "Spacecraft X position time series (GSE)",
                "datatype": "float",
                "ucd": "pos.cartesian.x",
                "unit": "R_E"
            },
            "y_gse_timeseries": {
                "info": "Spacecraft Y position time series (GSE)",
                "datatype": "float",
                "ucd": "pos.cartesian.y",
                "unit": "R_E"
            },
            "z_gse_timeseries": {
                "info": "Spacecraft Z position time series (GSE)",
                "datatype": "float",
                "ucd": "pos.cartesian.z",
                "unit": "R_E"
            },
            "radius_timeseries": {
                "info": "Radial distance time series",
                "datatype": "float",
                "ucd": "pos.distance",
                "unit": "R_E"
            },
            "lat_gse_timeseries": {
                "info": "Latitude time series (GSE)",
                "datatype": "float",
                "ucd": "pos.galactic.lat",
                "unit": "deg"
            },
            "lon_gse_timeseries": {
                "info": "Longitude time series (GSE)",
                "datatype": "float",
                "ucd": "pos.galactic.lon",
                "unit": "deg"
            },
            "lt_gse_timeseries": {
                "info": "Local time time series (GSE)",
                "datatype": "float",
                "ucd": "time.phase",
                "unit": "h"
            },
            "freq_min_timeseries": {
                "info": "Lower frequency bound time series",
                "datatype": "float",
                "ucd": "em.freq;stat.min",
                "unit": "kHz"
            },
            "freq_max_timeseries": {
                "info": "Upper frequency bound time series",
                "datatype": "float",
                "ucd": "em.freq;stat.max",
                "unit": "kHz"
            }
        }
    
    def build_properties(
        self,
        total_bursts: int,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Build collection-level properties.
        
        Parameters
        ----------
        total_bursts : int
            Total number of bursts in catalog
        time_min : Optional[datetime]
            Earliest burst time
        time_max : Optional[datetime]
            Latest burst time
            
        Returns
        -------
        dict
            Collection properties
        """
        properties = {
            "title": self.config.catalog['title'],
            "instrument_host_name": self.config.catalog['instrument_host'],
            "instrument_name": self.config.catalog['instrument_name'],
            "receiver_name": self.config.catalog['receiver'],
            "target_name": self.config.catalog['target'],
            "target_region": self.config.catalog['target_region'],
            "feature_name": self.config.catalog['feature'],
            "coordinate_system": self.config.catalog['coordinate_system'],
            "total_bursts": total_bursts,
            "creation_date": datetime.utcnow().isoformat() + 'Z',
            "version": self.config.catalog['version']
        }
        
        if time_min is not None:
            properties["time_min"] = time_min.isoformat() + 'Z'
        
        if time_max is not None:
            properties["time_max"] = time_max.isoformat() + 'Z'
        
        return properties


# ============================================================================
# Burst Converter
# ============================================================================

class BurstConverter:
    """Convert a single CSV row (burst) to a TFCat Feature."""
    
    def __init__(self, config: Config):
        """
        Initialize burst converter.
        
        Parameters
        ----------
        config : Config
            Configuration object
        """
        self.config = config
        self.parser = TimeSeriesParser(config.get_datetime_format())
        self.geometry_builder = GeometryBuilder(
            validate=config.conversion['validate_geometry']
        )
        self.logger = logging.getLogger(__name__)
    
    def convert(self, row: pd.Series, feature_id: int) -> Feature:
        """
        Convert one CSV row to TFCat Feature.
        
        Parameters
        ----------
        row : pd.Series
            One row from CSV DataFrame
        feature_id : int
            Unique feature ID
            
        Returns
        -------
        Feature
            TFCat Feature object
            
        Raises
        ------
        ValueError
            If conversion fails
        """
        # Parse time series
        timeseries = self._parse_timeseries(row)
        
        # Create metadata
        metadata = self._create_metadata(row)
        metadata.validate()
        
        # Filter valid points if configured
        if self.config.conversion['filter_invalid']:
            timeseries = timeseries.filter_valid()
        
        # Check minimum points
        min_points = self.config.conversion['min_points']
        if timeseries.n_points < min_points:
            raise ValueError(
                f"Insufficient valid points: {timeseries.n_points} < {min_points}"
            )
        
        # Create geometry
        coords = self.geometry_builder.create_polygon_coordinates(timeseries)
        geometry = Polygon([coords])
        
        # Create properties
        properties = self._create_properties(metadata, timeseries)
        
        # Create Feature
        return Feature(
            id=feature_id,
            geometry=geometry,
            properties=properties
        )
    
    def _parse_timeseries(self, row: pd.Series) -> BurstTimeSeries:
        """
        Parse time-series arrays from CSV row.
        
        Parameters
        ----------
        row : pd.Series
            CSV row
            
        Returns
        -------
        BurstTimeSeries
            Parsed time series
        """
        return BurstTimeSeries(
            timestamps=self.parser.parse_timestamp_array(row['burst_timestamp']),
            x_gse=self.parser.parse_float_array(row['x_gse']),
            y_gse=self.parser.parse_float_array(row['y_gse']),
            z_gse=self.parser.parse_float_array(row['z_gse']),
            freq_min=self.parser.parse_float_array(row['min_f_bound']),
            freq_max=self.parser.parse_float_array(row['max_f_bound']),
            radius=self.parser.parse_float_array(row['radius']),
            lat_gse=self.parser.parse_float_array(row['lat_gse']),
            lon_gse=self.parser.parse_float_array(row['lon_gse']),
            lt_gse=self.parser.parse_float_array(row['LT_gse'])
        )
    
    def _create_metadata(self, row: pd.Series) -> BurstMetadata:
        """
        Create burst metadata from CSV row.
        
        Parameters
        ----------
        row : pd.Series
            CSV row
            
        Returns
        -------
        BurstMetadata
            Burst metadata
        """
        return BurstMetadata(
            stime=row['stime'],
            etime=row['etime']
        )
    
    def _create_properties(
        self,
        metadata: BurstMetadata,
        timeseries: BurstTimeSeries
    ) -> Dict[str, Any]:
        """
        Create properties dictionary.
        
        Parameters
        ----------
        metadata : BurstMetadata
            Burst metadata
        timeseries : BurstTimeSeries
            Time series data
            
        Returns
        -------
        dict
            Properties dictionary
        """
        # Calculate means
        properties = {
            "stime": metadata.stime_iso,
            "etime": metadata.etime_iso,
            "duration_seconds": float(metadata.duration_seconds),
            "n_points": timeseries.n_points,
            
            # Mean values
            "x_gse_mean": float(np.nanmean(timeseries.x_gse)),
            "y_gse_mean": float(np.nanmean(timeseries.y_gse)),
            "z_gse_mean": float(np.nanmean(timeseries.z_gse)),
            "radius_mean": float(np.nanmean(timeseries.radius)),
            "lat_gse_mean": float(np.nanmean(timeseries.lat_gse)),
            "lon_gse_mean": float(np.nanmean(timeseries.lon_gse)),
            "lt_gse_mean": float(np.nanmean(timeseries.lt_gse)),
            
            # Time series (convert to list, removing NaNs)
            "timestamps": self._array_to_list(timeseries.timestamps),
            "x_gse_timeseries": self._array_to_list(timeseries.x_gse),
            "y_gse_timeseries": self._array_to_list(timeseries.y_gse),
            "z_gse_timeseries": self._array_to_list(timeseries.z_gse),
            "radius_timeseries": self._array_to_list(timeseries.radius),
            "lat_gse_timeseries": self._array_to_list(timeseries.lat_gse),
            "lon_gse_timeseries": self._array_to_list(timeseries.lon_gse),
            "lt_gse_timeseries": self._array_to_list(timeseries.lt_gse),
            "freq_min_timeseries": self._array_to_list(timeseries.freq_min),
            "freq_max_timeseries": self._array_to_list(timeseries.freq_max),
        }
        
        return properties
    
    @staticmethod
    def _array_to_list(arr: np.ndarray) -> List[float]:
        """
        Convert numpy array to list, handling NaNs.
        
        Parameters
        ----------
        arr : np.ndarray
            Input array
            
        Returns
        -------
        List[float]
            List of floats
        """
        # Keep NaNs in the list for now - they can be filtered later if needed
        return [float(x) for x in arr]


# ============================================================================
# Main Converter
# ============================================================================

class CSVToTFCatConverter:
    """Main orchestrator for CSV to TFCat conversion."""
    
    def __init__(self, config: Config):
        """
        Initialize converter.
        
        Parameters
        ----------
        config : Config
            Configuration object
        """
        self.config = config
        self.burst_converter = BurstConverter(config)
        self.metadata_builder = TFCatMetadata(config)
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging.
        
        Returns
        -------
        logging.Logger
            Configured logger
        """
        level = self.config.logging['level']
        logging.basicConfig(
            level=getattr(logging, level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def convert(self, csv_path: Path, output_path: Path) -> FeatureCollection:
        """
        Convert CSV to TFCat and save to file.
        
        Parameters
        ----------
        csv_path : Path
            Input CSV file path
        output_path : Path
            Output JSON file path
            
        Returns
        -------
        FeatureCollection
            TFCat FeatureCollection object
        """
        self.logger.info("="*70)
        self.logger.info("CSV to TFCat Conversion Started")
        self.logger.info("="*70)
        
        # Load CSV
        df = self._load_csv(csv_path)
        
        # Convert bursts to features
        features = self._convert_bursts(df)
        
        # Build TFCat collection
        collection = self._build_collection(features, df)
        
        # Save to file
        self._save_json(collection, output_path)
        
        self.logger.info("="*70)
        self.logger.info("Conversion Complete!")
        self.logger.info("="*70)
        
        return collection
    
    def _load_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        Load and validate CSV file.
        
        Parameters
        ----------
        csv_path : Path
            CSV file path
            
        Returns
        -------
        pd.DataFrame
            Loaded DataFrame
            
        Raises
        ------
        FileNotFoundError
            If CSV file not found
        ValueError
            If required columns missing
        """
        self.logger.info(f"Loading CSV: {csv_path}")
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        self.logger.info(f"Loaded {len(df)} rows")
        
        # Validate columns
        required = set(self.config.get_required_columns())
        available = set(df.columns)
        missing = required - available
        
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Available columns: {available}"
            )
        
        # Parse datetime columns
        self.logger.info("Parsing datetime columns...")
        df['stime'] = pd.to_datetime(df['stime'])
        df['etime'] = pd.to_datetime(df['etime'])
        
        # Limit rows if configured
        max_bursts = self.config.conversion['max_bursts']
        if max_bursts is not None and max_bursts < len(df):
            self.logger.info(f"Limiting to first {max_bursts} bursts")
            df = df.head(max_bursts)
        
        self.logger.info(f"✅ CSV loaded and validated")
        return df
    
    def _convert_bursts(self, df: pd.DataFrame) -> List[Feature]:
        """
        Convert all DataFrame rows to TFCat features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns
        -------
        List[Feature]
            List of TFCat features
        """
        self.logger.info(f"\nConverting {len(df)} bursts to features...")
        
        features: List[Feature] = []
        failed_count = 0
        progress_interval = self.config.logging['progress_interval']
        
        for idx, row in df.iterrows():
            try:
                feature = self.burst_converter.convert(row, feature_id=idx)
                features.append(feature)
                
                if (idx + 1) % progress_interval == 0:
                    self.logger.info(f"  Processed {idx + 1}/{len(df)} bursts")
            
            except Exception as e:
                self.logger.warning(f"Failed to convert burst {idx}: {e}")
                failed_count += 1
        
        self.logger.info(f"\n✅ Converted {len(features)} bursts successfully")
        if failed_count > 0:
            self.logger.warning(f"❌ Failed: {failed_count} bursts")
        
        return features
    
    def _build_collection(
        self,
        features: List[Feature],
        df: pd.DataFrame
    ) -> FeatureCollection:
        """
        Build complete TFCat FeatureCollection.
        
        Parameters
        ----------
        features : List[Feature]
            List of features
        df : pd.DataFrame
            Original DataFrame (for time range)
            
        Returns
        -------
        FeatureCollection
            Complete TFCat collection
        """
        self.logger.info("\nBuilding TFCat collection...")
        
        # Build metadata
        crs = self.metadata_builder.build_crs()
        fields = self.metadata_builder.build_fields()
        properties = self.metadata_builder.build_properties(
            total_bursts=len(features),
            time_min=df['stime'].min(),
            time_max=df['etime'].max()
        )
        
        collection = FeatureCollection(
            features=features,
            crs=crs,
            fields=fields,
            properties=properties
        )
        
        self.logger.info("✅ TFCat collection built")
        return collection
    
    def _save_json(self, collection: FeatureCollection, output_path: Path) -> None:
        """
        Save FeatureCollection to JSON file.
        
        Parameters
        ----------
        collection : FeatureCollection
            TFCat collection
        output_path : Path
            Output file path
        """
        self.logger.info(f"\nSaving to: {output_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with proper formatting
        with open(output_path, 'w') as f:
            json.dump(collection, f, indent=2)
        
        # Get file size
        size_mb = output_path.stat().st_size / 1e6
        self.logger.info(f"✅ Saved! File size: {size_mb:.2f} MB")