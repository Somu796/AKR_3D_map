# Wind AKR Data Pipeline
Pipeline to convert Wind AKR burst CSV data to TFCat JSON format and load into MongoDB.

## Overview
This pipeline performs the following steps:

1. **CSV → TFCat**: Convert raw CSV to standardized TFCat JSON format
2. **TFCat → MongoDB**: Load TFCat data into MongoDB with proper indexing

## Prerequisites

- Python 3.12+
- MongoDB running (remote)
- Required Python packages: `tfcat`, `pandas`, `numpy`, `pymongo`, `pyyaml`

## Installation
```bash
# Install dependencies
uv add tfcat pandas numpy pymongo pyyaml
```
## Usage
### Option 1: Run Complete Pipeline
```bash
chmod +x pipeline/run_all.sh
./pipeline/run_all.sh
```

### Option 2: Run Steps Individually
```bash
# Step 1: Convert CSV to TFCat
uv run pipeline/01_csv_to_tfcat.py

# Step 2: Load TFCat to MongoDB
uv run pipeline/02_tfcat_to_mongo.py
```

## Configuration

Edit `pipeline/config.yaml` to customize:

- Catalog metadata (title, instrument, etc.)
- Conversion settings (validation, filtering)
- File paths
- Logging level

## Output

### TFCat JSON (`data/processed/wind_akr_bursts.json`)

Standardized TFCat format with:
- **Features**: Array of burst events
- **CRS**: Coordinate reference system
- **Fields**: Data dictionary with units and UCDs
- **Properties**: Catalog-level metadata

### MongoDB (`akr_database.wind_bursts`)

Collection with:
- Indexed burst documents
- Converted datetime fields (ISODate)
- Optimized for spatial and temporal queries

## Troubleshooting

### CSV not found
```
Error: CSV file not found: data/raw/wind_akr_bursts.csv
```
**Solution**: Ensure CSV file is in correct location

### MongoDB connection failed
```
Error: Could not connect to MongoDB
```
**Solution**: 
- Check if MongoDB is running: `mongod --version`
- Set MONGO_URI environment variable: `export MONGO_URI=mongodb://localhost:27017/`

### Conversion errors
Check logs for specific burst failures. Common issues:
- Insufficient valid data points
- Invalid timestamps
- Missing required columns

## Architecture
```
CSV File
   ↓ [01_csv_to_tfcat.py using converter.py]
TFCat JSON
   ↓ [02_tfcat_to_mongo.py]
MongoDB
```

## Module Structure

- `config.yaml`: Configuration file
- `converter.py`: OOP converter library with type hints
  - `Config`: Configuration management
  - `BurstTimeSeries`: Time series data model
  - `TimeSeriesParser`: CSV array parsing
  - `GeometryBuilder`: TFCat polygon creation
  - `TFCatMetadata`: Metadata builders
  - `BurstConverter`: Single burst conversion
  - `CSVToTFCatConverter`: Main orchestrator
- `01_csv_to_tfcat.py`: CLI script for Step 1
- `02_tfcat_to_mongo.py`: CLI script for Step 2
- `run_all.sh`: Automated pipeline execution