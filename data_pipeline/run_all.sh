#!/bin/bash
set -e

echo "=========================================="
echo "  Wind AKR Data Pipeline"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    exit 1
fi

# Check if CSV exists
if [ ! -f "data/raw/wind_akr_bursts.csv" ]; then
    echo "❌ Error: CSV file not found: data/raw/wind_akr_bursts.csv"
    exit 1
fi

echo "Step 1/2: Converting CSV to TFCat..."
echo "--------------------------------------"
python pipeline/01_csv_to_tfcat.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Step 1 failed"
    exit 1
fi

echo ""
echo "Step 2/2: Loading TFCat to MongoDB..."
echo "--------------------------------------"
python pipeline/02_tfcat_to_mongo.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Step 2 failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "=========================================="
echo ""
echo "✅ CSV converted to TFCat"
echo "✅ Data loaded to MongoDB"
echo ""
echo "Next steps:"
echo "  1. Start backend: cd backend && uvicorn main:app --reload"
echo "  2. Open frontend: open frontend/index.html"
echo ""