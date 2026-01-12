# Wind AKR Data in TFCat Format: Complete Guide

**Purpose:** This document explains how Wind spacecraft Auroral Kilometric Radiation (AKR) burst data is represented in TFCat format.
**Author:** Sudipta Hazra  
**Date:** January 2026  
**Version:** 1.0


## Table of Contents

1. [What is TFCat?](#what-is-tfcat)
2. [Why Use TFCat?](#why-use-tfcat)
3. [Overall File Structure](#overall-file-structure)
4. [Your Data: Before and After](#your-data-before-and-after)
5. [Section 1: CRS (Coordinate Reference System)](#section-1-crs-coordinate-reference-system)
6. [Section 2: Fields (Data Dictionary)](#section-2-fields-data-dictionary)
7. [Section 3: Properties (Catalog Metadata)](#section-3-properties-catalog-metadata)
8. [Section 4: Features (Individual Bursts)](#section-4-features-individual-bursts)
9. [Understanding Geometry](#understanding-geometry)
10. [Understanding Properties](#understanding-properties)
11. [Complete Example](#complete-example)
12. [How to Use This Data](#how-to-use-this-data)
13. [Glossary](#glossary)

---

## What is TFCat?

### Definition

**TFCat** = **T**ime-**F**requency **Cat**alogue

It is a **standardised JSON format** for storing data about phenomena that occur in both **time** and **frequency** (like radio emissions).

**Original Article**: *Cecconi B, Louis CK, Bonnin X, Loh A and Taylor MB (2023) Time-frequency catalogue: JSON implementation and python library. Front. Astron. Space Sci. 9:1049677. doi: [10.3389/fspas.2022.1049677](https://doi.org/10.3389/fspas.2022.1049677)*

### Key Characteristics

**Standardised** - Can be used as a universal format.
**Self-documenting** - Includes descriptions and units of each entry  
**JSON-based** - One of the famous data formats can be saved in cloud databases such as MongoDB    

## Why Use TFCat?

| Feature | Benefit | Example |
|---------|---------|---------|
| **Built-in units** | Clear, unambiguous | `"unit": "R_E"` = Earth radii |
| **CRS definition** | Standard reference | `"time_coords_id": "unix"` = Unix timestamp |
| **Field metadata** | Self-explanatory | `"info": "Mean X position (GSE)"` |
| **Structured arrays** | Easy to parse | `[13.74, 13.739, 13.736]` |
| **Explicit geometry** | Clear relationships | Polygon in time-frequency space |
| **Schema validation** | Catches errors | Ensures data correctness |

---

## Overall File Structure

### High-Level View

A TFCat file has **four main sections**:

```
# Simple explanation:
1. CRS: How to interpret coordinate (SHARED by all bursts)

2. Fields: What each data field means (SHARED by all bursts)

3. Properties: Catalog information (SHARED by all bursts)

4. Features: Individual bursts (ARRAY of bursts)

# From original article:
1. TFCat Coordinate Reference System: A TFCat Coordination Reference System (CRS) contains a description of the time axis, a description of the spectral axis, and a reference position. The TFCat CRS could be of three types: name, link or local.

2. TFCat Field: A TFCat Field is defining the TFCat feature properties. Its minimal content includes a name, a description, a data type and a UCD (Unified Content Descriptors).

3. TFCat Feature: A TFCat Feature contains one or several TFCat Geometries, and a set of properties, provided as a set of (key, value) pairs.

4. TFCat Property: A TFCat Property is a (key, value) pair. There are two types of TFCat Properties: 
  a.  TFCat Feature Property: TFCat feature properties are used in a TFCat Feature. They must be defined with a corresponding TFCat Field.

  b. TFCat Collection Property: TFCat collection properties are used in a TFCat Feature Collection. They should provide generic information on the catalogue, such as: title, authors, reference, instrument, etc

5. TFCat Geometry: There are 7 types of TFCat Geometry: Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon, with an additional GeometryCollection, which is a set of any of the six other geometry types. Each geometry is composed of a set of coordinate pairs. A LineString is defined by a list of connected Point features, forming a continuous path. A Polygon is defined by a closed list of connected Point features, forming a closed contour.

```

### Visual Structure

```json
{
  "type": "FeatureCollection",
  
  "crs": {
    /* Section 1: Coordinate Reference System */
    /* Defines: time format, frequency units, reference frame */
  },
  
  "fields": {
    /* Section 2: Data Dictionary */
    /* Defines: what each property means, units, data types */
  },
  
  "properties": {
    /* Section 3: Catalog Metadata */
    /* Contains: title, instrument info, time range, total bursts */
  },
  
  "features": [
    /* Section 4: Individual Bursts */
    { /* Burst 0 */ },
    { /* Burst 1 */ },
    { /* Burst 2 */ },
    /* ... 5,000+ bursts total ... */
  ]
}
```

## AKR Burst Data: Before and After

### Before: CSV Format

**File:** `wind_akr_bursts.csv`

```csv
stime,etime,burst_timestamp,min_f_bound,max_f_bound,x_gse,y_gse,z_gse,radius,lat_gse,lon_gse,LT_gse
1995-04-23 00:45:38.992,1995-04-23 01:07:01.840,"1995-04-23T00:45:38.992,1995-04-23T00:46:40.992,...","540.0,388.0,388.0,...","80.0,80.0,80.0,104.0,...","13.74,13.739413,...","234.138,234.14,...","57.23,57.23,...","241.178,241.18,...","2.0,2.0,...","8.43,8.429,...","12.916,12.916,..."
```

**Problems: Data is not self-explanatory**
- No units specified within the data
- No coordinate system defined
- No field descriptions
- No relationship between time and frequency shown

---

### After: TFCat Format (Has to be modified: https://gitlab.obspm.fr/maser/catalogues/catalogue-format/-/blob/master/json/spec.md?plain=0#14-example)

**File:** `wind_akr_bursts.json`

```json
{
  "type": "FeatureCollection",
  "crs": {
    "type": "local",
    "properties": {
      "time_coords_id": "unix",              /* Time in Unix seconds */
      "spectral_coords": {
        "type": "frequency",
        "unit": "kHz"                        /* Frequency in kilohertz */
      }
    }
  },
  "fields": {
    "x_gse_mean": {
      "info": "Mean X position (GSE)",       /* Field description */
      "unit": "R_E"                          /* Units: Earth radii */
    }
  },
  "features": [
    {
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [798606338.99, 80.0],              /* [time, frequency] pairs */
          [798606400.99, 80.0],
          /* ... more coordinates ... */
        ]
      },
      "properties": {
        "stime": "1995-04-23T00:45:38.992Z",
        "x_gse_timeseries": [13.74, 13.739, ...] /* Real arrays */
      }
    }
  ]
}
```

**Improvements:**
- Arrays are proper JSON arrays
- Units explicitly defined
- Coordinate system specified
- Every field has description
- Time-frequency relationship shown as geometry

---

## Section 1: CRS (Coordinate Reference System)

### What is CRS?

**CRS** = Instructions for interpreting coordinates

Think of it like **units on a map**:
- Without CRS: "Location is 10, 20" (10 what? 20 what?)
- With CRS: "Location is 10 degrees latitude, 20 degrees longitude"

### Your Data's CRS

```json
"crs": {
  "type": "local",
  "properties": {
    "name": "Time-Frequency",
    "time_coords_id": "unix",
    "spectral_coords": {
      "type": "frequency",
      "unit": "kHz"
    },
    "ref_position_id": "wind"
  }
}
```

### What Each Part Means

| Field | Value | Explanation | Example |
|-------|-------|-------------|---------|
| **type** | `"local"` | Custom coordinate system (not geographic) | Not lat/lon, it's time/frequency |
| **name** | `"Time-Frequency"` | The two axes of the coordinate space | X-axis = time, Y-axis = frequency |
| **time_coords_id** | `"unix"` | Time format = Unix timestamps | Seconds since Jan 1, 1970 00:00:00 UTC |
| **spectral_coords.type** | `"frequency"` | Y-axis represents frequency | Not wavelength or energy |
| **spectral_coords.unit** | `"kHz"` | Frequency measured in kilohertz | 80 kHz, 540 kHz, etc. |
| **ref_position_id** | `"wind"` | Observations from Wind spacecraft | All positions relative to Wind |

### Example: Reading a Coordinate

Given coordinate: `[798606338.99, 80.0]`

**Using CRS to interpret:**
1. **First number (798606338.99):**
   - CRS says: `"time_coords_id": "unix"`
   - Meaning: Unix timestamp = 798606338.99 seconds
   - Converts to: 1995-04-23 00:45:38.99 UTC

2. **Second number (80.0):**
   - CRS says: `"spectral_coords": {"type": "frequency", "unit": "kHz"}`
   - Meaning: 80.0 kilohertz
   - Full interpretation: "At 1995-04-23 00:45:38.99, the frequency was 80.0 kHz"

### Why This Matters

**Without CRS:**
```
"What does [798606338.99, 80.0] mean?"
 - Unknown! Could be anything.
```

**With CRS:**
```
"What does [798606338.99, 80.0] mean?"
 - At time 1995-04-23 00:45:38.99 UTC, the frequency was 80.0 kHz
 - Observation made from Wind spacecraft
```

---

## Section 2: Fields (Data Dictionary)

### What are Fields?

**Fields** = Definitions of what each property means

Think of it like a **glossary** at the back of a book:
- You see an unfamiliar term in the text
- You look it up in the glossary
- You understand what it means

### Structure of a Field Definition

Each field has these attributes:

```json
"field_name": {
  "info": "Human-readable description",
  "datatype": "int | float | str | bool",
  "ucd": "Unified Content Descriptor (astronomy standard)",
  "unit": "Measurement unit (if applicable)"
}
```

### Example Field Definitions from Your Data

#### Example 1: Start Time

```json
"stime": {
  "info": "Burst start time (ISO 8601 format)",
  "datatype": "str",
  "ucd": "time.start"
}
```

**Translation:**
- **Field name:** `stime`
- **What it is:** When the burst began
- **Format:** ISO 8601 string (e.g., "1995-04-23T00:45:38.992Z")
- **Data type:** String (text)
- **Standard code:** `time.start` (universally recognized as "start time")

**Example value:** `"stime": "1995-04-23T00:45:38.992Z"`

---

#### Example 2: Mean X Position

```json
"x_gse_mean": {
  "info": "Mean spacecraft X position during burst (GSE coordinates)",
  "datatype": "float",
  "ucd": "pos.cartesian.x",
  "unit": "R_E"
}
```

**Translation:**
- **Field name:** `x_gse_mean`
- **What it is:** Average X-coordinate of Wind spacecraft during the burst
- **Coordinate system:** GSE (Geocentric Solar Ecliptic)
- **Data type:** Floating-point number
- **Standard code:** `pos.cartesian.x` (X position in Cartesian coordinates)
- **Units:** R_E (Earth radii, where 1 R_E = 6,371 km)

**Example value:** `"x_gse_mean": 13.74`  
**Interpretation:** Wind was, on average, 13.74 Earth radii from Earth's center in the X direction (toward the Sun)

---

#### Example 3: Duration

```json
"duration_seconds": {
  "info": "Burst duration in seconds",
  "datatype": "float",
  "ucd": "time.duration",
  "unit": "s"
}
```

**Translation:**
- **Field name:** `duration_seconds`
- **What it is:** How long the burst lasted
- **Data type:** Floating-point number
- **Standard code:** `time.duration`
- **Units:** s (seconds)

**Example value:** `"duration_seconds": 1282.85`  
**Interpretation:** This burst lasted 1282.85 seconds (about 21.4 minutes)

---

#### Example 4: Position Time Series

```json
"x_gse_timeseries": {
  "info": "Spacecraft X position time series (GSE)",
  "datatype": "float",
  "ucd": "pos.cartesian.x",
  "unit": "R_E"
}
```

**Translation:**
- **Field name:** `x_gse_timeseries`
- **What it is:** Array of X-coordinates showing how position changed over time
- **Data type:** Array of floating-point numbers
- **Standard code:** `pos.cartesian.x`
- **Units:** R_E (Earth radii)

**Example value:** `"x_gse_timeseries": [13.74, 13.739, 13.736, 13.734, ...]`  
**Interpretation:** Wind's X-position during the burst, measured at each time point

---

### Complete Field List for Your Data

Here are ALL the fields defined in your TFCat file:

| Field Name | Description | Data Type | Unit |
|------------|-------------|-----------|------|
| `stime` | Burst start time | string | - |
| `etime` | Burst end time | string | - |
| `duration_seconds` | Burst duration | float | seconds |
| `n_points` | Number of measurements | integer | - |
| `x_gse_mean` | Mean X position | float | R_E |
| `y_gse_mean` | Mean Y position | float | R_E |
| `z_gse_mean` | Mean Z position | float | R_E |
| `radius_mean` | Mean radial distance | float | R_E |
| `lat_gse_mean` | Mean latitude (GSE) | float | degrees |
| `lon_gse_mean` | Mean longitude (GSE) | float | degrees |
| `lt_gse_mean` | Mean local time (GSE) | float | hours |
| `timestamps` | Unix timestamp array | float array | seconds |
| `x_gse_timeseries` | X position array | float array | R_E |
| `y_gse_timeseries` | Y position array | float array | R_E |
| `z_gse_timeseries` | Z position array | float array | R_E |
| `radius_timeseries` | Radial distance array | float array | R_E |
| `lat_gse_timeseries` | Latitude array | float array | degrees |
| `lon_gse_timeseries` | Longitude array | float array | degrees |
| `lt_gse_timeseries` | Local time array | float array | hours |
| `freq_min_timeseries` | Lower frequency bound array | float array | kHz |
| `freq_max_timeseries` | Upper frequency bound array | float array | kHz |

### Why Fields Matter

**Scenario: You want to know Wind's position**

**Without field definitions:**
```json
"x_gse_mean": 13.74
```
❓ Questions: What is this? What units? What coordinate system?

**With field definitions:**
```json
// Look up "x_gse_mean" in fields:
{
  "info": "Mean spacecraft X position during burst (GSE coordinates)",
  "unit": "R_E"
}

// Now you know:
"x_gse_mean": 13.74  →  Wind was 13.74 Earth radii from Earth center 
                         in the X direction (GSE coordinates)
```

Complete understanding without guessing!


## Section 3: Properties (Catalog Metadata)

### What are Properties?

**Properties** = Information about the **entire catalog** (not individual bursts)

Think of it like **book cover information**:
- Title of the book
- Author
- Publication date
- Number of pages

### Your Catalog Properties

```json
"properties": {
  "title": "Wind AKR Burst Catalog",
  "instrument_host_name": "Wind",
  "instrument_name": "WAVES",
  "target_name": "Earth",
  "target_region": "magnetosphere",
  "feature_name": "Auroral Kilometric Radiation",
  "coordinate_system": "GSE",
  "total_bursts": 1000,
  "time_min": "1995-04-23T00:00:00Z",
  "time_max": "2004-12-31T23:59:59Z",
  "creation_date": "2025-01-09T12:34:56Z",
  "version": "1.0"
}
```

### What Each Property Means

| Property | Value | Explanation |
|----------|-------|-------------|
| **title** | "Wind AKR Burst Catalog" | Name of this catalog |
| **instrument_host_name** | "Wind" | Spacecraft that made observations |
| **instrument_name** | "WAVES" | Instrument onboard Wind that detected AKR |
| **target_name** | "Earth" | What was being observed |
| **target_region** | "magnetosphere" | Specific region around Earth |
| **feature_name** | "Auroral Kilometric Radiation" | Type of phenomenon detected |
| **coordinate_system** | "GSE" | Coordinate frame used (Geocentric Solar Ecliptic) |
| **total_bursts** | 5432 | Number of bursts in this catalog |
| **time_min** | "1995-04-23..." | Earliest burst in catalog |
| **time_max** | "2004-12-31..." | Latest burst in catalog |
| **creation_date** | "2025-01-09..." | When this TFCat file was created |
| **version** | "1.0" | Catalog version number |

### Why This Matters

When you share this file with someone, they can immediately see:
- What mission (Wind)
- What instrument (WAVES)
- What phenomenon (AKR)
- Time coverage (1995-2004)
- How many events (1000 bursts)

## Section 4: Features (Individual Bursts)

### What are Features?

**Features** = Individual burst events (one per CSV row)

Think of it like **chapters in a book**:
- The catalog is the book
- Each feature is a chapter
- Each chapter tells the story of one burst

### Structure of a Feature

Every feature has three parts:

```json
{
  "type": "Feature",
  "id": 0,                    // Unique identifier
  "geometry": { ... },        // Shape in time-frequency space
  "properties": { ... }       // All measurements for this burst
}
```

### The Three Components

#### 1. ID (Identifier)

```json
"id": 0
```

**What it is:** A unique number for this burst  
**Range:** 0 to 1000 (for 1000 bursts)  
**Purpose:** Reference specific bursts  

**Example usage:**  
"Show me burst #42" -> Look for feature with `"id": 42`

#### 2. Geometry (Time-Frequency Shape)

```json
"geometry": {
  "type": "Polygon",
  "coordinates": [
    [798606338.99, 80.0],
    [798606400.99, 80.0],
    [798607621.84, 540.0],
    // ... more points ...
  ]
}
```

**What it is:** The burst's shape in time-frequency space  
**Format:** Array of `[time, frequency]` coordinate pairs  
**Purpose:** Visualize how the burst evolved  

**See section "Understanding Geometry" for details**

#### 3. Properties (Measurements)

```json
"properties": {
  "stime": "1995-04-23T00:45:38.992Z",
  "etime": "1995-04-23T01:07:01.840Z",
  "duration_seconds": 1282.85,
  "x_gse_mean": 13.74,
  "x_gse_timeseries": [13.74, 13.739, ...],
  // ... more properties ...
}
```
**What it is:** All data about this burst  
**Format:** Key-value pairs  
**Purpose:** Store measurements, positions, times  

**See section "Understanding Properties" for details**

## Understanding Geometry

### What is Geometry?

**Geometry** describes the **shape of a burst in time-frequency space**.

### Visual Explanation

Imagine a graph with:
- **X-axis** = Time (when)
- **Y-axis** = Frequency (what frequencies were detected)

The burst traces a shape on this graph (taken from claude):

```
Frequency (kHz)
    ↑
540 |           •─────────•         ← Maximum frequency detected
    |          ╱           ╲
388 |         •             •       ← Minimum frequency detected
    |        ╱               ╲
104 |       •                 •
    |      ╱                   ╲
 80 |─────•─────────────────────•──→ Time
    00:45:38              01:07:01
    
    └───── Burst duration ──────┘
         (~21 minutes)
```

### How Geometry is Stored

The outline of this shape is stored as a **polygon**:

```json
"geometry": {
  "type": "Polygon",
  "coordinates": [
    // Top edge (forward in time at maximum frequency)
    [798606338.99, 80.0],      // Point 1: Start time, max freq
    [798606400.99, 80.0],      // Point 2: Next time, max freq
    [798606462.99, 80.0],      // Point 3: Next time, max freq
    // ... more points ...
    [798607621.84, 540.0],     // Last point at top
    
    // Right edge (going down to minimum frequency)
    [798607621.84, 388.0],     // Last time, min freq
    
    // Bottom edge (backward in time at minimum frequency)
    [798606462.99, 388.0],     // Backward, min freq
    [798606400.99, 388.0],     // Backward, min freq
    [798606338.99, 540.0],     // Back to start time, min freq
    
    // Close the polygon
    [798606338.99, 80.0]       // Same as first point (closes loop)
  ]
}
```

### Understanding Each Coordinate

Each coordinate is: `[time_in_unix_seconds, frequency_in_kHz]`

**Example:** `[798606338.99, 80.0]`

**Reading this coordinate:**
1. **Time:** 798606338.99 Unix seconds
   - Converts to: 1995-04-23 00:45:38.99 UTC
   - Meaning: "At this moment..."

2. **Frequency:** 80.0 kHz
   - Meaning: "...the burst had a frequency of 80.0 kilohertz"

**Full interpretation:**  
"At 1995-04-23 00:45:38.99 UTC, the AKR burst was detected at 80.0 kHz"

### Why Polygon?

The polygon traces the **boundary** of the burst:
- **Top edge** = Maximum frequency at each time
- **Bottom edge** = Minimum frequency at each time
- **Left edge** = Start of burst (going from min to max freq)
- **Right edge** = End of burst (going from max to min freq)

This shows:
- When burst started/ended
- What frequency range was covered
- How frequency changed over time

### Real Example

**Burst observed from 00:45:38 to 01:07:01:**

```
At 00:45:38 -> Frequencies: 80-540 kHz (wide range)
At 00:46:40 -> Frequencies: 80-388 kHz (narrowed)
At 00:47:42 -> Frequencies: 80-388 kHz (stayed narrow)
At 01:07:01 -> Frequencies: 388-540 kHz (shifted up)
```

The polygon captures this evolution!

## Understanding Properties

### What are Properties?

**Properties** = All the **measurements and metadata** for one burst.

### Two Types of Properties

#### Type 1: Summary Statistics (Single Values)

These give you **quick information** about the burst:

```json
"properties": {
  "stime": "1995-04-23T00:45:38.992Z",     // When burst started
  "etime": "1995-04-23T01:07:01.840Z",     // When burst ended
  "duration_seconds": 1282.85,              // How long it lasted
  "n_points": 23,                           // Number of measurements
  
  "x_gse_mean": 13.74,                      // Average X position
  "y_gse_mean": 234.14,                     // Average Y position
  "z_gse_mean": 57.23,                      // Average Z position
  "radius_mean": 241.18,                    // Average distance from Earth
  "lat_gse_mean": 2.0,                      // Average latitude
  "lon_gse_mean": 8.43,                     // Average longitude
  "lt_gse_mean": 12.916                     // Average local time
}
```

**Purpose:** Fast queries and quick understanding

**Example use:**
```
"Find all bursts where Wind was between X=10 and X=15"
-> Filter by x_gse_mean
```

---

#### Type 2: Time Series (Arrays)

These give you **detailed evolution** during the burst:

```json
"properties": {
  "timestamps": [798606338.99, 798606400.99, 798606462.99, ...],
  
  "x_gse_timeseries": [13.74, 13.739413, 13.736867, ...],
  "y_gse_timeseries": [234.138, 234.14, 234.14, ...],
  "z_gse_timeseries": [57.23, 57.23, 57.23, ...],
  
  "radius_timeseries": [241.178, 241.18, 241.18, ...],
  "lat_gse_timeseries": [2.0, 2.0, 2.0, ...],
  "lon_gse_timeseries": [8.43, 8.429, 8.426, ...],
  "lt_gse_timeseries": [12.915833, 12.915817, ...],
  
  "freq_min_timeseries": [540.0, 388.0, 388.0, ...],
  "freq_max_timeseries": [80.0, 80.0, 80.0, 104.0, ...]
}
```

**Purpose:** Detailed analysis and tracking changes

**Example use:**
```
"How did Wind's position change during burst #42?"
→ Plot x_gse_timeseries, y_gse_timeseries, z_gse_timeseries vs timestamps
```

---

### Understanding the Arrays

All time series arrays are **synchronized** - same length, same timestamps:

```
Index:       0              1             2            ...
timestamps: [798606338.99, 798606400.99, 798606462.99, ...]
x_gse:      [13.74,  13.739,  13.736,  13.734, ...]
y_gse:      [234.14, 234.14,  234.14,  234.13, ...]
freq_min:   [540.0,  388.0,   388.0,   540.0,  ...]
freq_max:   [80.0,   80.0,    80.0,    80.0,   ...]
```

**At index 0:**
- Time: 798606338.99 (00:45:38.99)
- Position: X=13.74, Y=234.14, Z=57.23 R_E
- Frequency: 80-540 kHz

**At index 1:**
- Time: 798606400.99 (00:46:40.99)
- Position: X=13.739, Y=234.14, Z=57.23 R_E
- Frequency: 80-388 kHz (narrowed!)

---

### Complete Property List

Every burst has these properties:

**Temporal (Time-related):**
- `stime` - Start time (ISO 8601 string)
- `etime` - End time (ISO 8601 string)
- `duration_seconds` - Duration (float)
- `n_points` - Number of measurements (integer)
- `timestamps` - Time array (float array, Unix seconds)

**Spatial - Mean Values:**
- `x_gse_mean` - Mean X position (float, R_E)
- `y_gse_mean` - Mean Y position (float, R_E)
- `z_gse_mean` - Mean Z position (float, R_E)
- `radius_mean` - Mean radial distance (float, R_E)
- `lat_gse_mean` - Mean latitude (float, degrees)
- `lon_gse_mean` - Mean longitude (float, degrees)
- `lt_gse_mean` - Mean local time (float, hours)

**Spatial - Time Series:**
- `x_gse_timeseries` - X position array (float array, R_E)
- `y_gse_timeseries` - Y position array (float array, R_E)
- `z_gse_timeseries` - Z position array (float array, R_E)
- `radius_timeseries` - Radial distance array (float array, R_E)
- `lat_gse_timeseries` - Latitude array (float array, degrees)
- `lon_gse_timeseries` - Longitude array (float array, degrees)
- `lt_gse_timeseries` - Local time array (float array, hours)

**Spectral (Frequency-related):**
- `freq_min_timeseries` - Lower frequency bound array (float array, kHz)
- `freq_max_timeseries` - Upper frequency bound array (float array, kHz)

---

## Complete Example

Let's see ONE complete burst from start to finish:

### Original CSV Row

```csv
stime,etime,burst_timestamp,min_f_bound,max_f_bound,x_gse,y_gse,z_gse,radius,lat_gse,lon_gse,LT_gse
1995-04-23 00:45:38.992,1995-04-23 01:07:01.840,"1995-04-23T00:45:38.992,1995-04-23T00:46:40.992,1995-04-23T00:47:42.992","540.0,388.0,388.0","80.0,80.0,80.0","13.74,13.739,13.736","234.14,234.14,234.14","57.23,57.23,57.23","241.18,241.18,241.18","2.0,2.0,2.0","8.43,8.429,8.426","12.916,12.916,12.915"
```

---

### Complete TFCat Feature (? Isn't is repetitive, same info in coordinates, timestamps, x_gse, y_gse, z_gse)

```json
{
  "type": "Feature",
  "id": 0,
  
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [798606338.99, 80.0],
      [798606400.99, 80.0],
      [798606462.99, 80.0],
      [798607621.84, 540.0],
      [798607621.84, 388.0],
      [798606462.99, 388.0],
      [798606400.99, 388.0],
      [798606338.99, 540.0],
      [798606338.99, 80.0]
    ]
  },
  
  "properties": {
    "stime": "1995-04-23T00:45:38.992Z",
    "etime": "1995-04-23T01:07:01.840Z",
    "duration_seconds": 1282.85,
    "n_points": 3,
    
    "x_gse_mean": 13.74,
    "y_gse_mean": 234.14,
    "z_gse_mean": 57.23,
    "radius_mean": 241.18,
    "lat_gse_mean": 2.0,
    "lon_gse_mean": 8.43,
    "lt_gse_mean": 12.916,
    
    "timestamps": [798606338.99, 798606400.99, 798606462.99],
    "x_gse_timeseries": [13.74, 13.739, 13.736],
    "y_gse_timeseries": [234.14, 234.14, 234.14],
    "z_gse_timeseries": [57.23, 57.23, 57.23],
    "radius_timeseries": [241.18, 241.18, 241.18],
    "lat_gse_timeseries": [2.0, 2.0, 2.0],
    "lon_gse_timeseries": [8.43, 8.429, 8.426],
    "lt_gse_timeseries": [12.916, 12.916, 12.915],
    "freq_min_timeseries": [540.0, 388.0, 388.0],
    "freq_max_timeseries": [80.0, 80.0, 80.0]
  }
}
```

---

### Reading This Burst

**What happened:**

1. **When:** From 1995-04-23 00:45:38 to 01:07:01 (21.4 minutes)

2. **Where:** Wind spacecraft was at:
   - Average position: (13.74, 234.14, 57.23) Earth radii
   - About 241 Earth radii from Earth's center
   - In the magnetosphere, sunward side

3. **What:** AKR burst detected
   - Frequency range: 80-540 kHz
   - Started wide (80-540 kHz)
   - Narrowed (80-388 kHz)
   - Back to (80-540 kHz) at end

4. **Evolution:** Spacecraft moved slightly:
   - X: 13.74 -> 13.739 -> 13.736 R_E (small change)
   - Y: stayed at ~234.14 R_E
   - Z: stayed at ~57.23 R_E

---

## How to Use This Data

### Loading the TFCat File

**In Python:**

```python
import json

# Load file
with open('wind_akr_bursts.json') as f:
    tfcat = json.load(f)

# Access components
crs = tfcat['crs']
fields = tfcat['fields']
properties = tfcat['properties']
features = tfcat['features']

print(f"Total bursts: {len(features)}")
# Output: Total bursts: 5432
```

---

### Accessing Individual Bursts

```python
# Get first burst
burst = features[0]

# Access ID
burst_id = burst['id']
print(f"Burst ID: {burst_id}")

# Access geometry
geometry = burst['geometry']
coordinates = geometry['coordinates']
print(f"First coordinate: {coordinates[0]}")
# Output: First coordinate: [798606338.99, 80.0]

# Access properties
props = burst['properties']
start_time = props['stime']
x_position = props['x_gse_mean']
print(f"Start: {start_time}, X position: {x_position} R_E")
# Output: Start: 1995-04-23T00:45:38.992Z, X position: 13.74 R_E
```

---

### Querying Data

**Example 1: Find bursts in a specific region**

```python
# Find bursts where X is between 10 and 15 R_E
bursts_in_region = [
    f for f in features
    if 10 <= f['properties']['x_gse_mean'] <= 15
]

print(f"Found {len(bursts_in_region)} bursts in region")
```

**Example 2: Find long-duration bursts**

```python
# Find bursts lasting more than 30 minutes (1800 seconds)
long_bursts = [
    f for f in features
    if f['properties']['duration_seconds'] > 1800
]

print(f"Found {len(long_bursts)} long bursts")
```

**Example 3: Get time series for analysis**

```python
# Get position time series for first burst
burst = features[0]
times = burst['properties']['timestamps']
x_pos = burst['properties']['x_gse_timeseries']
y_pos = burst['properties']['y_gse_timeseries']
z_pos = burst['properties']['z_gse_timeseries']

# Plot trajectory
import matplotlib.pyplot as plt
plt.plot(x_pos, y_pos)
plt.xlabel('X (R_E)')
plt.ylabel('Y (R_E)')
plt.title('Wind trajectory during burst')
plt.show()
```

---

### Using TFCat Library

**There's a Python library specifically for TFCat:**

```python
from tfcat import TFCat

# Load TFCat file
cat = TFCat.from_file('wind_akr_bursts.json')

# Access features
burst = cat.feature(0)

# Get properties
x_mean = burst.properties['x_gse_mean']

# Convert time
unix_time = burst.geometry.coordinates[0][0]
astropy_time = cat.crs.time_converter(unix_time)
print(astropy_time.iso)
# Output: 1995-04-23 00:45:38.992

# Plot feature (built-in visualization)
cat.plot_feature(0)
```

---

## Glossary

### General Terms

**AKR (Auroral Kilometric Radiation):** Natural radio emissions from Earth's auroral regions, typically 50-800 kHz.

**Burst:** One continuous detection event of AKR, lasting from minutes to hours.

**Feature:** In TFCat terminology, one individual burst event.

**FeatureCollection:** A collection of multiple features (bursts) with shared metadata.

**JSON (JavaScript Object Notation):** A text-based data format that's human-readable and machine-parseable.

**TFCat (Time-Frequency Catalogue):** A standardized JSON format for time-frequency phenomena.

---

### Coordinate Systems

**GSE (Geocentric Solar Ecliptic):** Coordinate system with:
- **X-axis:** Points toward the Sun
- **Y-axis:** Points toward dusk (perpendicular to X in ecliptic plane)
- **Z-axis:** Points toward ecliptic north pole

**Unix Time:** Number of seconds since January 1, 1970, 00:00:00 UTC.

**R_E (Earth Radius):** 1 R_E = 6,371 kilometers (distance from Earth's center to surface).

---

### TFCat Terminology

**CRS (Coordinate Reference System):** Defines how to interpret coordinates (time format, frequency units, reference frame).

**Fields:** Data dictionary defining what each property means, including units and data types.

**Geometry:** The shape of a feature in coordinate space (for AKR: time-frequency polygon).

**Properties (Feature-level):** Measurements and data for one specific burst.

**Properties (Collection-level):** Metadata about the entire catalog.

**UCD (Unified Content Descriptor):** Standardized astronomy vocabulary for describing data fields.

---

### Spacecraft Terms

**Wind:** NASA spacecraft launched 1994, orbits Earth's L1 Lagrange point.

**WAVES:** Radio and plasma wave instrument onboard Wind.

**RAD1 and RAD2:** Two radio receivers within WAVES instrument.

**Local Time:** Time of day at spacecraft's location (0-24 hours).

**Magnetosphere:** Region around Earth dominated by Earth's magnetic field.

---

### Data Terms

**Time Series:** Array of values measured at different times, showing evolution.

**Mean:** Average value over time.

**Polygon:** Closed shape defined by connected points.

**ISO 8601:** International standard for date/time representation (e.g., "1995-04-23T00:45:38Z").

---
### Key Takeaways

1. **TFCat is self-documenting:** Every field has units and descriptions
2. **Nothing is lost:** All CSV data preserved (and better organized)
3. **It's standardized:** Follows established conventions
4. **It's practical:** Easy to load, query, and analyze
5. **It's shareable:** Anyone can understand and use it

### Next Steps

**To work with this data:**

1. Load the TFCat JSON file
2. Check the CRS to understand coordinates
3. Check Fields to understand properties
4. Access Features to analyze individual bursts
5. Use the time series for detailed analysis

**Resources:**

- TFCat specification: https://gitlab.obspm.fr/maser/catalogues/tfcat
- AKR burst data files captured by WIND with detailed specifications: https://maser-lira.obspm.fr/publications/doi/bursts-of-auroral-kilometric-148.html

---

**Questions?** Refer to specific sections above, or check the TFCat documentation.

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Author:** AKR Detection Probability Project
