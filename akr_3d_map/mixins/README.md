---

## `ObservationTimeCalculator`   Full Documentation

This is a mixin class that populates a 3D spatial grid with time and count statistics derived from spacecraft and AKR burst data. It requires the host class to implement four methods: `_validate_and_get_grid()`, `_validate_coord_colnames()`, `_assign_bin_indices()`, and `get_dimension_names()`. Missing any of these raises an `AttributeError` at runtime.

All public methods return `Self`, enabling method chaining:
```python
cart.add_residence_time(...).add_observation_time(...).add_normalised_observation_time(...)
```

All methods silently discard positions that fall outside grid boundaries (bin index < 0). No warning is raised.

---

### `_add_time_intervals(df, timestamp_colname, variable, gap_hours=2)`

**Private helper   not called directly by the user.**

#### Inputs

| Parameter | Type | Meaning |
|---|---|---|
| `df` | DataFrame | Either AKR burst or satellite trajectory data depending on `variable` |
| `timestamp_colname` | str | The datetime column to diff for computing durations |
| `variable` | str | Controls which grouping logic to apply: `"residence_time"` or `"observation_time"` |
| `gap_hours` | int | Hours threshold above which two consecutive points are considered separate passes (residence only) |

#### Calculation Logic

- **`"observation_time"`**: sorts by `burst_id` then timestamp, then computes the time to the next row *within each burst*. The last row of each burst has no "next"   it gets NaN, which is filled forward from the previous interval.
- **`"residence_time"`**: sorts globally by timestamp, then detects gaps larger than `gap_hours` between consecutive points. Each gap starts a new `pass_id`. Time to next row is computed *within each pass*. The last row of each pass gets NaN, filled forward from the previous interval within that pass.

This ensures intervals never bleed across burst or pass boundaries   a gap of several hours does not inflate the last known interval.

#### Output

Same DataFrame with a new `time_interval` column containing the duration in **seconds** the spacecraft spent at each position.

> ⚠️ If `variable` is `None` or unrecognised, `next_time` is never assigned and a `NameError` is raised. No guard is currently in place.

---

### `add_burst_count(df, coord_colnames, burst_id_colname)`

#### Inputs

| Parameter | Type | Meaning |
|---|---|---|
| `df` | DataFrame | AKR burst DataFrame   each row is a timestamped measurement belonging to a burst |
| `coord_colnames` | tuple[str, str, str] | Column names for the three spatial coordinates to bin by |
| `burst_id_colname` | str | Column identifying which burst each row belongs to |

#### Calculation Logic

For each burst, only the **first row** (earliest timestamp) is kept   this represents the burst onset, i.e. where and when it started. Duplicate rows from the same burst are discarded. The onset positions are binned into the 3D grid, and the number of unique burst IDs landing in each cell is counted.

#### Output

**`grid.burst_count`**   integer count per cell of how many distinct AKR burst events *originated* in that region of space. Answers: *"how often did AKR activity start here?"*

---

### `add_observation_count(df, coord_colnames)`

#### Inputs

| Parameter | Type | Meaning |
|---|---|---|
| `df` | DataFrame | AKR burst DataFrame   every row is a measurement |
| `coord_colnames` | tuple[str, str, str] | Column names for the three spatial coordinates to bin by |

#### Calculation Logic

No filtering or grouping. Every single row in the DataFrame is binned into the grid and counted as +1. This is the most literal count of how many data points exist per cell.

#### Output

**`grid.observation_count`**   integer count per cell of how many AKR measurement rows exist in that region. Answers: *"how densely was this region sampled in the AKR data?"*

---

### `add_residence_count(df, coord_colnames, residence_timestamp_colname, gap_hours=2)`

#### Inputs

| Parameter | Type | Meaning |
|---|---|---|
| `df` | DataFrame | Full satellite trajectory DataFrame   every row is a position in time |
| `coord_colnames` | tuple[str, str, str] | Column names for the three spatial coordinates to bin by |
| `residence_timestamp_colname` | str | The datetime column used to detect gaps between passes |
| `gap_hours` | int | If two consecutive timestamps are more than this many hours apart, it is treated as a new separate orbital pass |

#### Calculation Logic

The trajectory is sorted chronologically. Consecutive timestamps are diffed   wherever the gap exceeds `gap_hours`, a new `pass_id` is assigned. This separates the continuous trajectory into distinct orbital passes. Each row is then binned, and the number of **unique `pass_id` values** per cell is counted.

#### Output

**`grid.residence_count`**   integer count per cell of how many separate times the spacecraft flew through that region. Answers: *"how many distinct visits did the spacecraft make here?"*

---

### `add_observation_time(df, coord_colnames, timestamp_colname)`

#### Inputs

| Parameter | Type | Meaning |
|---|---|---|
| `df` | DataFrame | AKR burst DataFrame |
| `coord_colnames` | tuple[str, str, str] | Column names for the three spatial coordinates to bin by |
| `timestamp_colname` | str | Datetime column used to compute time intervals between consecutive burst measurements |

#### Calculation Logic

Calls `_add_time_intervals` with `variable="observation_time"`. Intervals are computed within each burst group   time from one measurement to the next, staying within burst boundaries. Each row is then binned and the time intervals are **summed** per cell.

#### Output

**`grid.observation_time`**   total seconds of AKR burst activity recorded in each cell. Answers: *"how long was AKR actively observed in this region?"*

---

### `add_residence_time(df, coord_colnames, timestamp_colname, gap_hours=2)`

#### Inputs

| Parameter | Type | Meaning |
|---|---|---|
| `df` | DataFrame | Full satellite trajectory DataFrame |
| `coord_colnames` | tuple[str, str, str] | Column names for the three spatial coordinates to bin by |
| `timestamp_colname` | str | Datetime column used to compute time intervals between consecutive trajectory points |
| `gap_hours` | int | Hours threshold for separating distinct orbital passes before computing intervals |

#### Calculation Logic

Calls `_add_time_intervals` with `variable="residence_time"`. Intervals are computed within each orbital pass   time from one position to the next, never bleeding across pass boundaries. Each row is then binned and the time intervals are **summed** per cell.

#### Output

**`grid.residence_time`**   total seconds the spacecraft was physically present in each cell. Answers: *"how long did the spacecraft actually spend in this region?"*

> ⚠️ If the DataFrame already contains a `time_interval` column, `_add_time_intervals` is skipped entirely and `gap_hours` is silently ignored. Pre-computed intervals bypass all pass-gap logic.

---

### `add_normalised_observation_time(akr_df, satellite_residence_df, coord_colnames, akr_timestamp_colname, residence_timestamp_colname, gap_hours=2)`

#### Inputs

| Parameter | Type | Meaning |
|---|---|---|
| `akr_df` | DataFrame | AKR burst DataFrame   used to populate `observation_time` if not already filled |
| `satellite_residence_df` | DataFrame | Full satellite trajectory DataFrame   used to populate `residence_time` if not already filled |
| `coord_colnames` | tuple[str, str, str] | Column names for the three spatial coordinates to bin by |
| `akr_timestamp_colname` | str | Datetime column in the AKR DataFrame |
| `residence_timestamp_colname` | str | Datetime column in the trajectory DataFrame |
| `gap_hours` | int | Passed to `add_residence_time` if it needs to auto-populate   controls pass separation threshold |

#### Calculation Logic

First checks if `observation_time` and `residence_time` grids are populated (non-zero). If either is empty, it auto-populates it by calling the corresponding method. Then performs a cell-by-cell division:

```
normalised_observation_time = observation_time / residence_time
```

Cells where `residence_time = 0` (spacecraft never visited) are set to 0 using `np.divide(..., where=den != 0)` to avoid division errors. The result is clipped to `[0, 1]` to enforce a physical constraint   AKR cannot be active for longer than the spacecraft was present.

#### Output

**`grid.normalised_observation_time`**   a value between 0 and 1 per cell representing the fraction of spacecraft presence time during which AKR was active. Answers: *"correcting for how long the spacecraft was here, how active was AKR really?"*

This is the **primary scientifically meaningful metric**   without normalisation, cells visited more frequently will appear more active simply due to sampling bias, not true physical activity.

---

### Summary table

| Method | Input DataFrame | Calculation | Grid output | Scientific meaning |
|---|---|---|---|---|
| `add_burst_count` | AKR | Count unique burst onsets per cell | `burst_count` | How many AKR events started here |
| `add_observation_count` | AKR | Count all rows per cell | `observation_count` | How densely AKR data sampled here |
| `add_observation_time` | AKR | Sum time intervals within bursts per cell | `observation_time` | Total AKR active seconds here |
| `add_residence_count` | Trajectory | Count distinct orbital passes per cell | `residence_count` | How many times spacecraft visited here |
| `add_residence_time` | Trajectory | Sum time intervals within passes per cell | `residence_time` | Total seconds spacecraft was here |
| `add_normalised_observation_time` | Both | `observation_time / residence_time` per cell | `normalised_observation_time` | True AKR activity rate, bias-corrected |
