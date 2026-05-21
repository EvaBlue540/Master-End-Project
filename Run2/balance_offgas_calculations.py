
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import timedelta
from scipy.integrate import trapezoid
import argparse
import matplotlib.pyplot as plt

# Set font to CMU Serif (Computer Modern - LaTeX default)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif', 'DejaVu Serif']
# Prevent math text from being italicized
plt.rcParams['mathtext.default'] = 'regular'

# ============================================================
# PATHS & SETTINGS
# ============================================================
# Determine root directory (parent of the Reactor data pipeline scripts folder)
ROOT_DIR = Path(__file__).resolve().parent.parent
EXCEL_OUTPUT_DIR = ROOT_DIR / "Excel output"
PLOTS_DIR = ROOT_DIR / "plots2" / "offgas"

# Ensure output directories exist
EXCEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Look for balance file in Excel output folder first, then fallback to current directory
balance_file = str("path to balance file")  # Replace with actual path to balance file

path5_offgas = "path_to_luke5_offgas.csv"  # Replace with actual path to Luke 5 off-gas data
path6_offgas = "path_to_luke6_offgas.csv"  # Replace with actual path to Luke 6 off-gas data

EXPERIMENTS = [
    ("Luke5", path5_offgas, 212),
    ("Luke6", path6_offgas, 352),
]

start_date = pd.to_datetime("start_date_placeholder")  # Replace with actual start date of experiments
end_date   = pd.to_datetime("end_date_placeholder")    # Replace with actual end date of experiments

daily_blocks = [
    ("09:00", "15:50"),
    ("15:50", "21:00"),
    ("21:00", "03:30"),
    ("03:50", "09:00"),
]

gasflow = 200
gasflow_after = 400
gasflow_switch_cycle = 95
gas_volume = 22.4
N_content = 0.2

gamma_ac = 4.0
gamma_PHA = 4.5
gamma_biomass = 4.2

OUTPUT_DIR = PLOTS_DIR

# Selected cycles for detailed plotting (plots each cycle with its consecutive one)
selected_cycles = [15, 30, 39, 53, 59 ,73, 81,93]


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_csv_safely(path_candidates):
    """Load CSV with multiple encoding and separator attempts."""
    for p in path_candidates:
        if not os.path.exists(p):
            continue
        for enc in ("utf-16", "utf-8", "latin-1"):
            for sep in ('\t', ',', None):
                try:
                    if sep is None:
                        df = pd.read_csv(p, sep=None, encoding=enc, engine='python')
                    else:
                        df = pd.read_csv(p, sep=sep, encoding=enc)
                    if len(df.columns) >= 3:
                        return df
                except Exception:
                    continue
    raise FileNotFoundError("No readable CSV found in provided paths")


def ensure_time_column(df, expected_start=None, expected_end=None):
    """Parse and clean time column with flexible date format.
    
    Args:
        df: DataFrame with time column
        expected_start: Expected date range start (default: global start_date)
        expected_end: Expected date range end (default: global end_date)
    
    Returns:
        Cleaned DataFrame with 'time' column parsed and sorted
    """
    if expected_start is None:
        expected_start = start_date
    if expected_end is None:
        expected_end = end_date
        
    df = df.copy()
    time_col = df.columns[0]
    
    # Try parsing with explicit format first (day-month-year hour:minute)
    try:
        df['time'] = pd.to_datetime(df[time_col], format='%d-%m-%Y %H:%M')
        na_count = df['time'].isna().sum()
        if na_count > len(df) * 0.1:
            # Format didn't work, try alternatives
            raise ValueError("Too many NaT values")
    except (ValueError, TypeError):
        # Try ISO format
        df['time'] = pd.to_datetime(df[time_col], errors='coerce', format='mixed')
        na_count = df['time'].isna().sum()
        if na_count > len(df) * 0.1:  # More than 10% NaT - try dayfirst
            df['time'] = pd.to_datetime(df[time_col], errors='coerce', dayfirst=True)
            na_count = df['time'].isna().sum()
    
    na_count = df['time'].isna().sum()
    
    # Validate date range
    if len(df['time'].dropna()) > 0:
        valid_times = df['time'].dropna()
        range_start = pd.to_datetime(f"{expected_start.date()} 00:00:00")
        range_end = pd.to_datetime(f"{(expected_end + timedelta(days=1)).date()} 23:59:59")
        in_range = ((valid_times >= range_start) & (valid_times <= range_end)).sum()
    
    # Drop rows with invalid times and sort
    dropped = na_count
    df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    
    actual_start = df['time'].min()
    actual_end = df['time'].max()
    
    return df


def compute_gas_rates(df, gasflow_ml_min, gas_volume_l_per_mol, gasflow_switch_time=None, gasflow_after_ml_min=None):
    """Compute CO2 and O2 rates from gas fractions.
    
    Args:
        gasflow_switch_time: datetime when gas flow switches to higher rate (optional)
        gasflow_after_ml_min: gas flow rate after switch (optional)
    """
    col_names = list(df.columns)
    if len(col_names) < 4:
        raise ValueError("Expected at least 4 columns (time, CO2, N2, O2)")
    co2_col = col_names[1]
    n2_col = col_names[2]
    o2_col = col_names[3]
    N2_in = 78.7
    CO2_in = 0.051
    O2_in = 20.3
    if gas_volume_l_per_mol == 0:
        raise ZeroDivisionError("gas_volume cannot be zero")
    
    # Apply time-dependent gas flow if switch parameters provided
    if gasflow_switch_time is not None and gasflow_after_ml_min is not None:
        flow_ml_min = np.where(df['time'] >= gasflow_switch_time, gasflow_after_ml_min, gasflow_ml_min)
    else:
        flow_ml_min = gasflow_ml_min
    
    gas_in = (flow_ml_min / 1000.0) * 60.0 / gas_volume_l_per_mol * 1000.0
    N2_out = pd.to_numeric(df[n2_col], errors='coerce')
    CO2_frac = pd.to_numeric(df[co2_col], errors='coerce') / 100.0
    O2_frac = pd.to_numeric(df[o2_col], errors='coerce') / 100.0
    gas_out = gas_in * (N2_out / N2_in)
    CO2_rate = CO2_frac * gas_out - (CO2_in / 100.0) * gas_in
    O2_rate = O2_frac * gas_out - (O2_in / 100.0) * gas_in
    return CO2_rate.astype(float), O2_rate.astype(float)


def generate_integration_intervals(start_date, end_date, daily_blocks):
    """Generate time intervals for daily SBR cycles."""
    intervals = []
    prev_end = None
    num_days = (end_date - start_date).days + 1
    for d in range(num_days):
        day = start_date + timedelta(days=d)
        for start_t, end_t in daily_blocks:
            start_dt = pd.to_datetime(f"{day.date()} {start_t}")
            end_dt = pd.to_datetime(f"{day.date()} {end_t}")
            if prev_end and start_dt < prev_end:
                start_dt += timedelta(days=1)
                end_dt += timedelta(days=1)
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)
            # Special case: March 8's last block (03:50-09:00) ends at 08:50 on March 9
            if day.date() == pd.to_datetime("2026-03-08").date() and start_t == "03:50" and end_t == "09:00":
                end_dt = pd.to_datetime("2026-03-09 08:50")
            intervals.append((start_dt, end_dt))
            prev_end = end_dt
    return intervals


def integrate_intervals(df, CO2_rate, O2_rate, intervals):
    """Integrate gas rates over time intervals.
    
    Returns all intervals including those with insufficient data.
    Marks incomplete intervals with NaN values and quality flag.
    """
    results = []
    for start_int, end_int in intervals:
        mask = (df['time'] >= start_int) & (df['time'] <= end_int)
        times = df['time'][mask]
        
        if len(times) < 2:
            # No data or insufficient data points - record with NaN
            quality = "NO_DATA" if len(times) == 0 else "INSUFFICIENT"
            results.append({
                "start": start_int,
                "end": end_int,
                "duration_h": (end_int - start_int).total_seconds() / 3600.0,
                "CO2_mmol": np.nan,
                "O2_mmol": np.nan,
                "data_points": len(times),
                "data_quality": quality,
            })
        else:
            # Check if gas rate data is actually valid (not all NaN)
            CO2_valid = CO2_rate[mask].notna().sum()
            O2_valid = O2_rate[mask].notna().sum()
            min_data_points = 5
            
            if CO2_valid == 0 or O2_valid == 0:
                # No valid gas measurements
                results.append({
                    "start": start_int,
                    "end": end_int,
                    "duration_h": (end_int - start_int).total_seconds() / 3600.0,
                    "CO2_mmol": np.nan,
                    "O2_mmol": np.nan,
                    "data_points": len(times),
                    "data_quality": "NO_DATA",
                })
            elif min(CO2_valid, O2_valid) < min_data_points:
                # Not enough valid data points
                results.append({
                    "start": start_int,
                    "end": end_int,
                    "duration_h": (end_int - start_int).total_seconds() / 3600.0,
                    "CO2_mmol": np.nan,
                    "O2_mmol": np.nan,
                    "data_points": min(CO2_valid, O2_valid),
                    "data_quality": "INSUFFICIENT",
                })
            else:
                # Check temporal coverage
                valid_times = times[~((CO2_rate[mask].isna()) | (O2_rate[mask].isna()))]
                time_span = (valid_times.max() - valid_times.min()).total_seconds() / 3600.0
                expected_span = (end_int - start_int).total_seconds() / 3600.0
                coverage_ratio = time_span / expected_span if expected_span > 0 else 0
                
                if coverage_ratio < 0.7:  # Less than 70% of the interval has data
                    results.append({
                        "start": start_int,
                        "end": end_int,
                        "duration_h": (end_int - start_int).total_seconds() / 3600.0,
                        "CO2_mmol": np.nan,
                        "O2_mmol": np.nan,
                        "data_points": min(CO2_valid, O2_valid),
                        "data_quality": "INSUFFICIENT",
                    })
                else:
                    # Get valid data points (remove NaN for integration)
                    valid_mask = ~((CO2_rate[mask].isna()) | (O2_rate[mask].isna()))
                    valid_times = times[valid_mask]
                    valid_CO2 = CO2_rate[mask][valid_mask]
                    valid_O2 = O2_rate[mask][valid_mask]
                    
                    if len(valid_times) >= 2:
                        elapsed = (valid_times - start_int).dt.total_seconds() / 3600.0
                        CO2_int = trapezoid(valid_CO2, elapsed)
                        O2_int = trapezoid(valid_O2, elapsed)
                    else:
                        # Not enough valid points to integrate
                        CO2_int = np.nan
                        O2_int = np.nan
                    
                    results.append({
                        "start": start_int,
                        "end": end_int,
                        "duration_h": (end_int - start_int).total_seconds() / 3600.0,
                        "CO2_mmol": CO2_int,
                        "O2_mmol": O2_int,
                        "data_points": len(valid_times),
                        "data_quality": "COMPLETE",
                    })
    return pd.DataFrame(results)


def detect_feast_end_index(experiment_name, time_array, O2_C_array, gasflow_ml_min=200, reference_gasflow=200):
    
    if len(time_array) == 0:
        return 0

    window = 3
    O2_smooth = pd.Series(O2_C_array).rolling(window=window, center=True, min_periods=1).mean().values

    min_feast_time = 2.0
    max_feast_time = 6.5

    if experiment_name == "Luke5":
        gradient_threshold_baseline = -12
        min_sustained_points = 2
    else:
        gradient_threshold_baseline = -12
        min_sustained_points = 2
    
    # Scale gradient threshold by gas flow rate
    # Higher gas flow → higher O2 gradient → need higher (more negative) threshold
    # Example: 400 mL/min has 2x O2 values, so gradient is 2x, need 2x threshold magnitude
    flowrate_factor = gasflow_ml_min / reference_gasflow
    gradient_threshold = gradient_threshold_baseline * flowrate_factor

    valid_indices = np.where(time_array >= min_feast_time)[0]
    if len(valid_indices) == 0:
        feast_end_idx = len(time_array) - 1
    else:
        first_valid = valid_indices[0]
        dO2dt = np.gradient(O2_smooth, time_array)

        feast_end_idx = None
        for idx in range(first_valid, len(dO2dt) - min_sustained_points + 1):
            sustained_drop = np.all(dO2dt[idx:idx + min_sustained_points] < gradient_threshold)
            if sustained_drop:
                feast_end_idx = idx
                break

        if feast_end_idx is None:
            peak_o2 = np.max(O2_smooth[first_valid:])
            peak_idx = first_valid + np.argmax(O2_smooth[first_valid:])
            after_peak = O2_smooth[peak_idx:]
            o2_threshold = 0.65 * peak_o2
            drop_indices = np.where(after_peak < o2_threshold)[0]

            if len(drop_indices) > 0:
                feast_end_idx = peak_idx + drop_indices[0]
            else:
                feast_end_idx = first_valid + int(0.8 * (len(O2_smooth) - first_valid))

        max_idx = np.searchsorted(time_array, max_feast_time)
        if max_idx > 0:
            feast_end_idx = min(feast_end_idx, max_idx)

    feast_end_idx = min(feast_end_idx, len(time_array) - 1)

    feast_length_h = time_array[feast_end_idx] if feast_end_idx < len(time_array) else time_array[-1]
    if feast_length_h < min_feast_time:
        # Find first index where time >= min_feast_time
        min_indices = np.where(time_array >= min_feast_time)[0]
        if len(min_indices) > 0:
            feast_end_idx = min_indices[0]
    elif feast_length_h > max_feast_time:
        # Find last index where time <= max_feast_time
        max_indices = np.where(time_array <= max_feast_time)[0]
        if len(max_indices) > 0:
            feast_end_idx = max_indices[-1]

    return min(feast_end_idx, len(time_array) - 1)

def process_offgas_with_balance_loads(experiment_name, csv_path, df_balance_input,
                                     start_date, end_date, daily_blocks,
                                     gasflow, gas_volume, N_content,
                                     gamma_ac, gamma_PHA, gamma_biomass,
                                     gasflow_switch_cycle=None, gasflow_after_ml_min=None):
    
    print(f"Processing {experiment_name}...")
    
    # Load and prepare off-gas data
    df = load_csv_safely([csv_path])
    df = ensure_time_column(df)
    
    # Generate integration intervals (need this before computing gas rates for switch time)
    integration_intervals = generate_integration_intervals(start_date, end_date, daily_blocks)
    
    # Calculate gas flow switch time if cycle specified
    gasflow_switch_time = None
    if gasflow_switch_cycle is not None:
        # Cycle 95 N-phase = interval index (95-1)*2 + 1 = 189
        switch_interval_idx = (gasflow_switch_cycle - 1) * 2 + 1
        if switch_interval_idx < len(integration_intervals):
            gasflow_switch_time = integration_intervals[switch_interval_idx][0]
            print(f"  Gas flow switches to {gasflow_after_ml_min} mL/min at cycle {gasflow_switch_cycle} N-phase")
    
    # Compute gas rates
    CO2_rate, O2_rate = compute_gas_rates(df, gasflow, gas_volume,
                                         gasflow_switch_time=gasflow_switch_time,
                                         gasflow_after_ml_min=gasflow_after_ml_min)
    CO2_rate = pd.Series(CO2_rate, index=df.index)
    O2_rate = pd.Series(O2_rate, index=df.index)
    
    # ===== APPLY DST CORRECTION AND RENAME TO ADJUSTED =====
    # For 2026-03-30 (cycle 107) only:
    # - Remove jump point at 12:15
    # - Subtract 1 hour from times after 12:00
    
    cycle_date_str = "2026-03-30"
    cycle_number_DST = 107
    
    # Define time window for this cycle's C-phase
    cycle_start = pd.to_datetime(f"{cycle_date_str} 09:00")
    cycle_end = pd.to_datetime(f"{cycle_date_str} 15:50")
    
    # Remove jump point at 12:15 (12:10-12:20 window)
    jump_start = pd.to_datetime(f"{cycle_date_str} 12:10:00")
    jump_end = pd.to_datetime(f"{cycle_date_str} 12:20:00")
    jump_mask = (df['time'] > jump_start) & (df['time'] < jump_end)
    
    # Remove jump indices from rate series
    jump_indices = df[jump_mask].index
    if len(jump_indices) > 0:
        CO2_rate = CO2_rate.drop(jump_indices)
        O2_rate = O2_rate.drop(jump_indices)
        df = df.drop(jump_indices).copy()
        print(f"[DST] Cycle {cycle_number_DST}: Removed {len(jump_indices)} jump point(s) at 12:15")
    
    # Subtract 1 hour from ALL times after the jump point (globally, not just this cycle)
    mask_after_jump = df['time'] >= jump_end
    num_shifted = mask_after_jump.sum()
    if num_shifted > 0:
        df.loc[mask_after_jump, 'time'] = df.loc[mask_after_jump, 'time'] - pd.Timedelta(hours=1)
    
    # Rename to adjusted versions for clarity
    CO2_rate_adjusted = CO2_rate
    O2_rate_adjusted = O2_rate
    
    # Integrate over intervals (using standard intervals, data times now corrected)
    df_integrated = integrate_intervals(df, CO2_rate_adjusted, O2_rate_adjusted, integration_intervals)

    pha_results = []
    skip_reasons = {}  # Track why cycles were skipped
    O2_uptake = -O2_rate_adjusted
    exchange_ratio = 0.5
    PHA_leftover = 0

    # Main calculation loop - IDENTICAL to new off-gas.py but with balance-derived loads
    for i in range(0, len(df_integrated) - 1, 2):
        C_phase = df_integrated.iloc[i]
        N_phase = df_integrated.iloc[i + 1] if (i + 1) < len(df_integrated) else None
        cycle_number = i // 2 + 1

        # Determine cycle data quality (COMPLETE only if both phases are COMPLETE)
        c_phase_quality = C_phase.get('data_quality', 'COMPLETE')
        n_phase_quality = N_phase['data_quality'] if N_phase is not None else 'MISSING'
        cycle_data_quality = 'COMPLETE' if (c_phase_quality == 'COMPLETE' and n_phase_quality == 'COMPLETE') else c_phase_quality

        # Get balance data for this cycle
        if cycle_number > len(df_balance_input):
            continue
        
        balance_row = df_balance_input.iloc[cycle_number - 1]
        C_acetate = balance_row['Carbon_added_mCmol']  # Use from balance
        N_load = balance_row['Nitrogen_added_mNmol']    # Use from balance
        C_fed_grams = balance_row['C_medium_grams']
        N_fed_grams = balance_row['N_medium_grams']

        # SKIP if balance data is missing - avoid NaN propagation
        if pd.isna(C_acetate) or pd.isna(N_load):
            pha_results.append({
                "Cycle": cycle_number,
                "Date": balance_row.get('Date', pd.NaT),
                "C_fed_grams": np.nan,
                "N_fed_grams": np.nan,
                "C_acetate_mCmol": np.nan,
                "N_load_mNmol": np.nan,
                "CO2_C_mmol": np.nan,
                "O2_C_mmol": np.nan,
                "CO2_N_mmol": np.nan,
                "O2_N_mmol": np.nan,
                "PHA_new_mCmol": np.nan,
                "PHA_yield_Cmol_Cmol": np.nan,
                "PHA_removed_mCmol": np.nan,
                "PHA_available_N_mCmol": np.nan,
                "PHA_consumed_mCmol": np.nan,
                "Biomass_max_possible": np.nan,
                "Biomass_N_phase_growth_mCmol": np.nan,
                "Biomass_C_phase_growth_mCmol": np.nan,
                "Biomass_total_grown_mCmol": np.nan,
                "Feast_length_h": np.nan,
                "Feast_end_time": pd.NaT,
                "O2_int_feast_mmol": np.nan,
                "O2_int_famine_mmol": np.nan,
                "CO2_int_feast_mmol": np.nan,
                "CO2_int_famine_mmol": np.nan,
                "PHA_burned_famine_mCmol": np.nan,
                "Limitation": "SKIPPED",
                "PHA_leftover": np.nan,
                "N_unused_mNmol": np.nan,
                "Acetate_for_C_phase_growth_mCmol": np.nan,
                "C_acetate_for_PHA_mCmol": np.nan,
                "PHA_content_end_C_phase_%": np.nan,
                "PHA_content_end_N_phase_%": np.nan,
                "C_out_mCmol": np.nan,
                "O2_for_growth_mmol": np.nan,
                "O2_for_PHA_mmol": np.nan,
                "C_balance_error_%": np.nan,
            })
            continue

        start_C = C_phase['start']
        end_C = C_phase['end']
        
        # Special case adjustments for data gaps
        is_special_case = False
        if experiment_name == "Luke6" and start_C == pd.to_datetime("2026-02-23 09:00"):
            end_C = pd.to_datetime("2026-02-23 15:35")  # Extend to capture next point
            is_special_case = True
        elif experiment_name == "Luke6" and start_C == pd.to_datetime("2026-03-03 09:00"):
            start_C = pd.to_datetime("2026-03-03 09:10")  # Skip missing data period
            is_special_case = True
        
        mask_C = (df['time'] >= start_C) & (df['time'] <= end_C)
        time_segment = df['time'][mask_C]
        
        # Create time array relative to cycle start
        time_array = (time_segment - start_C).dt.total_seconds().values / 3600.0
        
        O2_C_array = O2_uptake[mask_C].values
        CO2_C_array = CO2_rate_adjusted[mask_C].values
        
        # Check if we have enough valid data points for processing
        num_valid_o2 = np.sum(~np.isnan(O2_C_array))
        num_valid_co2 = np.sum(~np.isnan(CO2_C_array))
        has_valid_data = (num_valid_o2 >= 5 and num_valid_co2 >= 5)

        # Only process if data quality is COMPLETE or if it's a special case with valid data
        if (cycle_data_quality == 'COMPLETE' or (is_special_case and has_valid_data)) and has_valid_data:
            # Re-integrate for special cases with adjusted times
            if is_special_case:
                # For special cases, use the re-extracted and filtered values
                O2_C = trapezoid(O2_C_array, time_array)
                CO2_C = trapezoid(CO2_C_array, time_array)
            else:
                # Use the pre-integrated values from df_integrated
                # (For cycles 107-108, data was already corrected at source, so pre-integrated values are correct)
                CO2_C = C_phase['CO2_mmol']
                O2_C = -C_phase['O2_mmol']
            
            # Mark special cases with valid data as COMPLETE for processing
            if is_special_case:
                cycle_data_quality = 'COMPLETE'
            
            CO2_N = N_phase['CO2_mmol'] if N_phase is not None else np.nan
            O2_N = -N_phase['O2_mmol'] if N_phase is not None else np.nan

            # Determine current gas flow for this cycle
            current_gasflow = gasflow
            if gasflow_switch_cycle is not None and cycle_number >= gasflow_switch_cycle:
                current_gasflow = gasflow_after
            
            # Detect feast end using O2 gradient (for metadata: feast_length_h, feast_end_time)
            feast_end_idx = detect_feast_end_index(experiment_name, time_array, O2_C_array,
                                                   gasflow_ml_min=current_gasflow, reference_gasflow=gasflow)
            
            # Calculate O2-based feast/famine
            O2_feast = O2_C_array[: feast_end_idx + 1]
            time_feast_o2 = time_array[: feast_end_idx + 1]
            feast_length_h = time_feast_o2[-1] if len(time_feast_o2) > 0 else 0
            feast_end_time = time_segment.iloc[feast_end_idx] if feast_end_idx < len(time_segment) else pd.NaT
            O2_int_feast = trapezoid(O2_feast, time_feast_o2) if len(time_feast_o2) > 1 else 0

            # O2 famine part - include boundary point for continuity
            if feast_end_idx < len(time_array) - 1:
                O2_famine = O2_C_array[feast_end_idx :]
                time_famine_o2 = time_array[feast_end_idx :]
                O2_int_famine = trapezoid(O2_famine, time_famine_o2) if len(time_famine_o2) > 1 else 0
            else:
                O2_int_famine = 0

            # Calculate CO2-based feast/famine using same feast end point
            CO2_feast = CO2_C_array[: feast_end_idx + 1]
            time_feast_co2 = time_array[: feast_end_idx + 1]
            CO2_int_feast = trapezoid(CO2_feast, time_feast_co2) if len(time_feast_co2) > 1 else 0

            # CO2 famine part - include boundary point for continuity
            if feast_end_idx < len(time_array) - 1:
                CO2_famine = CO2_C_array[feast_end_idx :]
                time_famine_co2 = time_array[feast_end_idx :]
                CO2_int_famine = trapezoid(CO2_famine, time_famine_co2) if len(time_famine_co2) > 1 else 0
            else:
                CO2_int_famine = 0

            e_resp_famine = 4 * O2_int_famine
            PHA_burned_famine = e_resp_famine / gamma_PHA if gamma_PHA != 0 else np.nan

            E_in = C_acetate * gamma_ac

            Biomass_tot = N_load / N_content
            E_biomass = gamma_biomass * Biomass_tot

            E_fixed = O2_N*4 + O2_C*4 + E_biomass

            PHA_removed = (E_in - E_fixed) / gamma_PHA

            PHA_removed = max(0.0, PHA_removed)  # physical constraint

            PHA_available_N = PHA_removed

            PHA_needed_for_N = (E_biomass+4*O2_N) / gamma_PHA
            
            if PHA_needed_for_N > PHA_available_N:
                PHA_consumed = PHA_available_N
                Biomass_N = (PHA_available_N * gamma_PHA - O2_N*4) / gamma_biomass
                limitation = "PHA-limited"
            else:        
                PHA_consumed = PHA_needed_for_N
                Biomass_N = Biomass_tot
                limitation = "N-limited"

            PHA_leftover = PHA_available_N - PHA_consumed

            PHA_end_C = PHA_removed/(1 - exchange_ratio)

            PHA_made = PHA_end_C-PHA_leftover
            
            N_unused = max(0.0, N_load - (Biomass_N*N_content))

            Biomass_C_phase_growth = N_unused / N_content if N_content > 0 else 0.0

            acetate_for_growth = Biomass_C_phase_growth / 0.5 
            acetate_for_PHA = C_acetate - acetate_for_growth

            e_in_C = gamma_ac * acetate_for_PHA
            e_resp_C = 4 * O2_C
            E_biomass_C = gamma_biomass * Biomass_C_phase_growth
            e_PHA = e_in_C - e_resp_C - E_biomass_C
            #PHA_made = e_PHA / gamma_PHA 

            # PHA yield
            pha_yield = PHA_made / C_acetate if C_acetate > 0 else np.nan

            
            # ===== TOTAL BIOMASS ACCOUNTING =====
            # Total biomass grown across both phases
            # N-phase: Biomass_actual (determined by PHA/N limitation)
            # C-phase: Biomass_C_phase_growth (from leftover N, only if PHA-limited)
            Biomass_total_grown = Biomass_N + Biomass_C_phase_growth

            # Calculate PHA content percentages
            # End of C-phase: PHA_removed / (PHA_removed + Biomass_total_grown)
            # Only considers what is in the effluent (PHA removed + biomass grown)
            total_end_C = PHA_removed + Biomass_total_grown
            if total_end_C > 0:
                PHA_content_C_phase = (PHA_removed / total_end_C) * 100.0
            else:
                PHA_content_C_phase = 0.0
            
            Absolute_Biomass_C = 2*Biomass_total_grown #total biomass at end of C = double the biomass grown (accounting for washout)
            Absolute_Biomass_N = Biomass_total_grown + Biomass_N #biomass at end of N. Biomass start N + biomass grown in N.
        
        
            # Factor of 2 accounts for steady-state washout
            total_end_N = PHA_leftover + Biomass_total_grown + Biomass_N  # Adjusted to reflect biomass in reactor at end of N-phase
            if total_end_N > 0:
                PHA_content_N_phase = (PHA_leftover / total_end_N) * 100.0
            else:
                PHA_content_N_phase = 0.0
            
        
            # Carbon IN (from acetate fed)
            C_in = C_acetate  # mCmol
            
            # Carbon OUT:
            # 1. Biomass grown:
            #    - N-phase: Biomass_actual (mCmol)
            #    - C-phase: Biomass_C_phase_growth (mCmol) from leftover N (only if PHA-limited)
            C_out_Biomass = Biomass_total_grown  # mCmol
            
            # 2. PHA removed (measured in mCmol, which is C-equivalent)
            C_out_PHA = PHA_removed  # mCmol
            
            # 3. CO2 total produced (C-phase + N-phase, already in mmol but = mCmol for CO2)
            C_out_CO2 = CO2_C + CO2_N  # mCmol
            
            # Total carbon out
            C_out_total = C_out_Biomass + C_out_PHA + C_out_CO2
            
            # Carbon balance error
            if C_in > 0:
                C_balance_error = ((C_out_total - C_in) / C_in) * 100  # % error
            else:
                C_balance_error = 0.0

            PHA_feast = (gamma_ac * acetate_for_PHA - O2_int_feast * 4 - E_biomass_C) / gamma_PHA

            Y_PHA_feast = PHA_feast / acetate_for_PHA if acetate_for_PHA > 0 else np.nan

            PHA_total_feast = PHA_feast + PHA_leftover

            biomass_feast = Absolute_Biomass_N+Biomass_C_phase_growth

            PHA_content_feast = PHA_total_feast / (PHA_total_feast + biomass_feast) * 100 if (PHA_total_feast + biomass_feast) > 0 else 0.0

            pha_results.append({
                "Cycle": cycle_number,
                "Date": balance_row['Date'],
                "Limitation": limitation,
                "C_fed_grams": C_fed_grams,
                "N_fed_grams": N_fed_grams,
                "C_acetate_mCmol": C_acetate,
                "C_acetate_for_PHA_mCmol": acetate_for_PHA,
                "Acetate_for_C_phase_growth_mCmol": acetate_for_growth,
                "N_load_mNmol": N_load,
                "CO2_C_mmol": CO2_C,
                "O2_C_mmol": O2_C,
                "CO2_N_mmol": CO2_N,
                "O2_N_mmol": O2_N,
                "PHA_new_mCmol": PHA_made,
                "PHA_yield_Cmol_Cmol": pha_yield,
                "Feast_length_h": feast_length_h,
                "Feast_end_time": feast_end_time,
                "PHA_feast_mCmol": PHA_feast,
                "PHA_yield_feast_Cmol_Cmol": Y_PHA_feast,
                "pha_content_feast_%": PHA_content_feast,
                "O2_int_feast_mmol": O2_int_feast,
                "O2_int_famine_mmol": O2_int_famine,
                "CO2_int_feast_mmol": CO2_int_feast,
                "CO2_int_famine_mmol": CO2_int_famine,
                "PHA_burned_famine_mCmol": PHA_burned_famine,
                "PHA_total_mCmol": PHA_end_C,
                "Biomass_C_phase_growth_mCmol": Biomass_C_phase_growth,
                "Biomass_total_grown_mCmol": Biomass_total_grown,
                "Biomass_C_mNmol": Absolute_Biomass_C,
                "PHA_removed_mCmol": PHA_removed,
                "PHA_available_N_mCmol": PHA_available_N,
                "PHA_consumed_mCmol": PHA_consumed,
                "Biomass_max_possible": Biomass_tot,
                "Biomass_N_phase_growth_mCmol": Biomass_N,
                "Biomass_N_mNmol": Absolute_Biomass_N,
                "PHA_leftover": PHA_leftover,
                "N_unused_mNmol": N_unused,
                "PHA_content_end_C_phase_%": PHA_content_C_phase,
                "PHA_content_end_N_phase_%": PHA_content_N_phase,
                "C_balance_error_%": C_balance_error,
            })
        else:
            # Insufficient data for this cycle
            # Log skipping details
            skip_reason = ""
            if not has_valid_data:
                skip_reason = f"Insufficient data points (O2: {num_valid_o2}, CO2: {num_valid_co2})"
            else:
                skip_reason = f"Data quality issue - C: {c_phase_quality}, N: {n_phase_quality}"
            
            skip_reasons[cycle_number] = skip_reason
            
            pha_results.append({
                "Cycle": cycle_number,
                "Date": balance_row.get('Date', pd.NaT),
                "Limitation": "SKIPPED",
                "C_fed_grams": C_fed_grams,
                "N_fed_grams": N_fed_grams,
                "C_acetate_mCmol": C_acetate,
                "C_acetate_for_PHA_mCmol": acetate_for_PHA,
                "Acetate_for_C_phase_growth_mCmol": acetate_for_growth,
                "N_load_mNmol": N_load,
                "CO2_C_mmol": np.nan,
                "O2_C_mmol": np.nan,
                "CO2_N_mmol": np.nan,
                "O2_N_mmol": np.nan,
                "PHA_new_mCmol": np.nan,
                "PHA_yield_Cmol_Cmol": np.nan,
                "Feast_length_h": np.nan,
                "Feast_end_time": pd.NaT,
                "Feast_CO2": pd.NaT,
                "PHA_feast_mCmol": np.nan,
                "PHA_yield_feast_Cmol_Cmol": np.nan,
                "O2_int_feast_mmol": np.nan,
                "O2_int_famine_mmol": np.nan,
                "CO2_int_feast_mmol": np.nan,
                "CO2_int_famine_mmol": np.nan,
                "PHA_burned_famine_mCmol": np.nan,
                "PHA_total_mCmol": np.nan,
                "Biomass_C_phase_growth_mCmol": np.nan,
                "Biomass_total_grown_mCmol": np.nan,
                "Biomass_C_mNmol": np.nan,  
                "PHA_removed_mCmol": np.nan,
                "PHA_available_N_mCmol": np.nan,
                "PHA_consumed_mCmol": np.nan,
                "Biomass_max_possible": np.nan,
                "Biomass_N_phase_growth_mCmol": np.nan,
                "Biomass_N_mNmol": np.nan,
                "PHA_leftover": np.nan,
                "N_unused_mNmol": np.nan,
                "PHA_content_end_C_phase_%": np.nan,
                "PHA_content_end_N_phase_%": np.nan,
                "C_balance_error_%": np.nan,
            })

    df_results = pd.DataFrame(pha_results)
    
    # Preserve datetime columns before rounding (don't round timestamps)
    datetime_cols = ['Date', 'Feast_end_time']
    datetime_data = {col: df_results[col] for col in datetime_cols if col in df_results.columns}
    
    # Round numeric columns only
    numeric_cols = [col for col in df_results.columns if col not in datetime_cols]
    df_results[numeric_cols] = df_results[numeric_cols].round(2)
    
    # Restore datetime columns
    for col in datetime_cols:
        if col in datetime_data:
            df_results[col] = datetime_data[col]

    # Diagnostic summary of integration results
    total_cycles = len(df_results)
    processed_cycles = (df_results['CO2_C_mmol'].notna()).sum()
    no_data_cycles = total_cycles - processed_cycles

    try:
        plot_offgas_phases(experiment_name, df, O2_uptake, CO2_rate_adjusted, df_integrated, df_results)
    except Exception as e:
        pass

    return df_results


def create_reactor_overview(df_results):
    """
    Create a simplified reactor overview dataframe from detailed results.
    
    Includes key cycling parameters and yields for quick assessment.
    """
    # Convert Date column to date only (remove time) if it contains timestamps
    dates = pd.to_datetime(df_results['Date']).dt.date
    
    # Calculate biomass/PHA yield for N-phase (using only N-phase growth)
    biomass_pha_yield_n = (df_results['Biomass_N_phase_growth_mCmol'] / df_results['PHA_consumed_mCmol']).round(3)
    
    overview = pd.DataFrame({
        'Cycle': df_results['Cycle'],
        'Date': dates,
        'Limitation': df_results['Limitation'],
        'N_load (mNmol)': df_results['N_load_mNmol'],
        'X_N_phase (mCmol)': df_results['Biomass_N_phase_growth_mCmol'],
        'X_C_phase (mCmol)': df_results['Biomass_C_phase_growth_mCmol'],
        'PHA_consumed (mCmol)': df_results['PHA_consumed_mCmol'],
        'Y_X_PHA (mCmol/mCmol)': biomass_pha_yield_n,
        'C_load (mCmol)': df_results['C_acetate_mCmol'],
        'PHA_made (mCmol)': df_results['PHA_new_mCmol'],
        'PHA_feast (mCmol)': df_results.get('PHA_feast_mCmol', np.nan),
        'Feast_phase (h)': df_results['Feast_length_h'],
        'Feast end time (O2)': df_results.get('Feast_end_time', pd.NaT),
        'Y_PHA_C (Cmol/Cmol)': df_results['PHA_yield_Cmol_Cmol'],
        'Y_PHA_feast (Cmol/Cmol)': df_results.get('PHA_yield_feast_Cmol_Cmol', np.nan),
        'PHA_%_EC': df_results['PHA_content_end_C_phase_%'],
        'PHA_%_EN': df_results['PHA_content_end_N_phase_%'],
    })
    
    return overview


def plot_offgas_phases(experiment_name, df, O2_uptake, CO2_rate, df_integrated, df_results):
    """Create detailed feast and full C-phase plots for selected cycles and their consecutive cycles."""

    def _plot_one(plot_type, cycle_indices):
        fig, ax = plt.subplots(figsize=(13, 7))

        all_times, all_o2, all_co2 = [], [], []
        for phase_idx in cycle_indices:
            row = df_integrated.loc[phase_idx]
            mask = (df['time'] >= row["start"]) & (df['time'] <= row["end"])
            all_times.extend(df['time'][mask])
            all_o2.extend(O2_uptake[mask])
            all_co2.extend(CO2_rate[mask])

        ax.plot(all_times, all_co2, color='#E63946', linewidth=2.5, label=r'$CO_2$ Production')
        ax.plot(all_times, all_o2, color='#2E86AB', linewidth=2.5, label=r'$O_2$ Consumption')
        
        # Add Feast End line with label (will appear after O2 Consumption in legend)
        feast_end_added = False
        for phase_idx in cycle_indices:
            if phase_idx % 2 == 0:  # C-phase
                cycle_nr = phase_idx // 2 + 1
                c_row = df_results.loc[df_results['Cycle'] == cycle_nr].iloc[0]
                feast_end = c_row.get('Feast_end_time', pd.NaT)
                if pd.notna(feast_end) and not feast_end_added:
                    feast_end = pd.Timestamp(feast_end)
                    ax.axvline(feast_end, linestyle='--', color='#457B9D', linewidth=1.8, alpha=0.8,
                               label=r'Feast End ($O_2$)')
                    feast_end_added = True
        
        # Add background shading underneath curves for phases: Feast (darkest), C-phase (lighter), N-phase (lightest)
        phase_labels_added = {'N-Phase': False, 'C-Phase': False, 'Feast-Phase': False}
        
        # First pass: Add feast phase shading
        for phase_idx in cycle_indices:
            if phase_idx % 2 == 0:  # C-phase
                row = df_integrated.loc[phase_idx]
                phase_start = pd.to_datetime(row["start"])
                cycle_nr = phase_idx // 2 + 1
                feast_row = df_results.loc[df_results['Cycle'] == cycle_nr]
                if not feast_row.empty and pd.notna(feast_row['Feast_end_time'].values[0]):
                    feast_end = pd.to_datetime(feast_row['Feast_end_time'].values[0])
                    feast_mask = (df['time'] >= phase_start) & (df['time'] <= feast_end)
                    feast_times = df.loc[feast_mask, 'time'].values
                    feast_o2 = O2_uptake[feast_mask].values
                    ax.fill_between(feast_times, 0, feast_o2, alpha=0.18, color='#0093A3',
                                   label=r'Cumulative $O_2$ - Feast-Phase' if not phase_labels_added['Feast-Phase'] else '')
                    phase_labels_added['Feast-Phase'] = True
        
        # Second pass: Add C-phase shading
        for phase_idx in cycle_indices:
            if phase_idx % 2 == 0:  # C-phase
                row = df_integrated.loc[phase_idx]
                phase_start = pd.to_datetime(row["start"])
                phase_end = pd.to_datetime(row["end"])
                
                # For C-phase, extend shading to N-start (6.5 + 20/60 hours after C-start)
                n_start = phase_start + pd.Timedelta(hours=6.5 + 20/60)
                shading_end = n_start
                
                # Get phase data up to N-start
                phase_mask = (df['time'] >= phase_start) & (df['time'] <= shading_end)
                phase_times = df.loc[phase_mask, 'time'].values
                phase_o2 = O2_uptake[phase_mask].values
                
                ax.fill_between(phase_times, 0, phase_o2, alpha=0.14, color='#00B2C4',
                               label=r'Cumulative $O_2$ - C-Phase' if not phase_labels_added['C-Phase'] else '')
                phase_labels_added['C-Phase'] = True
        
        # Third pass: Add N-phase shading
        for phase_idx in cycle_indices:
            if phase_idx % 2 == 1:  # N-phase
                row = df_integrated.loc[phase_idx]
                phase_start = pd.to_datetime(row["start"])
                phase_end = pd.to_datetime(row["end"])
                
                phase_mask = (df['time'] >= phase_start) & (df['time'] <= phase_end)
                phase_times = df.loc[phase_mask, 'time'].values
                phase_o2 = O2_uptake[phase_mask].values
                
                ax.fill_between(phase_times, 0, phase_o2, alpha=0.10, color='#33CCD6', 
                               label=r'Cumulative $O_2$ - N-Phase' if not phase_labels_added['N-Phase'] else '')
                phase_labels_added['N-Phase'] = True

        c_phase_indices = [idx for idx in cycle_indices if idx % 2 == 0]
        for cycle_count, phase_idx in enumerate(c_phase_indices):
            row = df_integrated.loc[phase_idx]
            cycle_nr = phase_idx // 2 + 1
            c_phase_start = row["start"]
            c_phase_end = row["end"]
            c_row = df_results.loc[df_results['Cycle'] == cycle_nr].iloc[0]

            if plot_type == 'feast':
                feast_end = c_row.get('Feast_end_time', pd.NaT)
                # Ensure feast_end is a valid Timestamp
                if pd.notna(feast_end):
                    feast_end = pd.Timestamp(feast_end)
                    # Use indices directly to ensure alignment
                    feast_mask = (df['time'] >= c_phase_start) & (df['time'] <= feast_end)
                    feast_idx_positions = np.where(feast_mask.values)[0]
                    
                    if len(feast_idx_positions) > 0:
                        # Extract feast phase data using consistent indexing
                        # Convert positional indices to actual dataframe index labels to handle non-sequential indices
                        feast_indices = df.index[feast_idx_positions]
                        feast_times = df.loc[feast_indices, 'time'].values
                        feast_o2 = O2_uptake.loc[feast_indices].values
                        ax.fill_between(feast_times, feast_o2, alpha=0.25, color='#A8DADC')
                
                c_phase_start_hour = pd.to_datetime(c_phase_start).hour
                c_phase_end_date = pd.to_datetime(c_phase_end).date()
                label_text = f"{float(c_row.get('Feast_length_h', np.nan)):.1f}h\nY$_{{PHA/Ac}}$={float(c_row.get('PHA_yield_feast_Cmol_Cmol', np.nan)):.2f}"
            else:
                c_phase_mask = (df['time'] >= c_phase_start) & (df['time'] <= c_phase_end)
                c_phase_times = df['time'][c_phase_mask]
                if len(c_phase_times) > 0:
                    ax.fill_between(c_phase_times, O2_uptake[c_phase_mask], alpha=0.25, color='#A8DADC')
                c_phase_start_hour = pd.to_datetime(c_phase_start).hour
                c_phase_end_date = pd.to_datetime(c_phase_end).date()
                label_text = f"{(c_phase_end - c_phase_start).total_seconds() / 3600:.1f}h\nY$_{{PHA/Ac}}$={float(c_row.get('PHA_yield_Cmol_Cmol', np.nan)):.2f}"

            base_date = pd.to_datetime(c_phase_start).date()
            if cycle_count == 1:
                base_date += timedelta(days=1)
            annotation_time = pd.to_datetime(f"{base_date} {'14:00' if cycle_count == 0 else '02:00'}")
            y_lim = ax.get_ylim()
            

        y_lim = ax.get_ylim()
        for idx in cycle_indices:
            row = df_integrated.loc[idx]
            start = pd.to_datetime(row["start"])
            if idx % 2 == 0:
                ax.axvline(start, color='black', linestyle='-', alpha=0.6, linewidth=2.2)
            else:
                ax.axvline(start, color='#696969', linestyle='--', alpha=0.5, linewidth=1.5)
                

        ax.set_ylabel(r'$O_2$ Consumption / $CO_2$ Production (mmol/h)', fontsize=12, fontweight='bold')
        title = f"Luke {experiment_name[-1]} - Length of Feast Phase" if plot_type == 'feast' else f"Luke {experiment_name[-1]} - Total C-Phase PHA Yield"
        ax.grid(True, linestyle='--', alpha=0.25, color='gray', which='major', axis='y')
        ax.set_axisbelow(True)

        important_times, important_labels = [], []
        n_start_time = None  # Track N-start for x-axis limit
        for idx in cycle_indices:
            row = df_integrated.loc[idx]
            
            if idx % 2 == 0:  # C-phase
                c_phase_start = pd.to_datetime(row["start"])
                cycle_nr = idx // 2 + 1
                
                # C-start label at t=0
                important_times.append(c_phase_start)
                important_labels.append(f"C-start\nt=0")
                
                # C-feast label (if available in feast plot)
                if plot_type == 'feast':
                    feast_row = df_results.loc[df_results['Cycle'] == cycle_nr]
                    if not feast_row.empty:
                        if pd.notna(feast_row['Feast_end_time'].values[0]):
                            feast_end = pd.to_datetime(feast_row['Feast_end_time'].values[0])
                            important_times.append(feast_end)
                            feast_hours = (feast_end - c_phase_start).total_seconds() / 3600
                            important_labels.append(f"C-feast\nt={feast_hours:.1f}")
                
                # N-start at 6.5 + 20/60 hours after C-start
                n_start = c_phase_start + pd.Timedelta(hours=6.5+20/60)
                important_times.append(n_start)
                n_start_hours = 6.5 + 20/60
                important_labels.append(f"N-start\nt={n_start_hours:.2f}")
                n_start_time = n_start  # Store for x-axis limit
                
                # N-end at 12 hours after C-start
                n_end = c_phase_start + pd.Timedelta(hours=12)
                important_times.append(n_end)
                important_labels.append(f"N-end\nt=12")

        if important_times:
            ax.set_xticks(important_times, minor=False)
            ax.set_xticklabels(important_labels, rotation=0, ha='center', fontsize=8)
            # Set x-axis to start at the first phase and end after the last phase
            x_min = important_times[0]
            x_max = important_times[-1]
            ax.set_xlim(x_min, x_max)
        # Use a coarser interval for minor ticks to avoid exceeding Locator.MAXTICKS
        ax.xaxis.set_minor_locator(plt.matplotlib.dates.HourLocator(interval=12))
        ax.tick_params(axis='x', which='minor', length=3, width=0.5)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', fontsize=10, framealpha=0.95)

        return fig, ax
    
    # Plot selected cycles (C-phase and N-phase of the same cycle)
    for cycle_num in selected_cycles:
        # Cycle number N corresponds to C-phase index 2*(N-1) and N-phase index 2*(N-1)+1
        c_phase_idx = 2 * (cycle_num - 1)
        n_phase_idx = c_phase_idx + 1
        
        # Only plot if both C and N phases exist in df_integrated
        if c_phase_idx in df_integrated.index and n_phase_idx in df_integrated.index:
            cycle_indices = [c_phase_idx, n_phase_idx]
            
            # Plot feast phase
            fig, ax = _plot_one('feast', cycle_indices)
            filename = OUTPUT_DIR / f"{experiment_name}_cycle{cycle_num}_feastphase.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot full C-phase
            fig, ax = _plot_one('full', cycle_indices)
            filename = OUTPUT_DIR / f"{experiment_name}_cycle{cycle_num}_totalcphase.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()


def generate_combined_feast_plot(all_results):
    """Generate combined feast phase length comparison plot across all experiments.
    
    Creates a scatter plot comparing feast phase lengths from Luke5 and Luke6 reactors.
    """
    try:
        fig, ax = plt.subplots(figsize=(16, 8))

        colors = {'Luke5': '#2E86AB', 'Luke6': '#A23B72'}
        
        for exp_name, df_results in all_results.items():
            cycles = df_results['Cycle'].values
            feast_lengths = df_results['Feast_length_h'].values
            
            # Plot with X markers only (no lines)
            ax.plot(cycles, feast_lengths, 'x', linewidth=0, markersize=10, 
                   label=exp_name, color=colors.get(exp_name, '#000000'), markeredgewidth=2)

        # Formatting
        ax.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feast Phase Length (h)', fontsize=12, fontweight='bold')
        ax.set_title('Feast Phase Length Comparison (Luke5 vs Luke6)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best', framealpha=0.95)

        # Set x-axis spacing
        max_cycle = max([df_results['Cycle'].max() for df_results in all_results.values()])
        x_ticks = list(range(0, int(max_cycle) + 1, 10)) + [int(max_cycle)]
        x_ticks = sorted(set(x_ticks))
        ax.set_xticks(x_ticks)
        ax.set_xlim(-2, max_cycle + 5)
        
        # Get y-axis limits
        max_feast = max([df_results['Feast_length_h'].max() for df_results in all_results.values()])
        ax.set_ylim(0, max_feast + 1)

        # Save
        output_path = OUTPUT_DIR / "feast_phase_length_combined.png"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        pass


# ============================================================
# MAIN EXECUTION
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run off-gas calculations using balance-derived loads."
    )
    parser.add_argument(
        "--luke5-offgas-csv",
        dest="luke5_offgas_csv",
        default=path5_offgas,
        help="Path to Luke5 off-gas CSV file",
    )
    parser.add_argument(
        "--luke6-offgas-csv",
        dest="luke6_offgas_csv",
        default=path6_offgas,
        help="Path to Luke6 off-gas CSV file",
    )
    parser.add_argument(
        "--balance-excel",
        dest="balance_excel",
        default=balance_file,
        help="Path to processed balance Excel file",
    )
    parser.add_argument(
        "--combined-excel",
        dest="combined_excel",
        default=str(EXCEL_OUTPUT_DIR / "pipeline_combined_results.xlsx"),
        help="Path to the final combined Excel output",
    )
    parser.add_argument(
        "--start-date",
        dest="start_date",
        default="2026-02-05",
        help="Start date for analysis (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        default="2026-03-30",
        help="End date for analysis (YYYY-MM-DD)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global path5_offgas, path6_offgas, balance_file, EXPERIMENTS, start_date, end_date
   
    balance_file = args.balance_excel
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    EXPERIMENTS = [
        ("Luke5", path5_offgas, 212),
        ("Luke6", path6_offgas, 352),
    ]

    balance_data = {}
    for exp_name in ["Luke5", "Luke6"]:
        balance_data[exp_name] = pd.read_excel(balance_file, sheet_name=exp_name)
    
    # Process each experiment
    all_results = {}
    all_overviews = {}
    total_cycles_offgas = 0
    for exp_name, offgas_path, C_conc in EXPERIMENTS:
        df_balance = balance_data[exp_name]
        
        df_results = process_offgas_with_balance_loads(
            exp_name, offgas_path, df_balance,
            start_date, end_date, daily_blocks,
            gasflow, gas_volume, N_content,
            gamma_ac, gamma_PHA, gamma_biomass,
            gasflow_switch_cycle=gasflow_switch_cycle,
            gasflow_after_ml_min=gasflow_after
        )
        
        total_cycles_offgas += len(df_results)
        all_results[exp_name] = df_results
        
        # Create reactor overview
        df_overview = create_reactor_overview(df_results)
        all_overviews[exp_name] = df_overview
        
        summarize_offgas_results(exp_name, df_results, df_overview)
    
    # Create output directories if they don't exist
    EXCEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Export
    output_file = args.combined_excel
    temp_file = f"{output_file}.tmp.xlsx"
    try:
        balance_sheets = pd.read_excel(balance_file, sheet_name=None)
        with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
            for sheet_name, sheet_df in balance_sheets.items():
                writer_sheet_name = f"balance_{sheet_name}"
                sheet_df.to_excel(writer, sheet_name=writer_sheet_name[:31], index=False)

            for exp_name, df_results in all_results.items():
                writer_sheet_name = f"offgas_{exp_name}"
                df_results.to_excel(writer, sheet_name=writer_sheet_name[:31], index=False)

            for exp_name, df_overview in all_overviews.items():
                writer_sheet_name = f"overview_{exp_name}"
                df_overview.to_excel(writer, sheet_name=writer_sheet_name[:31], index=False)

        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(temp_file, output_file)
    except Exception as e:
        print(f"Warning: Could not save Excel file: {e}")
        print(f"Data was computed successfully. Results available in memory.")
    
    # Also write balance_offgas_calculations.xlsx with just the offgas data (for backward compatibility)
    balance_offgas_file = str(EXCEL_OUTPUT_DIR / "balance_offgas_calculations.xlsx")
    try:
        with pd.ExcelWriter(balance_offgas_file, engine='openpyxl') as writer:
            for exp_name, df_results in all_results.items():
                df_results.to_excel(writer, sheet_name=exp_name, index=False)
    
    except Exception as e:
        print(f"Warning: Could not save legacy Excel file: {e}")
    
    # Export feast/famine data to separate file
    feast_famine_file = str(EXCEL_OUTPUT_DIR / "feast_famine_analysis.xlsx")
    try:
        feast_famine_columns = [
            "Cycle", "Date", "Feast_end_time", "Feast_length_h",
            "O2_int_feast_mmol", "CO2_int_feast_mmol", "PHA_feast_mCmol",
            "O2_int_famine_mmol", "CO2_int_famine_mmol", "PHA_burned_famine_mCmol",
            "PHA_new_mCmol", "Limitation"
        ]
        with pd.ExcelWriter(feast_famine_file, engine='openpyxl') as writer:
            for exp_name, df_results in all_results.items():
                # Select only the feast/famine columns that exist
                available_cols = [col for col in feast_famine_columns if col in df_results.columns]
                df_feast_famine = df_results[available_cols]
                writer_sheet_name = exp_name
                df_feast_famine.to_excel(writer, sheet_name=writer_sheet_name, index=False)
        print(f"[offgas] Feast/famine analysis exported to: {feast_famine_file}")
    except Exception as e:
        print(f"Warning: Could not save feast/famine file: {e}")
    
    # Export selected cycles only to a separate file (two sheets: one per experiment)
    selected_cycles_file = str(EXCEL_OUTPUT_DIR / "selected_cycles_results.xlsx")
    try:
        with pd.ExcelWriter(selected_cycles_file, engine='openpyxl') as writer:
            for exp_name, df_results in all_results.items():
                # Filter to only selected cycles
                df_selected = df_results[df_results['Cycle'].isin(selected_cycles)].reset_index(drop=True)
                if len(df_selected) > 0:
                    df_selected.to_excel(writer, sheet_name=exp_name, index=False)
                    print(f"[offgas] {exp_name}: {len(df_selected)} selected cycles exported")
                else:
                    print(f"[offgas] {exp_name}: No selected cycles found in results")
        print(f"[offgas] Selected cycles exported to: {selected_cycles_file}")
    except Exception as e:
        print(f"Warning: Could not save selected cycles file: {e}")
    
    # Generate combined feast plot comparing all experiments
    generate_combined_feast_plot(all_results)
    
    print(f"[offgas] Total cycles processed: {total_cycles_offgas}")
    print("[offgas] Plots generated: Feast phase analysis, combined feast phase length")
    print(f"[offgas] Output stored: {output_file}")
    print("[offgas] Complete.")


if __name__ == "__main__":
    main()


