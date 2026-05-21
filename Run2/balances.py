import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import argparse
from pathlib import Path

# Determine root directory and output paths
ROOT_DIR = Path(__file__).resolve().parent.parent
EXCEL_OUTPUT_DIR = ROOT_DIR / "Excel output"
EXCEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

start_date = pd.to_datetime("placeholder_start_date")  # Replace with actual start date of experiments
end_date   = pd.to_datetime("placeholder_end_date")    # Replace with actual end date of experiments

def load_csv_safely(path_candidates):
    """Load CSV with multiple encoding attempts."""
    for p in path_candidates:
        if not os.path.exists(p):
            continue
        # Try UTF-16 first, then fallback to other encodings
        for enc in ("utf-16", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(p, sep='\t', encoding=enc, engine='python')
                return df
            except Exception:
                # Tab separator failed, try auto-detect with comma
                try:
                    df = pd.read_csv(p, sep=None, encoding=enc, engine='python')
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
    """
    if expected_start is None:
        expected_start = start_date
    if expected_end is None:
        expected_end = end_date
        
    df = df.copy()
    time_col = df.columns[0]
    
    # Try parsing with ISO format first
    df['time'] = pd.to_datetime(df[time_col], errors='coerce', format='mixed')
    
    # If many NaT values OR dates seem wrong (outside expected range), try with dayfirst=True
    na_count = df['time'].isna().sum()
    if na_count > len(df) * 0.1:  # More than 10% NaT
        df['time'] = pd.to_datetime(df[time_col], errors='coerce', dayfirst=True)
    elif len(df['time'].dropna()) > 0:
        # Check if dates are mostly within expected range (add 1 day buffer for end of day measurements)
        valid_times = df['time'].dropna()
        range_start = pd.to_datetime(f"{expected_start.date()} 00:00:00")
        range_end = pd.to_datetime(f"{(expected_end + timedelta(days=1)).date()} 23:59:59")
        in_range = ((valid_times >= range_start) & (valid_times <= range_end)).sum()
        if in_range < len(valid_times) * 0.8:  # Less than 80% in expected range, try dayfirst
            df['time'] = pd.to_datetime(df[time_col], errors='coerce', dayfirst=True)
    
    df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    return df


def extract_balance_columns(df, start_date, end_date, acid_col_idx=2, base_col_idx=4, medium_col_idx=14, medium_col_name=None):
    
    col_names = list(df.columns)
    if len(col_names) < 13:
        raise ValueError("Expected at least 13 columns")
    
    # Determine medium scale column
    if medium_col_name and medium_col_name in col_names:
        medium_col = medium_col_name
    else:
        medium_col = col_names[medium_col_idx]
    
    # Extract columns
    df_extracted = df[['time']].copy()
    df_extracted['acid_scale'] = pd.to_numeric(df[col_names[acid_col_idx]], errors='coerce')  # 2M HCl
    df_extracted['base_scale'] = pd.to_numeric(df[col_names[base_col_idx]], errors='coerce')  # 1M NaOH
    df_extracted['medium_scale'] = pd.to_numeric(df[medium_col], errors='coerce')
    
    # Filter by date range: from start_date 09:00 to end_date +1 day 10:00
    # (Extended to next day 10:00 to capture complete nighttime cycle data, including N-phase ending at 09:50 DST)
    start_datetime = pd.to_datetime(f"{start_date.date()} 09:00:00")
    end_datetime = pd.to_datetime(f"{(end_date + timedelta(days=1)).date()} 10:00:00")
    
    df_extracted = df_extracted[(df_extracted['time'] >= start_datetime) & 
                                (df_extracted['time'] <= end_datetime)]
    
    return df_extracted.reset_index(drop=True)


def detect_refills_in_window(df_balance, time_start, time_end, refill_threshold=500):
    """Detect refill events by finding sudden weight increases."""
    window = df_balance[(df_balance['time'] >= time_start) & 
                        (df_balance['time'] <= time_end)].copy()
    
    if len(window) < 2:
        return False, None, None
    
    window = window.sort_values('time').reset_index(drop=True)
    for i in range(len(window) - 1):
        if window.iloc[i + 1]['medium_scale'] > window.iloc[i]['medium_scale'] + refill_threshold:
            return True, window.iloc[i + 1]['medium_scale'], window.iloc[i + 1]['time']
    
    return False, None, None


def get_scale_value_at_time(df_balance, target_time, look_ahead_range=3600):
    """Get scale reading at target time, with fallback to nearest future reading.
    
    Args:
        look_ahead_range: seconds to look ahead for next reading (default 3600s=60min for sparse data)
    """
    reading = df_balance[df_balance['time'] == target_time]['medium_scale'].values
    if len(reading) > 0:
        return reading[0]
    
    # Look for reading at or after target time, within look_ahead_range
    future = df_balance[(df_balance['time'] >= target_time) & 
                        (df_balance['time'] <= target_time + timedelta(seconds=look_ahead_range))]
    
    result = future.iloc[0]['medium_scale'] if len(future) > 0 else None
    return result


def _calculate_consumption_with_refill(df_balance, time_start, time_end, val_start, val_end):
    """Calculate consumption accounting for refills during the period."""
    if val_start is None or val_end is None:
        return np.nan
    
    refill_detected, weight_after_refill, refill_time = detect_refills_in_window(
        df_balance, time_start, time_end, refill_threshold=600
    )
    
    if refill_detected and weight_after_refill is not None and refill_time is not None:
        # Consumption with refill: (start - weight_before_refill) + (weight_after_refill - end)
        weight_before_refill = df_balance[
            (df_balance['time'] <= refill_time) & 
            (df_balance['time'] >= time_start)
        ]['medium_scale'].min()
        return abs(val_start - weight_before_refill) + abs(weight_after_refill - val_end)
    else:
        # Simple consumption: start - end
        return abs(val_start - val_end)


def process_medium_scale_cycles(df_balance, start_date, end_date, c_concentration=212):
    """Process medium scale consumption for 12-hour cycles (Daytime: 09:00-21:00, Nighttime: 21:00-08:50).
    
    Includes ALL cycles (64 total from Feb 5-Mar 8) with NaN values when data is unavailable.
    """
    results = []
    cycle_number = 1
    
    current_date = start_date
    dst_threshold = pd.to_datetime("2026-03-29 00:00:00")
    
    while current_date.date() <= end_date.date():
        # Account for daylight saving time starting March 29, 2026 (shift times by 1 hour from that date onwards)
        
        # ===== DAYTIME CYCLE =====
        # Daytime: C-phase 09:00-11:30, N-phase 15:45-21:00
        c_start_day = pd.to_datetime(f"{current_date.date()} 09:00:00")
        c_end_day = pd.to_datetime(f"{current_date.date()} 11:30:00")
        n_start_day = pd.to_datetime(f"{current_date.date()} 15:45:00")
        n_end_day = pd.to_datetime(f"{current_date.date()} 20:30:00")
        
        # Apply DST shift only to phases that start at or after the DST threshold
        c_start_day = c_start_day + (timedelta(hours=1) if c_start_day >= dst_threshold else timedelta(0))
        c_end_day = c_end_day + (timedelta(hours=1) if c_end_day >= dst_threshold else timedelta(0))
        n_start_day = n_start_day + (timedelta(hours=1) if n_start_day >= dst_threshold else timedelta(0))
        n_end_day = n_end_day + (timedelta(hours=1) if n_end_day >= dst_threshold else timedelta(0))
        
        val_c_start_day = df_balance[df_balance['time'] == c_start_day]['medium_scale'].values
        val_c_start_day = val_c_start_day[0] if len(val_c_start_day) > 0 else get_scale_value_at_time(df_balance, c_start_day, look_ahead_range=600)
        val_c_end_day = get_scale_value_at_time(df_balance, c_end_day)
        val_n_start_day = get_scale_value_at_time(df_balance, n_start_day)
        val_n_end_day = get_scale_value_at_time(df_balance, n_end_day)
        
        # Initialize with NaN values
        cycle_data_day = {
            'Cycle': cycle_number,
            'Date': current_date.date(),
            'C_medium_grams': np.nan,
            'C_pump_rate_L_h': np.nan,
            'Carbon_added_mCmol': np.nan,
            'N_medium_grams': np.nan,
            'N_pump_rate_L_h': np.nan,
            'Nitrogen_added_mNmol': np.nan,
            'Data_Quality': 'INSUFFICIENT',
        }
        
        if val_c_start_day is not None and val_c_end_day is not None:
            # C consumption with refill detection
            c_grams_day = _calculate_consumption_with_refill(df_balance, c_start_day, c_end_day, 
                                                              val_c_start_day, val_c_end_day)
            if c_grams_day > 700:
                refill_detected, weight_after_refill, _ = detect_refills_in_window(
                    df_balance, c_start_day, c_end_day, refill_threshold=600
                )
                if refill_detected and weight_after_refill is not None:
                    c_grams_day = abs(weight_after_refill - val_c_start_day)
            
            c_liters_day = c_grams_day / 1000.0
            c_pump_rate_day = c_liters_day / 2.0
            c_carbon_day = c_liters_day * c_concentration
            
            cycle_data_day['C_medium_grams'] = c_grams_day
            cycle_data_day['C_pump_rate_L_h'] = c_pump_rate_day
            cycle_data_day['Carbon_added_mCmol'] = c_carbon_day
        elif current_date.date() == pd.to_datetime("2026-03-03").date() and c_start_day.hour == 9:
            # Special case: March 3 daytime C-phase - hardcode 500g consumption
            c_grams_day = 500.0
            c_liters_day = c_grams_day / 1000.0
            c_pump_rate_day = c_liters_day / 2.0
            c_carbon_day = c_liters_day * c_concentration
            
            cycle_data_day['C_medium_grams'] = c_grams_day
            cycle_data_day['C_pump_rate_L_h'] = c_pump_rate_day
            cycle_data_day['Carbon_added_mCmol'] = c_carbon_day
        
        if val_n_start_day is not None and val_n_end_day is not None:
            # N consumption
            n_grams_day = _calculate_consumption_with_refill(df_balance, n_start_day, n_end_day,
                                                          val_n_start_day, val_n_end_day)
            n_liters_day = n_grams_day / 1000.0
            n_pump_rate_day = n_liters_day / 2.0
            n_nitrogen_day = n_liters_day * 8.8
            
            cycle_data_day['N_medium_grams'] = n_grams_day
            cycle_data_day['N_pump_rate_L_h'] = n_pump_rate_day
            cycle_data_day['Nitrogen_added_mNmol'] = n_nitrogen_day
        
        # Mark as COMPLETE only if both C and N data are available
        if (not np.isnan(cycle_data_day['C_medium_grams']) and 
            not np.isnan(cycle_data_day['N_medium_grams'])):
            cycle_data_day['Data_Quality'] = 'COMPLETE'
        elif (not np.isnan(cycle_data_day['C_medium_grams']) or 
              not np.isnan(cycle_data_day['N_medium_grams'])):
            cycle_data_day['Data_Quality'] = 'PARTIAL'
        
        results.append(cycle_data_day)
        cycle_number += 1
        
        # ===== NIGHTTIME CYCLE =====
        # Nighttime: C-phase 21:00-23:30, N-phase 03:45-08:50 (next day)
        c_start_night = pd.to_datetime(f"{current_date.date()} 21:00:00")
        c_end_night = pd.to_datetime(f"{current_date.date()} 23:30:00")
        n_start_night = pd.to_datetime(f"{(current_date + timedelta(days=1)).date()} 03:45:00")
        n_end_night = pd.to_datetime(f"{(current_date + timedelta(days=1)).date()} 08:50:00")
        
        # Apply DST shift only to phases that start at or after the DST threshold
        c_start_night = c_start_night + (timedelta(hours=1) if c_start_night >= dst_threshold else timedelta(0))
        c_end_night = c_end_night + (timedelta(hours=1) if c_end_night >= dst_threshold else timedelta(0))
        n_start_night = n_start_night + (timedelta(hours=1) if n_start_night >= dst_threshold else timedelta(0))
        n_end_night = n_end_night + (timedelta(hours=1) if n_end_night >= dst_threshold else timedelta(0))
        
        val_c_start_night = get_scale_value_at_time(df_balance, c_start_night)
        val_c_end_night = get_scale_value_at_time(df_balance, c_end_night)
        val_n_start_night = get_scale_value_at_time(df_balance, n_start_night)
        val_n_end_night = get_scale_value_at_time(df_balance, n_end_night)
        
        # Initialize with NaN values
        cycle_data_night = {
            'Cycle': cycle_number,
            'Date': current_date.date(),
            'C_medium_grams': np.nan,
            'C_pump_rate_L_h': np.nan,
            'Carbon_added_mCmol': np.nan,
            'N_medium_grams': np.nan,
            'N_pump_rate_L_h': np.nan,
            'Nitrogen_added_mNmol': np.nan,
            'Data_Quality': 'INSUFFICIENT',
        }
        
        if val_c_start_night is not None and val_c_end_night is not None:
            # C consumption
            c_grams_night = _calculate_consumption_with_refill(df_balance, c_start_night, c_end_night,
                                                                val_c_start_night, val_c_end_night)
            if c_grams_night > 700:
                refill_detected, weight_after_refill, _ = detect_refills_in_window(
                    df_balance, c_start_night, c_end_night, refill_threshold=600
                )
                if refill_detected and weight_after_refill is not None:
                    c_grams_night = abs(weight_after_refill - val_c_start_night)
            
            c_liters_night = c_grams_night / 1000.0
            c_pump_rate_night = c_liters_night / 2.0
            c_carbon_night = c_liters_night * c_concentration
            
            cycle_data_night['C_medium_grams'] = c_grams_night
            cycle_data_night['C_pump_rate_L_h'] = c_pump_rate_night
            cycle_data_night['Carbon_added_mCmol'] = c_carbon_night
        
        if val_n_start_night is not None and val_n_end_night is not None:
            # N consumption
            n_grams_night = _calculate_consumption_with_refill(df_balance, n_start_night, n_end_night,
                                                                val_n_start_night, val_n_end_night)
            n_liters_night = n_grams_night / 1000.0
            n_pump_rate_night = n_liters_night / 2.0
            n_nitrogen_night = n_liters_night * 8.8
            
            cycle_data_night['N_medium_grams'] = n_grams_night
            cycle_data_night['N_pump_rate_L_h'] = n_pump_rate_night
            cycle_data_night['Nitrogen_added_mNmol'] = n_nitrogen_night
        
        # Mark as COMPLETE only if both C and N data are available
        if (not np.isnan(cycle_data_night['C_medium_grams']) and 
            not np.isnan(cycle_data_night['N_medium_grams'])):
            cycle_data_night['Data_Quality'] = 'COMPLETE'
        elif (not np.isnan(cycle_data_night['C_medium_grams']) or 
              not np.isnan(cycle_data_night['N_medium_grams'])):
            cycle_data_night['Data_Quality'] = 'PARTIAL'
        
        results.append(cycle_data_night)
        cycle_number += 1
        
        current_date += timedelta(days=1)
    
    df_results = pd.DataFrame(results)
    
    # Round numeric columns, preserve Data_Quality column
    numeric_cols = df_results.select_dtypes(include=['number']).columns
    df_results[numeric_cols] = df_results[numeric_cols].round(3)
    
    # Manual override for cycle 45 (outlier)
    if 45 in df_results['Cycle'].values:
        df_results.loc[df_results['Cycle'] == 45, 'N_medium_grams'] = 500
        df_results.loc[df_results['Cycle'] == 45, 'N_pump_rate_L_h'] = 500 / 1000.0 / 2.0
        df_results.loc[df_results['Cycle'] == 45, 'Nitrogen_added_mNmol'] = (500 / 1000.0) * 8.8
    
    # Manual override for cycle 97
    if 97 in df_results['Cycle'].values:
        df_results.loc[df_results['Cycle'] == 97, 'N_medium_grams'] = 500
        df_results.loc[df_results['Cycle'] == 97, 'N_pump_rate_L_h'] = 500 / 1000.0 / 2.0
        df_results.loc[df_results['Cycle'] == 97, 'Nitrogen_added_mNmol'] = (500 / 1000.0) * 8.8
    
    # Trim trailing cycles with no data (all numeric columns are NaN)
    # Find the last cycle with at least some data
    numeric_cols = df_results.select_dtypes(include=['number']).columns
    has_data = df_results[numeric_cols].notna().any(axis=1)
    if has_data.any():
        last_valid_idx = has_data[has_data].index[-1]
        df_results = df_results.loc[:last_valid_idx].reset_index(drop=True)
    
    return df_results



def process_acid_base_cycles(df_balance, start_date, end_date, acid_conc_M=2.0, base_conc_M=1.0, exclude_base_from_cycle=None):
    """Extract acid and base consumption for C-phase and N-phase per cycle."""
    results = []
    cycle_number = 1
    current_date = start_date
    
    while current_date.date() <= end_date.date():
        # Check if we should exclude base for this cycle onwards
        exclude_base = exclude_base_from_cycle is not None and cycle_number >= exclude_base_from_cycle
        
        # Account for daylight saving time starting March 29, 2026 (shift times by 1 hour from that date onwards)
        time_shift = timedelta(hours=1) if current_date.date() >= pd.to_datetime("2026-03-29").date() else timedelta(hours=0)
        
        # ===== DAYTIME CYCLE =====
        # C-phase: 09:00 to 15:30
        time_c_start = pd.to_datetime(f"{current_date.date()} 09:00:00") + time_shift
        time_c_end = pd.to_datetime(f"{current_date.date()} 15:30:00") + time_shift
        # N-phase: 15:45 to 20:30
        time_n_start = pd.to_datetime(f"{current_date.date()} 15:45:00") + time_shift
        time_n_end = pd.to_datetime(f"{current_date.date()} 20:30:00") + time_shift
        
        # Get all scale values at phase boundaries
        val_acid_c_start = df_balance[df_balance['time'] == time_c_start]['acid_scale'].values
        val_acid_c_end = df_balance[df_balance['time'] == time_c_end]['acid_scale'].values
        val_base_c_start = df_balance[df_balance['time'] == time_c_start]['base_scale'].values
        val_base_c_end = df_balance[df_balance['time'] == time_c_end]['base_scale'].values
        
        val_acid_n_start = df_balance[df_balance['time'] == time_n_start]['acid_scale'].values
        val_acid_n_end = df_balance[df_balance['time'] == time_n_end]['acid_scale'].values
        val_base_n_start = df_balance[df_balance['time'] == time_n_start]['base_scale'].values
        val_base_n_end = df_balance[df_balance['time'] == time_n_end]['base_scale'].values
        
        # C-phase consumption (both acid and base)
        if len(val_acid_c_start) > 0 and len(val_acid_c_end) > 0:
            acid_c = max(0, val_acid_c_start[0] - val_acid_c_end[0])
            acid_h_c = acid_c / 1000.0 * acid_conc_M * 1000.0
        else:
            acid_c = np.nan
            acid_h_c = np.nan
        
        if exclude_base or len(val_base_c_start) == 0 or len(val_base_c_end) == 0:
            base_c = np.nan
            base_oh_c = np.nan
        else:
            base_c = max(0, val_base_c_start[0] - val_base_c_end[0])
            base_oh_c = base_c / 1000.0 * base_conc_M * 1000.0
        
        # N-phase consumption (both acid and base)
        if len(val_acid_n_start) > 0 and len(val_acid_n_end) > 0:
            acid_n = max(0, val_acid_n_start[0] - val_acid_n_end[0])
            acid_h_n = acid_n / 1000.0 * acid_conc_M * 1000.0
        else:
            acid_n = np.nan
            acid_h_n = np.nan
        
        if exclude_base or len(val_base_n_start) == 0 or len(val_base_n_end) == 0:
            base_n = np.nan
            base_oh_n = np.nan
        else:
            base_n = max(0, val_base_n_start[0] - val_base_n_end[0])
            base_oh_n = base_n / 1000.0 * base_conc_M * 1000.0
        
        results.append({
            'Cycle': cycle_number,
            'Date': current_date.date(),
            'C_Acid_mL': acid_c,
            'C_Acid_H+_mmol': acid_h_c,
            'C_Base_mL': base_c,
            'C_Base_OH-_mmol': base_oh_c,
            'N_Acid_mL': acid_n,
            'N_Acid_H+_mmol': acid_h_n,
            'N_Base_mL': base_n,
            'N_Base_OH-_mmol': base_oh_n,
        })
        cycle_number += 1
        
        # ===== NIGHTTIME CYCLE =====
        # C-phase: 21:00 to 03:30 (next day)
        time_nc_start = pd.to_datetime(f"{current_date.date()} 21:00:00") + time_shift
        time_nc_end = pd.to_datetime(f"{(current_date + timedelta(days=1)).date()} 03:30:00") + time_shift
        # N-phase: 03:45 to 09:00 (next day)
        time_nn_start = pd.to_datetime(f"{(current_date + timedelta(days=1)).date()} 03:45:00") + time_shift
        time_nn_end = pd.to_datetime(f"{(current_date + timedelta(days=1)).date()} 09:00:00") + time_shift
        
        # Get all scale values at phase boundaries
        val_acid_nc_start = df_balance[df_balance['time'] == time_nc_start]['acid_scale'].values
        val_acid_nc_end = df_balance[df_balance['time'] == time_nc_end]['acid_scale'].values
        val_base_nc_start = df_balance[df_balance['time'] == time_nc_start]['base_scale'].values
        val_base_nc_end = df_balance[df_balance['time'] == time_nc_end]['base_scale'].values
        
        val_acid_nn_start = df_balance[df_balance['time'] == time_nn_start]['acid_scale'].values
        val_acid_nn_end = df_balance[df_balance['time'] == time_nn_end]['acid_scale'].values
        val_base_nn_start = df_balance[df_balance['time'] == time_nn_start]['base_scale'].values
        val_base_nn_end = df_balance[df_balance['time'] == time_nn_end]['base_scale'].values
        
        # C-phase consumption (both acid and base)
        if len(val_acid_nc_start) > 0 and len(val_acid_nc_end) > 0:
            acid_nc = max(0, val_acid_nc_start[0] - val_acid_nc_end[0])
            acid_h_nc = acid_nc / 1000.0 * acid_conc_M * 1000.0
        else:
            acid_nc = np.nan
            acid_h_nc = np.nan
        
        if exclude_base or len(val_base_nc_start) == 0 or len(val_base_nc_end) == 0:
            base_nc = np.nan
            base_oh_nc = np.nan
        else:
            base_nc = max(0, val_base_nc_start[0] - val_base_nc_end[0])
            base_oh_nc = base_nc / 1000.0 * base_conc_M * 1000.0
        
        # N-phase consumption (both acid and base)
        if len(val_acid_nn_start) > 0 and len(val_acid_nn_end) > 0:
            acid_nn = max(0, val_acid_nn_start[0] - val_acid_nn_end[0])
            acid_h_nn = acid_nn / 1000.0 * acid_conc_M * 1000.0
        else:
            acid_nn = np.nan
            acid_h_nn = np.nan
        
        if exclude_base or len(val_base_nn_start) == 0 or len(val_base_nn_end) == 0:
            base_nn = np.nan
            base_oh_nn = np.nan
        else:
            base_nn = max(0, val_base_nn_start[0] - val_base_nn_end[0])
            base_oh_nn = base_nn / 1000.0 * base_conc_M * 1000.0
        
        results.append({
            'Cycle': cycle_number,
            'Date': current_date.date(),
            'C_Acid_mL': acid_nc,
            'C_Acid_H+_mmol': acid_h_nc,
            'C_Base_mL': base_nc,
            'C_Base_OH-_mmol': base_oh_nc,
            'N_Acid_mL': acid_nn,
            'N_Acid_H+_mmol': acid_h_nn,
            'N_Base_mL': base_nn,
            'N_Base_OH-_mmol': base_oh_nn,
        })
        cycle_number += 1
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(results).round(2)


def summarize_balance_results(exp_name, df_cycles, df_acid_base):
    complete_count = int((df_cycles['Data_Quality'] == 'COMPLETE').sum())
    partial_count = int((df_cycles['Data_Quality'] == 'PARTIAL').sum())
    insufficient_count = int((df_cycles['Data_Quality'] == 'INSUFFICIENT').sum())

    print(f"[{exp_name}] Medium-scale cycles: {len(df_cycles)} total | complete={complete_count}, partial={partial_count}, insufficient={insufficient_count}")
    print(
        f"[{exp_name}] Medium-scale means: C={df_cycles['C_medium_grams'].mean():.2f} g, "
        f"N={df_cycles['N_medium_grams'].mean():.2f} g, carbon={df_cycles['Carbon_added_mCmol'].mean():.2f} mCmol, "
        f"nitrogen={df_cycles['Nitrogen_added_mNmol'].mean():.2f} mNmol"
    )

    acid_mean = df_acid_base['C_Acid_H+_mmol'].mean()
    base_mean = df_acid_base['C_Base_OH-_mmol'].mean()
    print(
        f"[{exp_name}] Acid/base summary: C-phase acid={acid_mean:.2f} mmol, C-phase base={base_mean:.2f} mmol, "
        f"rows={len(df_acid_base)}"
    )
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process balance CSV files and export cycle summaries."
    )
    parser.add_argument(
        "--luke5-balance-csv",
        dest="luke5_balance_csv",
        default="path_to_luke5_balance_csv.csv",
        help="Path to Luke5 balance CSV file",
    )
    parser.add_argument(
        "--luke6-balance-csv",
        dest="luke6_balance_csv",
        default="path_to_luke6_balance_csv.csv",
        help="Path to Luke6 balance CSV file",
    )
    parser.add_argument(
        "--start-date",
        dest="start_date",
        default="placeholder_start_date",
        help="Start date for analysis (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        default="placeholder_end_date",
        help="End date for analysis (YYYY-MM-DD)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global start_date, end_date
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    
    print(f"\n[balance] Configuration: start_date={start_date.date()}, end_date={end_date.date()}")
    print()

    # Create output directories if they don't exist
    EXCEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    EXPERIMENTS_WITH_CONC = [
        ("Luke5", args.luke5_balance_csv, 212),
        ("Luke6", args.luke6_balance_csv, 352),
    ]
    all_results = {}
    
    for exp_name, exp_path, c_conc in EXPERIMENTS_WITH_CONC:
        df = load_csv_safely([exp_path])
        df = ensure_time_column(df)
        
        if exp_name == "Luke5":
            df_balance = extract_balance_columns(df, start_date, end_date, acid_col_idx=2, base_col_idx=4, medium_col_idx=17, medium_col_name="MT_80018.Weight High Resolution")
        else:
            df_balance = extract_balance_columns(df, start_date, end_date, acid_col_idx=4, base_col_idx=1, medium_col_idx=19, medium_col_name="MT_80100.Weight High Resolution.1")
        
        
        df_cycles = process_medium_scale_cycles(df_balance, start_date, end_date, c_concentration=c_conc)
        print(f"[{exp_name}] Generated {len(df_cycles)} cycles (cycles 1-{df_cycles['Cycle'].max()})")
        if exp_name in ["Luke5", "Luke6"] and 59 in df_cycles['Cycle'].values:
            # Manual patch for cycle 59 N-phase medium amount
            df_cycles.loc[df_cycles['Cycle'] == 59, 'N_medium_grams'] = 500
            df_cycles.loc[df_cycles['Cycle'] == 59, 'N_pump_rate_L_h'] = 500 / 1000.0 / 2.0
            df_cycles.loc[df_cycles['Cycle'] == 59, 'Nitrogen_added_mNmol'] = (500 / 1000.0) * 8.8
        exclude_base_cycle = 29 if exp_name == "Luke6" else None
        df_acid_base = process_acid_base_cycles(df_balance, start_date, end_date, acid_conc_M=2.0, base_conc_M=1.0, exclude_base_from_cycle=exclude_base_cycle)
        
        all_results[exp_name] = df_cycles
        all_results[f"{exp_name}_acid_base"] = df_acid_base
        
        summarize_balance_results(exp_name, df_cycles, df_acid_base)
    
    output_file = str(EXCEL_OUTPUT_DIR / "processed_balance_dataL5L6.xlsx")
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for exp_name, df_cycles in all_results.items():
                df_cycles.to_excel(writer, sheet_name=exp_name, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to write Excel file: {e}")
        raise
    
    print()
    print(f"[OK] Balance data exported to: {output_file}")


if __name__ == "__main__":
    main()
