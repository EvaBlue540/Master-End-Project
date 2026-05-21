"""Off-gas data processing for Luke 3 reactor.

This script processes off-gas data for Luke 3 with 18h cycles
(9h C-phase + 9h N-phase) starting at 9:00 each day.
"""

from datetime import timedelta
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# Set font to CMU Serif (Computer Modern - LaTeX default)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif', 'DejaVu Serif']
# Prevent math text from being italicized
plt.rcParams['mathtext.default'] = 'regular'

# ============================================================
# LOAD DATA & EXPERIMENTS DEFINITION
# ============================================================
path3 = "path_to_luke3_offgas.csv"  # Replace with actual path to Luke 3 off-gas data

# Experiments: (name, path, C_acetate in mCmol, cycle_code)
EXPERIMENTS = [
    ("Luke3", path3, 176, "CN40"),
]

# USER SETTINGS
start_date = pd.to_datetime("2025-11-08 09:00")
# End date: 2025-12-11 at 09:00 (cycle start) - 30 days of data
end_date = pd.to_datetime("2025-12-11 09:00")

# Daily integration blocks: 18h cycle = 9h C-phase (09:00-18:00) + 9h N-phase (18:00-03:00)
# The N-phase crosses midnight, handled by generate_integration_intervals
daily_blocks = [("09:00", "18:00"), ("18:00", "03:00")]  # No longer used, kept for reference

gasflow = 200  # mL/min
gas_volume = 22.4  # L/mol

N_load = 4.4  # mmol
N_content = 0.2  # N fraction in biomass

gamma_ac = 4.0  # Cmol/Cmol acetate
gamma_PHA = 4.5  # Cmol/Cmol PHA
gamma_biomass = 4.2  # Cmol/Cmol biomass

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_csv_safely(path_candidates):
    """Load CSV with multiple encoding and separator attempts."""
    for p in path_candidates:
        if not os.path.exists(p):
            print(f"  Warning: Path not found: {p}")
            continue
        
        for enc in ("utf-16", "utf-8", "latin-1"):
            for sep in (None, '\t', ','):
                try:
                    df = pd.read_csv(p, sep=sep, encoding=enc, engine='python')
                    if len(df.columns) < 3:
                        continue
                    if len(df) == 0:
                        print(f"  Warning: CSV loaded but empty from {p}")
                        continue
                    return df
                except Exception as e:
                    continue
    
    raise FileNotFoundError(
        f"Could not load CSV from any path. Tried {len(path_candidates)} paths with "
        "encodings (utf-16, utf-8, latin-1) and separators (None, tab, comma)"
    )


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
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    df = df.copy()
    time_col = df.columns[0]
    initial_rows = len(df)
    
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
    if na_count > len(df) * 0.1:  # More than 10% NaT - parsing likely failed
        df['time'] = pd.to_datetime(df[time_col], errors='coerce', dayfirst=True)
        na_count_after = df['time'].isna().sum()
        if na_count_after > na_count:
            print(f"  Warning: dayfirst parsing increased NaT count from {na_count} to {na_count_after}")
    else:
        # Dates parsed successfully. Only retry with dayfirst if many dates are way outside expected range
        # (not if they're just slightly off - they might be from system startup/shutdown)
        valid_times = df['time'].dropna()
        if len(valid_times) == 0:
            raise ValueError("No valid timestamps found in time column")
        
        # Check if dates make sense (allow some tolerance outside expected range)
        range_start = pd.to_datetime(f"{(expected_start - timedelta(days=2)).date()} 00:00:00")
        range_end = pd.to_datetime(f"{(expected_end + timedelta(days=2)).date()} 23:59:59")
        in_range = ((valid_times >= range_start) & (valid_times <= range_end)).sum()
        
        if in_range < len(valid_times) * 0.5:  # Less than 50% in reasonable range - definitely wrong
            print(f"  Warning: Only {in_range}/{len(valid_times)} timestamps in reasonable range, retrying with dayfirst=True")
            df_retry = df.copy()
            df_retry['time'] = pd.to_datetime(df_retry[time_col], errors='coerce', dayfirst=True)
            na_count_retry = df_retry['time'].isna().sum()
            # Only use retry if it doesn't make things significantly worse
            if na_count_retry <= na_count + 10:  # Allow small increase in NaT
                df['time'] = df_retry['time']
            else:
                print(f"  Note: dayfirst retry resulted in {na_count_retry} NaT (worse than {na_count}), keeping original parse")
    
    # Drop rows with NaT timestamps
    rows_with_nat = df['time'].isna().sum()
    if rows_with_nat > 0:
        print(f"  Note: Dropping {rows_with_nat} rows with invalid timestamps ({rows_with_nat/initial_rows*100:.1f}%)")
        df = df.dropna(subset=['time'])
    
    if len(df) == 0:
        raise ValueError("All timestamps were invalid after parsing")
    
    # Drop the original time column and make 'time' the first column
    df = df.drop(columns=[time_col])
    cols = ['time'] + [c for c in df.columns if c != 'time']
    df = df[cols]
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # Check for duplicate timestamps
    duplicates = df['time'].duplicated().sum()
    if duplicates > 0:
        print(f"  Warning: Found {duplicates} duplicate timestamps")
    
    # Verify data range
    actual_start = df['time'].min()
    actual_end = df['time'].max()
    print(f"  Data range: {actual_start} to {actual_end} ({len(df)} rows)")
    
    return df


def compute_gas_rates(
    df,
    gasflow_ml_min,
    gas_volume_l_per_mol,
):
    """Compute CO2 and O2 rates from gas fractions."""
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
    else:
        flow_ml_min = np.full(len(df), gasflow_ml_min)
    gas_in = (flow_ml_min / 1000.0) * 60.0 / gas_volume_l_per_mol * 1000.0
    N2_out = pd.to_numeric(df[n2_col], errors='coerce')
    CO2_frac = pd.to_numeric(df[co2_col], errors='coerce') / 100.0
    O2_frac = pd.to_numeric(df[o2_col], errors='coerce') / 100.0
    gas_out = gas_in * (N2_out / N2_in) 
    CO2_rate = CO2_frac * gas_out - (CO2_in / 100.0) * gas_in
    O2_rate = O2_frac * gas_out - (O2_in / 100.0) * gas_in
    return CO2_rate.astype(float), O2_rate.astype(float)


def generate_integration_intervals(start_date, end_date, daily_blocks):
    """Generate time intervals for 18h SBR cycles.
    
    Continuous 18h cycles with no gaps:
    - Each cycle: 9h C-phase + 9h N-phase
    - Next cycle starts immediately after previous N-phase ends
    - Pattern repeats every 3 days (4 cycles per 72h)
    
    Example starting at 2025-11-06 09:00:
      Cycle 1: C 09:00-18:00, N 18:00-03:00
      Cycle 2: C 03:00-12:00, N 12:00-21:00
      Cycle 3: C 21:00-06:00, N 06:00-15:00
      Cycle 4: C 15:00-00:00, N 00:00-09:00 (back to 09:00)
    """
    intervals = []
    current_time = start_date
    
    while current_time <= end_date:
        # C-phase: 9 hours
        c_phase_end = current_time + timedelta(hours=9)
        intervals.append((current_time, c_phase_end))
        
        # N-phase: 9 hours
        n_phase_end = c_phase_end + timedelta(hours=9)
        intervals.append((c_phase_end, n_phase_end))
        
        # Next cycle starts immediately after N-phase ends (continuous, no gap)
        current_time = n_phase_end
    
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
                    elapsed = (times - start_int).dt.total_seconds() / 3600.0
                    CO2_int = trapezoid(CO2_rate[mask], elapsed)
                    O2_int = trapezoid(O2_rate[mask], elapsed)
                    results.append({
                        "start": start_int,
                        "end": end_int,
                        "duration_h": (end_int - start_int).total_seconds() / 3600.0,
                        "CO2_mmol": CO2_int,
                        "O2_mmol": O2_int,
                        "data_points": len(times),
                        "data_quality": "COMPLETE",
                    })
    return pd.DataFrame(results)


def plot_offgas_phases(experiment_name, df, O2_uptake, CO2_rate, df_integrated,
                       df_feast_pha_sbr, df_pha_sbr, plot_type="feast",
                       cycle_code="", cycle_numbers=None):
    """Create detailed feast and full C-phase plots for selected cycles."""

    # Determine which cycles to plot
    if cycle_numbers is not None:
        cycles_to_plot = cycle_numbers
    else:
        # Default: last two complete cycles
        C_phases = df_integrated.iloc[::2]
        last_two_C = C_phases.tail(2)
        cycles_to_plot = [idx // 2 + 1 for idx in last_two_C.index]

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

        # Add Feast End line
        feast_end_added = False
        for phase_idx in cycle_indices:
            if phase_idx % 2 == 0:  # C-phase
                cycle_nr = phase_idx // 2 + 1
                feast_row = df_feast_pha_sbr.loc[df_feast_pha_sbr['Cycle'] == cycle_nr]
                if not feast_row.empty:
                    feast_end = feast_row['Feast end time'].values[0]
                    if pd.notna(feast_end) and not feast_end_added:
                        feast_end = pd.Timestamp(feast_end)
                        ax.axvline(feast_end, linestyle='--', color='#457B9D', linewidth=1.8, alpha=0.8,
                                   label=r'Feast End ($O_2$)')
                        feast_end_added = True

        # Background shading
        phase_labels_added = {'N-Phase': False, 'C-Phase': False, 'Feast-Phase': False}

        # Feast phase shading
        for phase_idx in cycle_indices:
            if phase_idx % 2 == 0:
                row = df_integrated.loc[phase_idx]
                phase_start = pd.to_datetime(row["start"])
                cycle_nr = phase_idx // 2 + 1
                feast_row = df_feast_pha_sbr.loc[df_feast_pha_sbr['Cycle'] == cycle_nr]
                if not feast_row.empty and pd.notna(feast_row['Feast end time'].values[0]):
                    feast_end = pd.to_datetime(feast_row['Feast end time'].values[0])
                    feast_mask = (df['time'] >= phase_start) & (df['time'] <= feast_end)
                    feast_times = df.loc[feast_mask, 'time'].values
                    feast_o2 = O2_uptake[feast_mask].values
                    ax.fill_between(feast_times, 0, feast_o2, alpha=0.18, color='#0093A3',
                                   label=r'Cumulative $O_2$ - Feast-Phase' if not phase_labels_added['Feast-Phase'] else '')
                    phase_labels_added['Feast-Phase'] = True

        # C-phase shading
        for phase_idx in cycle_indices:
            if phase_idx % 2 == 0:
                row = df_integrated.loc[phase_idx]
                phase_start = pd.to_datetime(row["start"])
                n_start = phase_start + pd.Timedelta(hours=9.0)  # C-phase is 9h, then N-phase starts
                phase_mask = (df['time'] >= phase_start) & (df['time'] <= n_start)
                phase_times = df.loc[phase_mask, 'time'].values
                phase_o2 = O2_uptake[phase_mask].values
                ax.fill_between(phase_times, 0, phase_o2, alpha=0.14, color='#00B2C4',
                               label=r'Cumulative $O_2$ - C-Phase' if not phase_labels_added['C-Phase'] else '')
                phase_labels_added['C-Phase'] = True

        # N-phase shading
        for phase_idx in cycle_indices:
            if phase_idx % 2 == 1:
                row = df_integrated.loc[phase_idx]
                phase_start = pd.to_datetime(row["start"])
                phase_end = pd.to_datetime(row["end"])
                phase_mask = (df['time'] >= phase_start) & (df['time'] <= phase_end)
                phase_times = df.loc[phase_mask, 'time'].values
                phase_o2 = O2_uptake[phase_mask].values
                ax.fill_between(phase_times, 0, phase_o2, alpha=0.10, color='#33CCD6',
                               label=r'Cumulative $O_2$ - N-Phase' if not phase_labels_added['N-Phase'] else '')
                phase_labels_added['N-Phase'] = True

        # Per-cycle annotations
        c_phase_indices = [idx for idx in cycle_indices if idx % 2 == 0]
        for cycle_count, phase_idx in enumerate(c_phase_indices):
            row = df_integrated.loc[phase_idx]
            cycle_nr = phase_idx // 2 + 1
            c_phase_start = row["start"]
            c_phase_end = row["end"]

            if plot_type == 'feast':
                feast_row = df_feast_pha_sbr.loc[df_feast_pha_sbr['Cycle'] == cycle_nr]
                if not feast_row.empty:
                    feast_end = feast_row['Feast end time'].values[0]
                    if pd.notna(feast_end):
                        feast_end = pd.Timestamp(feast_end)
                        feast_mask = (df['time'] >= c_phase_start) & (df['time'] <= feast_end)
                        feast_idx_positions = np.where(feast_mask.values)[0]
                        if len(feast_idx_positions) > 0:
                            feast_indices = df.index[feast_idx_positions]
                            feast_times = df.loc[feast_indices, 'time'].values
                            feast_o2 = O2_uptake.loc[feast_indices].values
                            ax.fill_between(feast_times, feast_o2, alpha=0.25, color='#A8DADC')
                    feast_length = feast_row['Feast phase length (h) - O2'].values[0]
                    pha_yield_val = feast_row['PHA yield feast (Cmol/Cmol)'].values[0]
                    label_text = f"{float(feast_length):.1f}h\nY$_{{PHA/Ac}}$={float(pha_yield_val):.2f}"
            else:
                c_phase_mask = (df['time'] >= c_phase_start) & (df['time'] <= c_phase_end)
                c_phase_times = df['time'][c_phase_mask]
                if len(c_phase_times) > 0:
                    ax.fill_between(c_phase_times, O2_uptake[c_phase_mask], alpha=0.25, color='#A8DADC')
                pha_row = df_pha_sbr.loc[df_pha_sbr['Cycle'] == cycle_nr]
                if not pha_row.empty:
                    pha_yield_val = pha_row['PHA yield (Cmol/Cmol)'].values[0]
                else:
                    pha_yield_val = np.nan
                c_duration = (pd.to_datetime(c_phase_end) - pd.to_datetime(c_phase_start)).total_seconds() / 3600
                label_text = f"{c_duration:.1f}h\nY$_{{PHA/Ac}}$={float(pha_yield_val):.2f}"

        # Phase boundary lines
        c_start = None
        for idx in cycle_indices:
            row = df_integrated.loc[idx]
            start = pd.to_datetime(row["start"])
            if idx % 2 == 0:
                c_start = start
                ax.axvline(c_start, color='black', linestyle='-', alpha=0.6, linewidth=2.2)
            else:
                if c_start is not None:
                    ax.axvline(c_start + pd.Timedelta(hours=9), color='#696969', linestyle='--', alpha=0.5, linewidth=1.5)

        ax.set_ylabel(r'$O_2$ Consumption / $CO_2$ Production (mmol/h)', fontsize=12, fontweight='bold')
        cycle_label = f" ({cycle_code})" if cycle_code else ""
        if plot_type == 'feast':
            title = f"Luke {experiment_name[-1]}{cycle_label} - Length of Feast Phase"
        else:
            title = f"Luke {experiment_name[-1]}{cycle_label} - Total C-Phase PHA Yield"
        ax.grid(True, linestyle='--', alpha=0.25, color='gray', which='major', axis='y')
        ax.set_axisbelow(True)

        # X-axis ticks at key event times
        important_times, important_labels = [], []
        for idx in cycle_indices:
            row = df_integrated.loc[idx]
            if idx % 2 == 0:
                c_phase_start = pd.to_datetime(row["start"])
                cycle_nr = idx // 2 + 1

                important_times.append(c_phase_start)
                important_labels.append(f"C-start\nt=0")

                if plot_type == 'feast':
                    feast_row = df_feast_pha_sbr.loc[df_feast_pha_sbr['Cycle'] == cycle_nr]
                    if not feast_row.empty and pd.notna(feast_row['Feast end time'].values[0]):
                        feast_end = pd.to_datetime(feast_row['Feast end time'].values[0])
                        important_times.append(feast_end)
                        feast_hours = (feast_end - c_phase_start).total_seconds() / 3600
                        important_labels.append(f"C-feast\nt={feast_hours:.1f}")

                n_start = c_phase_start + pd.Timedelta(hours=9)
                important_times.append(n_start)
                important_labels.append(f"N-start\nt=9")

                n_end = c_phase_start + pd.Timedelta(hours=18)
                important_times.append(n_end)
                important_labels.append(f"N-end\nt=18")

        if important_times:
            ax.set_xticks(important_times, minor=False)
            ax.set_xticklabels(important_labels, rotation=0, ha='center', fontsize=8)
            ax.set_xlim(important_times[0], important_times[-1])

        ax.xaxis.set_minor_locator(plt.matplotlib.dates.HourLocator(interval=12))
        ax.tick_params(axis='x', which='minor', length=3, width=0.5)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', fontsize=10, framealpha=0.95)

        return fig, ax

    # Plot each selected cycle
    for cycle_num in cycles_to_plot:
        c_phase_idx = 2 * (cycle_num - 1)
        n_phase_idx = c_phase_idx + 1

        if c_phase_idx in df_integrated.index and n_phase_idx in df_integrated.index:
            cycle_indices = [c_phase_idx, n_phase_idx]

            fig, ax = _plot_one(plot_type, cycle_indices)
            cycle_label = f"_cycle{cycle_num}" if cycle_numbers is not None else ""
            plot_type_label = "feastphase" if plot_type == "feast" else "totalcphase"
            filename = f"plots_EXP1/{experiment_name}{cycle_label}_{plot_type_label}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"* Plot saved: {filename}")

def process_experiment(experiment_name, csv_path, C_acetate, start_date, end_date, 
                       daily_blocks, gasflow, gas_volume, N_load, N_content, 
                       gamma_ac, gamma_PHA, gamma_biomass, cycle_code=""):
    """
    Process a single off-gas experiment.
    Generates Excel file and plots with experiment-specific names.
    """
    print(f"\n{'='*60}")
    print(f"Processing {experiment_name}...")
    print(f"Data range: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    
    # Load and prepare data
    df = load_csv_safely([csv_path])
    df = ensure_time_column(df)

    # Generate integration intervals first to determine cycle/phase timing.
    integration_intervals = generate_integration_intervals(start_date, end_date, daily_blocks)
    
    # DEBUG: Print interval structure
    print(f"\nGenerated {len(integration_intervals)} intervals ({len(integration_intervals)//2} complete 18h cycles)")
    print(f"First cycle (Cycle 1):")
    if len(integration_intervals) >= 2:
        s, e = integration_intervals[0]
        print(f"  C-phase: {s.strftime('%Y-%m-%d %H:%M')} to {e.strftime('%Y-%m-%d %H:%M')} (9h)")
        s, e = integration_intervals[1]
        print(f"  N-phase: {s.strftime('%Y-%m-%d %H:%M')} to {e.strftime('%Y-%m-%d %H:%M')} (9h)")
    if len(integration_intervals) > 3:
        print(f"Last cycle (Cycle {len(integration_intervals)//2}):")
        s, e = integration_intervals[-2]
        print(f"  C-phase: {s.strftime('%Y-%m-%d %H:%M')} to {e.strftime('%Y-%m-%d %H:%M')} (9h)")
        s, e = integration_intervals[-1]
        print(f"  N-phase: {s.strftime('%Y-%m-%d %H:%M')} to {e.strftime('%Y-%m-%d %H:%M')} (9h)")
    
    # Compute gas rates
    CO2_rate, O2_rate = compute_gas_rates(
        df,
        gasflow,
        gas_volume,
    )
    CO2_rate = pd.Series(CO2_rate, index=df.index)
    O2_rate = pd.Series(O2_rate, index=df.index)
    
    # Integrate over intervals
    df_integrated = integrate_intervals(df, CO2_rate, O2_rate, integration_intervals)
    
    # Keep all cycles (gaps still represent real experimental cycles)
    pha_results = []
    feast_pha_results = []
    famine_pha_results = []

    O2_uptake = -O2_rate

    exchange_ratio = 0.5
    PHA_leftover = 0

    # Main PHA calculation loop - process all cycles (incomplete cycles included)
    num_intervals = len(df_integrated)
    for i in range(0, num_intervals - 1, 2):
        C_phase = df_integrated.iloc[i]
        N_phase = df_integrated.iloc[i + 1] if i + 1 < num_intervals else None
        cycle_number = i // 2 + 1
        
        # Determine cycle data quality (COMPLETE only if both phases are COMPLETE)
        c_phase_quality = C_phase['data_quality']
        n_phase_quality = N_phase['data_quality'] if N_phase is not None else 'MISSING'
        cycle_data_quality = 'COMPLETE' if (c_phase_quality == 'COMPLETE' and n_phase_quality == 'COMPLETE') else c_phase_quality

        CO2_C = C_phase['CO2_mmol']
        O2_C = -C_phase['O2_mmol']
        CO2_N = N_phase['CO2_mmol'] if N_phase is not None else np.nan
        O2_N = -N_phase['O2_mmol'] if N_phase is not None else np.nan

        start_C = C_phase['start']
        end_C = C_phase['end']
        
        # Special case adjustments for data gaps
        is_special_case = False
        # No special cases for Luke 3 - use standard processing
        
        mask_C = (df['time'] >= start_C) & (df['time'] <= end_C)
        time_segment = df['time'][mask_C]

        time_array = (time_segment - start_C).dt.total_seconds().values / 3600.0
        O2_C_array = O2_uptake[mask_C].values
        CO2_C_array = CO2_rate[mask_C].values
        
        # Check if we have enough valid data points for processing
        num_valid_o2 = np.sum(~np.isnan(O2_C_array))
        num_valid_co2 = np.sum(~np.isnan(CO2_C_array))
        has_valid_data = (num_valid_o2 >= 5 and num_valid_co2 >= 5)
        
        # Only process feast/famine phases if data quality is COMPLETE or if it's a special case with valid data
        if (cycle_data_quality == 'COMPLETE' or (is_special_case and has_valid_data)) and has_valid_data:
            # Re-integrate for special cases with adjusted times
            if is_special_case:
                # For all special cases, use the re-extracted and filtered values
                O2_C = trapezoid(O2_C_array, time_array)
                CO2_C = trapezoid(CO2_C_array, time_array)
            else:
                # Use the pre-integrated values from df_integrated
                CO2_C = C_phase['CO2_mmol']
                O2_C = -C_phase['O2_mmol']
            
            # Mark special cases with valid data as COMPLETE for feast/famine processing
            if is_special_case:
                cycle_data_quality = 'COMPLETE'

            # Smooth gas signals with minimal processing (3-point rolling mean)
            window = 3
            O2_smooth = pd.Series(O2_C_array).rolling(window=window, center=True, min_periods=1).mean().values
            CO2_smooth = pd.Series(CO2_C_array).rolling(window=window, center=True, min_periods=1).mean().values

            # Detect feast phase end with improved robustness
            # - Requires sustained low gradient (not just single fluctuation)
            # - Applies boundary conditions: 2 ≤ feast_length ≤ 6.5 hours
            min_feast_time = 2.0
            max_feast_time = 6.5
            
            # Parameters for gradient-based detection (Luke 3)
            gradient_threshold = -10  # Steep slope required to detect feast end
            min_sustained_points = 2  # Require 4 consecutive points (30 min with 15-min intervals)
            
            valid_indices = np.where(time_array >= min_feast_time)[0]
            if len(valid_indices) == 0:
                feast_end_idx = len(time_array) - 1
            else:
                first_valid = valid_indices[0]
                dO2dt = np.gradient(O2_smooth, time_array)
                
                # Find sustained drop: requires min_sustained_points consecutive points below threshold
                feast_end_idx = None
                for i in range(first_valid, len(dO2dt) - min_sustained_points + 1):
                    sustained_drop = np.all(dO2dt[i:i+min_sustained_points] < gradient_threshold)
                    if sustained_drop:
                        feast_end_idx = i
                        break
                
                # Fallback: if no sustained drop found, use when O2 drops to 65% of peak (not the peak itself)
                if feast_end_idx is None:
                    peak_o2 = np.max(O2_smooth[first_valid:])
                    o2_threshold = 0.65 * peak_o2
                    # Find first point after peak where O2 drops below threshold
                    peak_idx = first_valid + np.argmax(O2_smooth[first_valid:])
                    after_peak = O2_smooth[peak_idx:]
                    drop_indices = np.where(after_peak < o2_threshold)[0]
                    if len(drop_indices) > 0:
                        feast_end_idx = peak_idx + drop_indices[0]
                    else:
                        # If still no drop, use 80% through the phase
                        feast_end_idx = first_valid + int(0.8 * (len(O2_smooth) - first_valid))
                
                # Apply maximum feast time boundary (6.5 hours)
                max_idx = np.searchsorted(time_array, max_feast_time)
                if max_idx > 0:
                    feast_end_idx = min(feast_end_idx, max_idx)
            
            feast_end_idx = min(feast_end_idx, len(time_array) - 1)

            CO2_feast = CO2_C_array[: feast_end_idx + 1]
            O2_feast = O2_C_array[: feast_end_idx + 1]
            time_feast = time_array[: feast_end_idx + 1]
            feast_length_h = time_feast[-1] if len(time_feast) > 0 else 0
            
            # Validate feast length boundaries
            # Minimum: 2 hours (can be exactly 2), Maximum: 6.5 hours
            if feast_length_h < min_feast_time:
                # Feast too short - extend to minimum 2 hours
                min_idx = np.searchsorted(time_array, min_feast_time)
                if min_idx < len(time_array):
                    feast_end_idx = min_idx
                    CO2_feast = CO2_C_array[: feast_end_idx + 1]
                    O2_feast = O2_C_array[: feast_end_idx + 1]
                    time_feast = time_array[: feast_end_idx + 1]
                    feast_length_h = time_feast[-1] if len(time_feast) > 0 else 0
            elif feast_length_h > max_feast_time:
                # Feast too long - truncate to maximum 6.5 hours
                max_idx = np.searchsorted(time_array, max_feast_time)
                if max_idx > 0:
                    feast_end_idx = max_idx
                    CO2_feast = CO2_C_array[: feast_end_idx + 1]
                    O2_feast = O2_C_array[: feast_end_idx + 1]
                    time_feast = time_array[: feast_end_idx + 1]
                    feast_length_h = time_feast[-1] if len(time_feast) > 0 else 0
            
            CO2_int_feast = trapezoid(CO2_feast, time_feast) if len(time_feast) > 1 else 0
            O2_int_feast = trapezoid(O2_feast, time_feast) if len(time_feast) > 1 else 0

            # Famine part
            if feast_end_idx < len(time_array) - 1:
                CO2_famine = CO2_C_array[feast_end_idx + 1 :]
                O2_famine = O2_C_array[feast_end_idx + 1 :]
                time_famine = time_array[feast_end_idx + 1 :]
                CO2_int_famine = trapezoid(CO2_famine, time_famine) if len(time_famine) > 1 else 0
                O2_int_famine = trapezoid(O2_famine, time_famine) if len(time_famine) > 1 else 0
            else:
                CO2_int_famine = 0
                O2_int_famine = 0

            e_resp_famine = 4 * O2_int_famine
            PHA_burned_famine = e_resp_famine / gamma_PHA if gamma_PHA != 0 else np.nan
            PHA_from_CO2 = CO2_int_famine
            if CO2_int_famine > 1e-6 and PHA_burned_famine > 0:
                consistency_ratio = PHA_burned_famine / PHA_from_CO2 if PHA_from_CO2 != 0 else np.nan
                consistency_deviation = (consistency_ratio - 1) * 100 if not np.isnan(consistency_ratio) else np.nan
            else:
                consistency_ratio = np.nan
                consistency_deviation = np.nan

            # C-phase PHA formation
            e_in_C = gamma_ac * C_acetate
            e_resp_C = 4 * O2_C
            e_PHA_C = e_in_C - e_resp_C
            PHA_new = e_PHA_C / gamma_PHA if gamma_PHA != 0 else np.nan
            pha_yield = PHA_new / C_acetate if C_acetate != 0 else np.nan

            e_PHA_feast = e_in_C - 4 * O2_int_feast
            PHA_feast = e_PHA_feast / gamma_PHA if gamma_PHA != 0 else np.nan
            Y_PHA_feast = PHA_feast / C_acetate if C_acetate != 0 else np.nan

            PHA_total = PHA_leftover + PHA_new
            PHA_removed = exchange_ratio * PHA_total
            PHA_available_N = (1 - exchange_ratio) * PHA_total

            Biomass_max = N_load / N_content if N_content != 0 else np.nan
            e_needed = gamma_biomass * Biomass_max + 4 * O2_N
            PHA_needed = e_needed / gamma_PHA if gamma_PHA != 0 else np.nan

            if not np.isnan(PHA_available_N) and not np.isnan(PHA_needed) and PHA_available_N >= PHA_needed:
                Biomass_actual = Biomass_max
                PHA_consumed = PHA_needed
                limitation = "N-limited"
            else:
                Biomass_actual = (gamma_PHA * PHA_available_N - 4 * O2_N) / gamma_biomass if gamma_biomass != 0 else np.nan
                PHA_consumed = PHA_available_N
                limitation = "PHA-limited"

            PHA_leftover = PHA_available_N - PHA_consumed
            PHA_fraction_effluent = PHA_removed / (Biomass_actual + PHA_removed) * 100 if (Biomass_actual + PHA_removed) != 0 else 0
            biomass_yield = Biomass_actual / PHA_consumed if PHA_consumed != 0 else np.nan
            
            feast_end_time = df['time'][mask_C].iloc[feast_end_idx] if len(df['time'][mask_C]) > feast_end_idx else np.nan
        else:
            # Incomplete data - all metric calculations return NaN
            PHA_new = PHA_removed = PHA_consumed = pha_yield = np.nan
            biomass_yield = Biomass_actual = np.nan
            CO2_int_feast = O2_int_feast = feast_length_h = np.nan
            PHA_feast = Y_PHA_feast = np.nan
            CO2_int_famine = O2_int_famine = np.nan
            PHA_burned_famine = PHA_from_CO2 = np.nan
            consistency_ratio = consistency_deviation = np.nan
            feast_end_time = np.nan
            limitation = "INSUFFICIENT_DATA"
            PHA_fraction_effluent = 0
            # PHA_leftover persists across cycles

        C_phase = df_integrated.iloc[i]
        cycle_date = pd.to_datetime(C_phase['start']).date()  # Gets just the date part


        pha_results.append({
            "Cycle": cycle_number,
            "Cycle Date": pd.to_datetime(C_phase['start']).date(),
            "Data Quality": cycle_data_quality,
            "PHA new (mCmol)": PHA_new,
            "PHA yield (Cmol/Cmol)": pha_yield,
            "PHA removed (mCmol)": PHA_removed,
            "PHA consumed N (mCmol)": PHA_consumed,
            "Biomass yield (Cmol/Cmol)": biomass_yield,
            "PHA leftover (mCmol)": PHA_leftover,
            "Biomass formed (mCmol)": Biomass_actual,
            "Limitation": limitation,
            "PHA content": PHA_fraction_effluent,
        })

        feast_pha_results.append({
            "Cycle": cycle_number,
            "Cycle Date": pd.to_datetime(C_phase['start']).date(),
            "Data Quality": cycle_data_quality,
            "PHA new (mCmol)": PHA_new,
            "PHA yield (Cmol/Cmol)": pha_yield,
            "Feast phase length (h) - O2": feast_length_h,
            "PHA feast (mCmol)": PHA_feast,
            "PHA yield feast (Cmol/Cmol)": Y_PHA_feast,
            "Feast end time": feast_end_time,
        })

        famine_pha_results.append({
            "Cycle": cycle_number,
            "Cycle Date": pd.to_datetime(C_phase['start']).date(),
            "Data Quality": cycle_data_quality,
            "CO2 famine (mmol)": CO2_int_famine,
            "O2 famine (mmol)": O2_int_famine,
            "PHA burned famine (mCmol)": PHA_burned_famine,
            "PHA burned famine (CO2-based)": PHA_from_CO2,
            "Consistency ratio (O2/CO2)": consistency_ratio,
            "Consistency deviation (%)": consistency_deviation,
        })

    # Convert to DataFrames
    df_pha_sbr = pd.DataFrame(pha_results).round(2)
    df_feast_pha_sbr = pd.DataFrame(feast_pha_results).round(2)
    df_famine_pha_sbr = pd.DataFrame(famine_pha_results).round(2)

    # Print feast phase dataframe
    print(f"\n{'='*60}")
    print(f"Feast Phase DataFrame for {experiment_name}:")
    print(f"{'='*60}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_feast_pha_sbr.to_string(index=False))
    print(f"{'='*60}\n")

    # Print cycle dates
    print(f"Cycle Dates for {experiment_name}:")
    print(f"{'='*80}")
    for i in range(0, len(df_integrated) - 1, 2):
        C_phase = df_integrated.iloc[i]
        N_phase = df_integrated.iloc[i + 1] if i + 1 < len(df_integrated) else None
        cycle_number = i // 2 + 1
        
        c_quality = C_phase.get('data_quality', 'COMPLETE')
        n_quality = N_phase['data_quality'] if N_phase is not None else 'MISSING'
        
        c_start = pd.to_datetime(C_phase['start']).strftime('%Y-%m-%d %H:%M')
        c_end = pd.to_datetime(C_phase['end']).strftime('%Y-%m-%d %H:%M')
        n_start = pd.to_datetime(N_phase['start']).strftime('%Y-%m-%d %H:%M') if N_phase is not None else 'N/A'
        n_end = pd.to_datetime(N_phase['end']).strftime('%Y-%m-%d %H:%M') if N_phase is not None else 'N/A'
        
        print(f"Cycle {cycle_number:2d} | C-phase: {c_start} - {c_end} [{c_quality:12s}] | N-phase: {n_start} - {n_end} [{n_quality:12s}]")
    print(f"{'='*80}\n")

    # Ensure output directories exist
    os.makedirs("plots_EXP1", exist_ok=True)
    
    # Generate plots using consolidated function
    # Plot last two cycles (default)
    plot_offgas_phases(experiment_name, df, O2_uptake, CO2_rate, df_integrated, 
                   df_feast_pha_sbr, df_pha_sbr, plot_type="feast", cycle_code=cycle_code)
    plot_offgas_phases(experiment_name, df, O2_uptake, CO2_rate, df_integrated, 
                   df_feast_pha_sbr, df_pha_sbr, plot_type="full", cycle_code=cycle_code)
    
    # Plot individual cycles (select every ~4 cycles for ~36 total cycles)
    selected_cycles = [1, 2, 5, 9, 20, 25, 30, 37, 41]
    for cycle_num in selected_cycles:
        plot_offgas_phases(experiment_name, df, O2_uptake, CO2_rate, df_integrated, 
                       df_feast_pha_sbr, df_pha_sbr, plot_type="feast", cycle_code=cycle_code,
                       cycle_numbers=[cycle_num])
        plot_offgas_phases(experiment_name, df, O2_uptake, CO2_rate, df_integrated, 
                       df_feast_pha_sbr, df_pha_sbr, plot_type="full", cycle_code=cycle_code,
                       cycle_numbers=[cycle_num])
    

# ============================================================
    gas_balance = []
    for i in range(0, len(df_integrated) - 1, 2):
        C_phase = df_integrated.iloc[i]
        N_phase = df_integrated.iloc[i + 1] if i + 1 < len(df_integrated) else None
        cycle_number = i // 2 + 1
        
        CO2_C = C_phase['CO2_mmol']
        O2_C = -C_phase['O2_mmol']
        CO2_N = N_phase['CO2_mmol'] if N_phase is not None else np.nan
        O2_N = -N_phase['O2_mmol'] if N_phase is not None else np.nan
        
        c_phase_quality = C_phase['data_quality']
        n_phase_quality = N_phase['data_quality'] if N_phase is not None else 'MISSING'
        cycle_quality = c_phase_quality if c_phase_quality != 'COMPLETE' or n_phase_quality != 'COMPLETE' else 'COMPLETE'
        
        gas_balance.append({
            'Cycle': cycle_number,
            'Cycle Date': pd.to_datetime(C_phase['start']).date(),
            'Data Quality': cycle_quality,
            'CO2 C-phase (mmol)': CO2_C,
            'O2 C-phase (mmol)': O2_C,
            'CO2 N-phase (mmol)': CO2_N,
            'O2 N-phase (mmol)': O2_N,
            'CO2 total (mmol)': CO2_C + CO2_N if not np.isnan(CO2_C) and not np.isnan(CO2_N) else np.nan,
            'O2 total (mmol)': O2_C + O2_N if not np.isnan(O2_C) and not np.isnan(O2_N) else np.nan,
        })

    df_gas_balance = pd.DataFrame(gas_balance).round(2)
    
    # Calculate cycle count (all cycles including incomplete ones)
    num_cycles = len(pha_results)
    complete_cycles = sum(1 for r in pha_results if r['Data Quality'] == 'COMPLETE')
    incomplete_cycles = num_cycles - complete_cycles
  
    # Save Excel with experiment-specific filename
    out_xlsx = f"Excel output/PHA_SBR_results_{experiment_name}_EXP1_18h.xlsx"
    with pd.ExcelWriter(out_xlsx) as writer:
        df_pha_sbr.to_excel(writer, sheet_name="PHA_results", index=False)
        df_feast_pha_sbr.to_excel(writer, sheet_name="Feast_PHA", index=False)
        df_famine_pha_sbr.to_excel(writer, sheet_name='Famine_PHA', index=False)
        df_gas_balance.to_excel(writer, sheet_name="gas summary", index=False)
    print(f"* {experiment_name}: {num_cycles} total cycles ({complete_cycles} complete, {incomplete_cycles} with missing/insufficient data)")
    print(f"* Excel file saved: {out_xlsx}")

    # Create a beautiful plot showing last two complete cycles (C+N phases)
    fig, ax = plt.subplots(figsize=(15, 7))
    
    C_phases = df_integrated.iloc[::2]  # Get C phases (even indices)
    N_phases = df_integrated.iloc[1::2]  # Get N phases (odd indices)
    last_two_C = C_phases.tail(2)
    last_two_cycles = []
    
    # Collect last two complete cycles
    for idx in last_two_C.index:
        last_two_cycles.append(idx)
        # Find corresponding N phase
        n_idx = idx + 1
        if n_idx in N_phases.index:
            last_two_cycles.append(n_idx)
    
    # Plot continuous O2 consumption across all phases
    all_times, all_o2, all_co2 = [], [], []
    for phase_idx in last_two_cycles:
        row = df_integrated.loc[phase_idx]
        mask = (df['time'] >= row["start"]) & (df['time'] <= row["end"])
        all_times.extend(df['time'][mask])
        all_o2.extend(O2_uptake[mask])
        all_co2.extend(CO2_rate[mask])
    
    # Plot as continuous line on same axis (CO2 first so it's in background)
    ax.plot(all_times, all_co2, color='#E63946', linewidth=2.5, label='CO₂ Production')
    ax.plot(all_times, all_o2, color='#2E86AB', linewidth=2.5, label='O₂ Consumption')
    
    c_phase_indices = [idx for idx in last_two_cycles if idx % 2 == 0]
    for cycle_count, phase_idx in enumerate(c_phase_indices):
        row = df_integrated.loc[phase_idx]
        cycle_nr = phase_idx // 2 + 1
        feast_row = df_feast_pha_sbr.loc[df_feast_pha_sbr["Cycle"] == cycle_nr]
        if not feast_row.empty:
            feast_start = row["start"]
            feast_end = feast_row["Feast end time"].values[0]
            c_phase_end = row["end"]
            feast_mask = (df['time'] >= feast_start) & (df['time'] <= feast_end)
            feast_times = df['time'][feast_mask]
            if len(feast_times) > 0:
                ax.fill_between(feast_times, O2_uptake[feast_mask], alpha=0.25, color='#A8DADC',
                               label='Total O₂ Feast-Phase' if cycle_count == 0 else '')
                ax.axvline(feast_end, linestyle='--', color='#457B9D', linewidth=1.8, alpha=0.8)
                # Note: Effluent timing is reactor-specific and not applicable to continuous 18h cycles
                # Cycles can start at 09:00, 03:00, 21:00, or 15:00, so fixed times don't work
                feast_hours = (pd.to_datetime(feast_end) - pd.to_datetime(feast_start)).total_seconds() / 3600
                pha_yield = feast_row["PHA yield feast (Cmol/Cmol)"].values[0]
                # Place annotation at midpoint of C-phase (works for all cycle types)
                c_phase_midpoint = pd.to_datetime(feast_start) + (pd.to_datetime(feast_end) - pd.to_datetime(feast_start)) / 2
                y_lim = ax.get_ylim()
                ax.annotate(f'{feast_hours:.1f}h\nY$_{{PHA/Ac}}$={pha_yield:.2f}',
                           xy=(c_phase_midpoint, y_lim[1] * 0.5), xytext=(0, 0),
                           textcoords='offset points', fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='#D4A574', alpha=0.85, edgecolor='none'),
                           color='white', ha='center', va='center')
    
    y_lim = ax.get_ylim()
    for idx in last_two_cycles:
        row = df_integrated.loc[idx]
        start = pd.to_datetime(row["start"])
        if idx % 2 == 0:
            ax.axvline(start, color='black', linestyle='-', alpha=0.6, linewidth=2.2)
            ax.text(start, y_lim[1] * 0.98, f'Cycle {idx // 2 + 1}',
                   fontsize=11, fontweight='bold', rotation=0,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8E8E8', alpha=0.85, edgecolor='black', linewidth=1.5),
                   ha='center', va='top')
        else:
            ax.axvline(start, color='#D9B3E6', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(start, y_lim[1] * 0.88, 'N Phase',
                   fontsize=9, fontweight='normal', rotation=0,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6CCFF', alpha=0.7, edgecolor='gray', linewidth=1),
                   ha='center', va='top')
    
    ax.set_xlabel("Date & Time", fontsize=12, fontweight='bold')
    ax.set_ylabel("O₂ Consumption (mmol/h)", fontsize=12, fontweight='bold')
    cycle_label = f" ({cycle_code})" if cycle_code else ""
    ax.set_title(f"Luke {experiment_name[-1]}{cycle_label} - Length of Feast Phase",
                fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, linestyle='--', alpha=0.25, color='gray', which='major', axis='y')
    ax.set_axisbelow(True)
    
    important_times, important_labels = [], []
    for idx in last_two_cycles:
        row = df_integrated.loc[idx]
        start = pd.to_datetime(row["start"])
        important_times.append(start)
        important_labels.append(start.strftime('%m-%d\n%H:%M'))
        if idx % 2 == 0:
            cycle_nr = idx // 2 + 1
            feast_row = df_feast_pha_sbr.loc[df_feast_pha_sbr["Cycle"] == cycle_nr]
            if not feast_row.empty:
                feast_end = pd.to_datetime(feast_row["Feast end time"].values[0])
                if pd.notna(feast_end):
                    important_times.append(feast_end)
                    important_labels.append(feast_end.strftime('%m-%d\n%H:%M'))
    
    if important_times:
        ax.set_xticks(important_times, minor=False)
        ax.set_xticklabels(important_labels, rotation=0, ha='center', fontsize=8)
    # Use 6-hour intervals for minor ticks (coarser than hourly to avoid too many ticks)
    ax.xaxis.set_minor_locator(plt.matplotlib.dates.HourLocator(interval=6))
    ax.tick_params(axis='x', which='minor', length=3, width=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=10, framealpha=0.95)
    
    plot_filename = f"plots_EXP1/{experiment_name}_feastphase_EXP1_18h.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"* Plot saved: {plot_filename}\n")

    fig, ax = plt.subplots(figsize=(15, 7))
    C_phases = df_integrated.iloc[::2]
    N_phases = df_integrated.iloc[1::2]
    last_two_C = C_phases.tail(2)
    last_two_cycles = []
    for idx in last_two_C.index:
        last_two_cycles.append(idx)
        n_idx = idx + 1
        if n_idx in N_phases.index:
            last_two_cycles.append(n_idx)
    
    all_times, all_o2, all_co2 = [], [], []
    for phase_idx in last_two_cycles:
        row = df_integrated.loc[phase_idx]
        mask = (df['time'] >= row["start"]) & (df['time'] <= row["end"])
        all_times.extend(df['time'][mask])
        all_o2.extend(O2_uptake[mask])
        all_co2.extend(CO2_rate[mask])
    
    ax.plot(all_times, all_co2, color='#E63946', linewidth=2.5, label='CO₂ Production')
    ax.plot(all_times, all_o2, color='#2E86AB', linewidth=2.5, label='O₂ Consumption')
    c_phase_indices = [idx for idx in last_two_cycles if idx % 2 == 0]
    for cycle_count, phase_idx in enumerate(c_phase_indices):
        row = df_integrated.loc[phase_idx]
        cycle_nr = phase_idx // 2 + 1
        pha_row = df_pha_sbr.loc[df_pha_sbr["Cycle"] == cycle_nr]
        if not pha_row.empty:
            c_phase_start = row["start"]
            c_phase_end = row["end"]
            c_phase_mask = (df['time'] >= c_phase_start) & (df['time'] <= c_phase_end)
            c_phase_times = df['time'][c_phase_mask]
            if len(c_phase_times) > 0:
                ax.fill_between(c_phase_times, O2_uptake[c_phase_mask], alpha=0.25, color='#A8DADC',
                               label='Total O₂ C-Phase' if cycle_count == 0 else '')
                # Note: Effluent timing is reactor-specific and not applicable to continuous 18h cycles
                # Cycles can start at 09:00, 03:00, 21:00, or 15:00, so fixed times don't work
                c_phase_hours = (pd.to_datetime(c_phase_end) - pd.to_datetime(c_phase_start)).total_seconds() / 3600
                pha_yield = pha_row["PHA yield (Cmol/Cmol)"].values[0]
                # Place annotation at midpoint of C-phase (works for all cycle types)
                c_phase_midpoint = pd.to_datetime(c_phase_start) + (pd.to_datetime(c_phase_end) - pd.to_datetime(c_phase_start)) / 2
                y_lim = ax.get_ylim()
                ax.annotate(f'{c_phase_hours:.1f}h\nY$_{{PHA/Ac}}$={pha_yield:.2f}',
                           xy=(c_phase_midpoint, y_lim[1] * 0.5), xytext=(0, 0),
                           textcoords='offset points', fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='#D4A574', alpha=0.85, edgecolor='none'),
                           color='white', ha='center', va='center')
    
    y_lim = ax.get_ylim()
    for idx in last_two_cycles:
        row = df_integrated.loc[idx]
        start = pd.to_datetime(row["start"])
        if idx % 2 == 0:
            ax.axvline(start, color='black', linestyle='-', alpha=0.6, linewidth=2.2)
            ax.text(start, y_lim[1] * 0.98, f'Cycle {idx // 2 + 1}',
                   fontsize=11, fontweight='bold', rotation=0,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8E8E8', alpha=0.85, edgecolor='black', linewidth=1.5),
                   ha='center', va='top')
        else:
            ax.axvline(start, color='#D9B3E6', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(start, y_lim[1] * 0.88, 'N Phase',
                   fontsize=9, fontweight='normal', rotation=0,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6CCFF', alpha=0.7, edgecolor='gray', linewidth=1),
                   ha='center', va='top')
    
    ax.set_xlabel("Date & Time", fontsize=12, fontweight='bold')
    ax.set_ylabel("O₂ Consumption (mmol/h)", fontsize=12, fontweight='bold')
    cycle_label = f" ({cycle_code})" if cycle_code else ""
    ax.set_title(f"Luke {experiment_name[-1]}{cycle_label} - Total C-Phase PHA Yield",
                fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, linestyle='--', alpha=0.25, color='gray', which='major', axis='y')
    ax.set_axisbelow(True)
    
    important_times, important_labels = [], []
    for idx in last_two_cycles:
        row = df_integrated.loc[idx]
        start = pd.to_datetime(row["start"])
        important_times.append(start)
        important_labels.append(start.strftime('%m-%d\n%H:%M'))
    
    if important_times:
        ax.set_xticks(important_times, minor=False)
        ax.set_xticklabels(important_labels, rotation=0, ha='center', fontsize=8)
    # Use 6-hour intervals for minor ticks (coarser than hourly to avoid too many ticks)
    ax.xaxis.set_minor_locator(plt.matplotlib.dates.HourLocator(interval=6))
    ax.tick_params(axis='x', which='minor', length=3, width=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=10, framealpha=0.95)
    
    plot_filename2 = f"plots_EXP1/{experiment_name}_totalcphase_EXP1.png"
    plt.tight_layout()
    plt.savefig(plot_filename2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"* Plot saved: {plot_filename2}\n")


# ============================================================
# MAIN EXECUTION: Process Luke 3
# ============================================================
if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs('Excel output', exist_ok=True)
    os.makedirs('plots_EXP1', exist_ok=True)
    
    for experiment_name, csv_path, C_acetate, cycle_code in EXPERIMENTS:
        process_experiment(
            experiment_name, csv_path, C_acetate,
            start_date, end_date, daily_blocks,
            gasflow, gas_volume,
            N_load, N_content,
            gamma_ac, gamma_PHA, gamma_biomass,
            cycle_code=cycle_code,
        )
        

         
    
    print(f"\n{'='*60}")
    print("Luke 3 experiment processed successfully!")
    print(f"Data range: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Cycle configuration: 18h (9h C-phase + 9h N-phase)")
    print(f"C_acetate = 176 mCmol")
    print(f"{'='*60}")
