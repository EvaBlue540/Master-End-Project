
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Set font to CMU Serif (Computer Modern - LaTeX default)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif', 'DejaVu Serif']
# Prevent math text from being italicized
plt.rcParams['mathtext.default'] = 'regular'


BASE_DIR = Path(__file__).resolve().parent
EXCEL_CANDIDATES = [
    BASE_DIR / "Excel output" / "pipeline_combined_results.xlsx",
    BASE_DIR / "pipeline_combined_results.xlsx",
    BASE_DIR / "Excel output" / "balance_offgas_calculations.xlsx",
    BASE_DIR / "balance_offgas_calculations.xlsx",
]
OUTPUT_DIR = BASE_DIR / "plots2" / "mass_balances" / "stable_average"
COMBINED_OUTPUT_DIR = BASE_DIR / "plots2" / "mass_balances" / "combined"
SHEETS = ["offgas_Luke5", "offgas_Luke6"]
CYCLE_RANGES = {
    "offgas_Luke5": (16, 94),
    "offgas_Luke6": (17, 80),
}


def load_workbook() -> dict[str, pd.DataFrame]:
    excel_file = next((path for path in EXCEL_CANDIDATES if path.exists()), None)
    if excel_file is None:
        raise FileNotFoundError(
            "Could not find pipeline_combined_results.xlsx or balance_offgas_calculations.xlsx in Excel output/ or the script directory."
        )

    all_data: dict[str, pd.DataFrame] = {}
    xls = pd.ExcelFile(excel_file)
    for sheet_name in xls.sheet_names:
        try:
            all_data[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
        except Exception as exc:
            pass

    return all_data


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def filter_stable_cycles(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Filter cycles in range for the given sheet and exclude SKIPPED cycles."""
    cycle_start, cycle_end = CYCLE_RANGES.get(sheet_name, (16, 94))
    # Filter by cycle range
    df_filtered = df[(df["Cycle"] >= cycle_start) & (df["Cycle"] <= cycle_end)].copy()
    
    # Exclude skipped cycles if Limitation column exists
    if "Limitation" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Limitation"] != "SKIPPED"]
    
    return df_filtered


def get_cycle_range(sheet_name: str) -> tuple[int, int]:
    """Get the cycle range for the given sheet."""
    return CYCLE_RANGES.get(sheet_name, (16, 94))


def series_for(df: pd.DataFrame, *column_names: str) -> pd.Series:
    for column_name in column_names:
        if column_name in df.columns:
            return df[column_name]
    available = ", ".join(df.columns)
    raise KeyError(f"None of these columns were found: {column_names}. Available columns: {available}")


def save_figure(fig: plt.Figure, filename: str, reactor_dir: str = "") -> None:
    if reactor_dir:
        path = OUTPUT_DIR / reactor_dir / filename
    else:
        path = OUTPUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_combined_figure(fig: plt.Figure, filename: str) -> None:
    path = COMBINED_OUTPUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_c_phase_combined(all_data: dict[str, pd.DataFrame]) -> None:
    """Plot C-phase balance for both reactors side by side with stacked outputs."""
    luke5_data = filter_stable_cycles(all_data["offgas_Luke5"], "offgas_Luke5") if "offgas_Luke5" in all_data else None
    luke6_data = filter_stable_cycles(all_data["offgas_Luke6"], "offgas_Luke6") if "offgas_Luke6" in all_data else None
    
    if luke5_data is None or luke6_data is None or luke5_data.empty or luke6_data.empty:
        return

    # Get averages
    l5_acetate_in = luke5_data["C_acetate_mCmol"].mean()
    l5_acetate_in_std = luke5_data["C_acetate_mCmol"].std()
    l5_pha = luke5_data["PHA_new_mCmol"].mean()
    l5_pha_std = luke5_data["PHA_new_mCmol"].std()
    l5_co2 = luke5_data["CO2_C_mmol"].mean()
    l5_co2_std = luke5_data["CO2_C_mmol"].std()
    l5_biomass = luke5_data["Biomass_C_phase_growth_mCmol"].mean()
    l5_biomass_std = luke5_data["Biomass_C_phase_growth_mCmol"].std()
    # Calculate carbon balance closure for each cycle, then take mean and std
    l5_closure_per_cycle = 100 * (luke5_data["PHA_new_mCmol"] + luke5_data["Biomass_C_phase_growth_mCmol"] + luke5_data["CO2_C_mmol"]) / luke5_data["C_acetate_mCmol"]
    l5_closure = l5_closure_per_cycle.mean()
    l5_closure_std = l5_closure_per_cycle.std()
    
    l6_acetate_in = luke6_data["C_acetate_mCmol"].mean()
    l6_acetate_in_std = luke6_data["C_acetate_mCmol"].std()
    l6_pha = luke6_data["PHA_new_mCmol"].mean()
    l6_pha_std = luke6_data["PHA_new_mCmol"].std()
    l6_co2 = luke6_data["CO2_C_mmol"].mean()
    l6_co2_std = luke6_data["CO2_C_mmol"].std()
    l6_biomass = luke6_data["Biomass_C_phase_growth_mCmol"].mean()
    l6_biomass_std = luke6_data["Biomass_C_phase_growth_mCmol"].std()
    # Calculate carbon balance closure for each cycle, then take mean and std
    l6_closure_per_cycle = 100 * (luke6_data["PHA_new_mCmol"] + luke6_data["Biomass_C_phase_growth_mCmol"] + luke6_data["CO2_C_mmol"]) / luke6_data["C_acetate_mCmol"]
    l6_closure = l6_closure_per_cycle.mean()
    l6_closure_std = l6_closure_per_cycle.std()

    fig, ax = plt.subplots(figsize=(5, 7.9))
    bar_width = 0.175  # Half the original width
    x = [0, 0.3, 0.85, 1.15]  # Adjusted for centered bars with tighter spacing
    
    # Luke 5 input bar
    ax.bar(x[0], l5_acetate_in, bar_width, yerr=l5_acetate_in_std, color="#666666", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_acetate_in > 0:
        ax.text(x[0], l5_acetate_in/2, f"{int(round(l5_acetate_in))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    
    # Luke 5 output bars (stacked): PHA at bottom, Biomass middle, CO2 on top
    ax.bar(x[1], l5_pha, bar_width, yerr=l5_pha_std, color="#F18F01", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_pha > 0:
        ax.text(x[1], l5_pha/2, f"{int(round(l5_pha))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[1], l5_biomass, bar_width, bottom=l5_pha, yerr=l5_biomass_std, color="#06A77D", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_biomass > 0:
        ax.text(x[1], l5_pha + l5_biomass/2, f"{int(round(l5_biomass))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[1], l5_co2, bar_width, bottom=l5_pha + l5_biomass, yerr=l5_co2_std, color="#208CB9", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_co2 > 0:
        ax.text(x[1], l5_pha + l5_biomass + l5_co2/2, f"{int(round(l5_co2))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    # Add closure annotation
    closure_text = f"{l5_closure:.0f}±{l5_closure_std:.0f}%"
    if l5_pha + l5_biomass + l5_co2 > 0:
        ax.text(x[1], l5_pha + l5_biomass + l5_co2 + 3, closure_text, ha="center", va="bottom", fontsize=8, fontweight="bold")
    
    # Luke 6 input bar
    ax.bar(x[2], l6_acetate_in, bar_width, yerr=l6_acetate_in_std, color="#666666", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_acetate_in > 0:
        ax.text(x[2], l6_acetate_in/2, f"{int(round(l6_acetate_in))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    
    # Luke 6 output bars (stacked): PHA at bottom, Biomass middle, CO2 on top
    ax.bar(x[3], l6_pha, bar_width, yerr=l6_pha_std, color="#F18F01", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_pha > 0:
        ax.text(x[3], l6_pha/2, f"{int(round(l6_pha))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[3], l6_biomass, bar_width, bottom=l6_pha, yerr=l6_biomass_std, color="#06A77D", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_biomass > 0:
        ax.text(x[3], l6_pha + l6_biomass/2, f"{int(round(l6_biomass))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[3], l6_co2, bar_width, bottom=l6_pha + l6_biomass, yerr=l6_co2_std, color="#208CB9", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_co2 > 0:
        ax.text(x[3], l6_pha + l6_biomass + l6_co2/2, f"{int(round(l6_co2))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    # Add closure annotation
    closure_text = f"{l6_closure:.0f}±{l6_closure_std:.0f}%"
    if l6_pha + l6_biomass + l6_co2 > 0:
        ax.text(x[3], l6_pha + l6_biomass + l6_co2 + 3, closure_text, ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Carbon (mmolC)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["IN", "OUT$_{C}$", "IN", "OUT$_{C}$"], fontsize=11, fontweight="bold")
    
    # Add vertical divider line between reactors
    ax.axvline(x=0.575, color="gray", linestyle="--", linewidth=2, alpha=0.6, zorder=0)
    
    # Add reactor labels
    ax.text(0.02, 0.98, "R2-CN24", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="normal", style="italic", color="black")
    ax.text(0.82, 0.98, "R3-CN40", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="normal", style="italic", color="black")
    ax.set_xlim(-0.25, 1.4)
    ax.set_ylim(0, 200)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.legend(
        handles=[
            Patch(facecolor="#F18F01", alpha=0.8, edgecolor="black", label="PHA"),
            Patch(facecolor="#06A77D", alpha=0.8, edgecolor="black", label="Biomass"),
            Patch(facecolor="#208CB9", alpha=0.8, edgecolor="black", label="$CO_2$"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        fontsize=9,
        frameon=False,
        fancybox=False,
        edgecolor="black",
    )
    
    save_combined_figure(fig, "c_phase_combined.png")


def plot_c_phase_combined_with_feast(all_data: dict[str, pd.DataFrame]) -> None:
    """Plot C-phase balance with feast phase for both reactors side by side."""
    luke5_data = filter_stable_cycles(all_data["offgas_Luke5"], "offgas_Luke5") if "offgas_Luke5" in all_data else None
    luke6_data = filter_stable_cycles(all_data["offgas_Luke6"], "offgas_Luke6") if "offgas_Luke6" in all_data else None
    
    if luke5_data is None or luke6_data is None or luke5_data.empty or luke6_data.empty:
        return

    # Get averages for Luke 5
    l5_acetate_in = luke5_data["C_acetate_mCmol"].mean()
    l5_acetate_in_std = luke5_data["C_acetate_mCmol"].std()
    
    l5_pha_feast = luke5_data["PHA_feast_mCmol"].mean()
    l5_pha_feast_std = luke5_data["PHA_feast_mCmol"].std()
    l5_co2_feast = luke5_data["CO2_int_feast_mmol"].mean()
    l5_co2_feast_std = luke5_data["CO2_int_feast_mmol"].std()
    
    l5_pha_total = luke5_data["PHA_new_mCmol"].mean()
    l5_pha_total_std = luke5_data["PHA_new_mCmol"].std()
    l5_co2_total = luke5_data["CO2_C_mmol"].mean()
    l5_co2_total_std = luke5_data["CO2_C_mmol"].std()
    l5_biomass_total = luke5_data["Biomass_C_phase_growth_mCmol"].mean()
    l5_biomass_total_std = luke5_data["Biomass_C_phase_growth_mCmol"].std()
    # Calculate carbon balance closure: all_out summed / all_IN * 100
    l5_closure_per_cycle = 100 * (luke5_data["PHA_new_mCmol"] + luke5_data["Biomass_C_phase_growth_mCmol"] + luke5_data["CO2_C_mmol"]) / luke5_data["C_acetate_mCmol"]
    l5_closure = l5_closure_per_cycle.mean()
    l5_closure_std = l5_closure_per_cycle.std()
    l5_closure_per_cycle_feast = 100 * (luke5_data["PHA_feast_mCmol"] + luke5_data["CO2_int_feast_mmol"] + luke5_data["Biomass_C_phase_growth_mCmol"]) / luke5_data["C_acetate_mCmol"]
    l5_closure_feast = l5_closure_per_cycle_feast.mean()
    l5_closure_feast_std = l5_closure_per_cycle_feast.std()

    # Get averages for Luke 6
    l6_acetate_in = luke6_data["C_acetate_mCmol"].mean()
    l6_acetate_in_std = luke6_data["C_acetate_mCmol"].std()
    
    l6_pha_feast = luke6_data["PHA_feast_mCmol"].mean()
    l6_pha_feast_std = luke6_data["PHA_feast_mCmol"].std()
    l6_co2_feast = luke6_data["CO2_int_feast_mmol"].mean()
    l6_co2_feast_std = luke6_data["CO2_int_feast_mmol"].std()
    
    l6_pha_total = luke6_data["PHA_new_mCmol"].mean()
    l6_pha_total_std = luke6_data["PHA_new_mCmol"].std()
    l6_co2_total = luke6_data["CO2_C_mmol"].mean()
    l6_co2_total_std = luke6_data["CO2_C_mmol"].std()
    l6_biomass_total = luke6_data["Biomass_C_phase_growth_mCmol"].mean()
    l6_biomass_total_std = luke6_data["Biomass_C_phase_growth_mCmol"].std()
    # Calculate carbon balance closure: all_out summed / all_IN * 100
    l6_closure_per_cycle = 100 * (luke6_data["PHA_new_mCmol"] + luke6_data["Biomass_C_phase_growth_mCmol"] + luke6_data["CO2_C_mmol"]) / luke6_data["C_acetate_mCmol"]
    l6_closure = l6_closure_per_cycle.mean()
    l6_closure_std = l6_closure_per_cycle.std()
    l6_closure_per_cycle_feast = 100 * (luke6_data["PHA_feast_mCmol"] + luke6_data["CO2_int_feast_mmol"] + luke6_data["Biomass_C_phase_growth_mCmol"]) / luke6_data["C_acetate_mCmol"]
    l6_closure_feast = l6_closure_per_cycle_feast.mean()    
    l6_closure_feast_std = l6_closure_per_cycle_feast.std()

    fig, ax = plt.subplots(figsize=(7, 7.9))
    bar_width = 0.12
    x = [0, 0.2, 0.4, 0.85, 1.05, 1.25]  # IN, FEAST, OUT for each reactor (tighter spacing)
    
    # Luke 5 IN
    ax.bar(x[0], l5_acetate_in, bar_width, yerr=l5_acetate_in_std, color="#666666", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_acetate_in > 0:
        ax.text(x[0], l5_acetate_in/2, f"{int(round(l5_acetate_in))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    
    # Luke 5 FEAST (stacked: PHA at bottom, CO2 on top)
    ax.bar(x[1], l5_pha_feast, bar_width, yerr=l5_pha_feast_std, color="#F18F01", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_pha_feast > 0:
        ax.text(x[1], l5_pha_feast/2, f"{int(round(l5_pha_feast))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    ax.bar(x[1], l5_biomass_total, bar_width, bottom=l5_pha_feast, yerr=l5_biomass_total_std, color="#06A77D", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_biomass_total > 0:
        ax.text(x[1], l5_pha_feast + l5_biomass_total/2, f"{int(round(l5_biomass_total))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    ax.bar(x[1], l5_co2_feast, bar_width, bottom=l5_pha_feast + l5_biomass_total, yerr=l5_co2_feast_std, color="#208CB9", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_co2_feast > 0:
        ax.text(x[1], l5_pha_feast + l5_biomass_total + l5_co2_feast/2, f"{int(round(l5_co2_feast))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    # Add closure annotation for FEAST
    closure_text = f"{l5_closure_feast:.0f}±{l5_closure_feast_std:.0f}%"
    if l5_pha_feast + l5_co2_feast > 0:
        ax.text(x[1], l5_pha_feast + l5_co2_feast + 3, closure_text, ha="center", va="bottom", fontsize=7, fontweight="bold")
    
    # Luke 5 OUT (total, stacked: PHA at bottom, Biomass middle, CO2 on top)
    ax.bar(x[2], l5_pha_total, bar_width, yerr=l5_pha_total_std, color="#F18F01", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_pha_total > 0:
        ax.text(x[2], l5_pha_total/2, f"{int(round(l5_pha_total))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    ax.bar(x[2], l5_biomass_total, bar_width, bottom=l5_pha_total, yerr=l5_biomass_total_std, color="#06A77D", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_biomass_total > 0:
        ax.text(x[2], l5_pha_total + l5_biomass_total/2, f"{int(round(l5_biomass_total))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    ax.bar(x[2], l5_co2_total, bar_width, bottom=l5_pha_total + l5_biomass_total, yerr=l5_co2_total_std, color="#208CB9", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_co2_total > 0:
        ax.text(x[2], l5_pha_total + l5_biomass_total + l5_co2_total/2, f"{int(round(l5_co2_total))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    # Add closure annotation
    closure_text = f"{l5_closure:.0f}±{l5_closure_std:.0f}%"
    if l5_pha_total + l5_biomass_total + l5_co2_total > 0:
        ax.text(x[2], l5_pha_total + l5_biomass_total + l5_co2_total + 3, closure_text, ha="center", va="bottom", fontsize=7, fontweight="bold")
    
    # Luke 6 IN
    ax.bar(x[3], l6_acetate_in, bar_width, yerr=l6_acetate_in_std, color="#666666", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_acetate_in > 0:
        ax.text(x[3], l6_acetate_in/2, f"{int(round(l6_acetate_in))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    
    # Luke 6 FEAST (stacked: PHA at bottom, CO2 on top)
    ax.bar(x[4], l6_pha_feast, bar_width, yerr=l6_pha_feast_std, color="#F18F01", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_pha_feast > 0:
        ax.text(x[4], l6_pha_feast/2, f"{int(round(l6_pha_feast))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    ax.bar(x[4], l6_biomass_total, bar_width, bottom=l6_pha_feast, yerr=l6_biomass_total_std, color="#06A77D", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_biomass_total > 0:
        ax.text(x[4], l6_pha_feast + l6_biomass_total/2, f"{int(round(l6_biomass_total))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    ax.bar(x[4], l6_co2_feast, bar_width, bottom=l6_pha_feast + l6_biomass_total, yerr=l6_co2_feast_std, color="#208CB9", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_co2_feast > 0:
        ax.text(x[4], l6_pha_feast + l6_biomass_total + l6_co2_feast/2, f"{int(round(l6_co2_feast))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    # Add closure annotation for FEAST
    closure_text = f"{l6_closure_feast:.0f}±{l6_closure_feast_std:.0f}%"
    if l6_pha_feast + l6_co2_feast > 0:
        ax.text(x[4], l6_pha_feast + l6_co2_feast + 3, closure_text, ha="center", va="bottom", fontsize=7, fontweight="bold")
    
    # Luke 6 OUT (total, stacked: PHA at bottom, Biomass middle, CO2 on top)
    ax.bar(x[5], l6_pha_total, bar_width, yerr=l6_pha_total_std, color="#F18F01", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_pha_total > 0:
        ax.text(x[5], l6_pha_total/2, f"{int(round(l6_pha_total))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    ax.bar(x[5], l6_biomass_total, bar_width, bottom=l6_pha_total, yerr=l6_biomass_total_std, color="#06A77D", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_biomass_total > 0:
        ax.text(x[5], l6_pha_total + l6_biomass_total/2, f"{int(round(l6_biomass_total))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    ax.bar(x[5], l6_co2_total, bar_width, bottom=l6_pha_total + l6_biomass_total, yerr=l6_co2_total_std, color="#208CB9", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_co2_total > 0:
        ax.text(x[5], l6_pha_total + l6_biomass_total + l6_co2_total/2, f"{int(round(l6_co2_total))}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")
    # Add closure annotation
    closure_text = f"{l6_closure:.0f}±{l6_closure_std:.0f}%"
    if l6_pha_total + l6_biomass_total + l6_co2_total > 0:
        ax.text(x[5], l6_pha_total + l6_biomass_total + l6_co2_total + 3, closure_text, ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("Carbon (mmolC)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["IN", "OUT$_{feast}$", "OUT$_{C}$", "IN", "OUT$_{feast}$", "OUT$_{C}$"], fontsize=10, fontweight="bold")
    
    # Add vertical divider line between reactors
    ax.axvline(x=0.625, color="gray", linestyle="--", linewidth=2, alpha=0.6, zorder=0)
    
    # Add reactor labels
    ax.text(0.02, 0.98, "R2-CN24", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="normal", style="italic", color="black")
    ax.text(0.87, 0.98, "R3-CN40", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="normal", style="italic", color="black")
    ax.set_xlim(-0.15, 1.4)
    ax.set_ylim(0, 200)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.legend(
        handles=[
            Patch(facecolor="#F18F01", alpha=0.8, edgecolor="black", label="PHA"),
            Patch(facecolor="#06A77D", alpha=0.8, edgecolor="black", label="Biomass"),
            Patch(facecolor="#208CB9", alpha=0.8, edgecolor="black", label="$CO_2$"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        fontsize=9,
        frameon=False,
        fancybox=False,
        edgecolor="black",
    )
    
    save_combined_figure(fig, "c_phase_combined_with_feast.png")


def plot_n_phase_consumed_combined(all_data: dict[str, pd.DataFrame]) -> None:
    """Plot N-phase PHA consumed balance for both reactors side by side with stacked outputs."""
    luke5_data = filter_stable_cycles(all_data["offgas_Luke5"], "offgas_Luke5") if "offgas_Luke5" in all_data else None
    luke6_data = filter_stable_cycles(all_data["offgas_Luke6"], "offgas_Luke6") if "offgas_Luke6" in all_data else None
    
    if luke5_data is None or luke6_data is None or luke5_data.empty or luke6_data.empty:
        return

    # Get averages
    l5_pha_consumed = luke5_data["PHA_consumed_mCmol"].mean()
    l5_pha_consumed_std = luke5_data["PHA_consumed_mCmol"].std()
    l5_biomass = series_for(luke5_data, "Biomass_formed_mCmol", "Biomass_N_phase_growth_mCmol").mean()
    l5_biomass_std = series_for(luke5_data, "Biomass_formed_mCmol", "Biomass_N_phase_growth_mCmol").std()
    l5_co2 = luke5_data["CO2_N_mmol"].mean()
    l5_co2_std = luke5_data["CO2_N_mmol"].std()
    # Calculate carbon balance closure: all_out summed / all_IN * 100
    l5_closure_per_cycle = 100 * (luke5_data["Biomass_N_phase_growth_mCmol"] + luke5_data["CO2_N_mmol"]) / luke5_data["PHA_consumed_mCmol"]
    l5_closure = l5_closure_per_cycle.mean()
    l5_closure_std = l5_closure_per_cycle.std()
    
    l6_pha_consumed = luke6_data["PHA_consumed_mCmol"].mean()
    l6_pha_consumed_std = luke6_data["PHA_consumed_mCmol"].std()
    l6_biomass = series_for(luke6_data, "Biomass_formed_mCmol", "Biomass_N_phase_growth_mCmol").mean()
    l6_biomass_std = series_for(luke6_data, "Biomass_formed_mCmol", "Biomass_N_phase_growth_mCmol").std()
    l6_co2 = luke6_data["CO2_N_mmol"].mean()
    l6_co2_std = luke6_data["CO2_N_mmol"].std()
    # Calculate carbon balance closure: all_out summed / all_IN * 100
    l6_closure_per_cycle = 100 * (luke6_data["Biomass_N_phase_growth_mCmol"] + luke6_data["CO2_N_mmol"]) / luke6_data["PHA_consumed_mCmol"]
    l6_closure = l6_closure_per_cycle.mean()
    l6_closure_std = l6_closure_per_cycle.std()

    fig, ax = plt.subplots(figsize=(5, 7.9))
    bar_width = 0.175  # Half the original width
    x = [0, 0.3, 0.85, 1.15]  # Adjusted for centered bars with tighter spacing
    
    # Luke 5 input bar
    ax.bar(x[0], l5_pha_consumed, bar_width, yerr=l5_pha_consumed_std, color="#666666", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_pha_consumed > 0:
        ax.text(x[0], l5_pha_consumed/2, f"{int(round(l5_pha_consumed))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    
    # Luke 5 output bars (stacked: Biomass at bottom, CO2 on top)
    ax.bar(x[1], l5_biomass, bar_width, yerr=l5_biomass_std, color="#06A77D", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_biomass > 0:
        ax.text(x[1], l5_biomass/2, f"{int(round(l5_biomass))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[1], l5_co2, bar_width, bottom=l5_biomass, yerr=l5_co2_std, color="#208CB9", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_co2 > 0:
        ax.text(x[1], l5_biomass + l5_co2/2, f"{int(round(l5_co2))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    # Add closure annotation
    closure_text = f"{l5_closure:.0f}±{l5_closure_std:.0f}%"
    if l5_biomass + l5_co2 > 0:
        ax.text(x[1], l5_biomass + l5_co2 + 3, closure_text, ha="center", va="bottom", fontsize=8, fontweight="bold")
    
    # Luke 6 input bar
    ax.bar(x[2], l6_pha_consumed, bar_width, yerr=l6_pha_consumed_std, color="#666666", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_pha_consumed > 0:
        ax.text(x[2], l6_pha_consumed/2, f"{int(round(l6_pha_consumed))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    
    # Luke 6 output bars (stacked: Biomass at bottom, CO2 on top)
    ax.bar(x[3], l6_biomass, bar_width, yerr=l6_biomass_std, color="#06A77D", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_biomass > 0:
        ax.text(x[3], l6_biomass/2, f"{int(round(l6_biomass))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[3], l6_co2, bar_width, bottom=l6_biomass, yerr=l6_co2_std, color="#208CB9", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_co2 > 0:
        ax.text(x[3], l6_biomass + l6_co2/2, f"{int(round(l6_co2))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    # Add closure annotation
    closure_text = f"{l6_closure:.0f}±{l6_closure_std:.0f}%"
    if l6_biomass + l6_co2 > 0:
        ax.text(x[3], l6_biomass + l6_co2 + 3, closure_text, ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Carbon (mmolC)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["IN", "OUT$_{N}$", "IN", "OUT$_{N}$"], fontsize=11, fontweight="bold")
    
    # Add vertical divider line between reactors
    ax.axvline(x=0.575, color="gray", linestyle="--", linewidth=2, alpha=0.6, zorder=0)
    
    # Add reactor labels
    ax.text(0.02, 0.98, "R2-CN24", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="normal", style="italic", color="black")
    ax.text(0.82, 0.98, "R3-CN40", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="normal", style="italic", color="black")
    ax.set_xlim(-0.25, 1.4)
    ax.set_ylim(0, 200)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.legend(
        handles=[
            Patch(facecolor="#06A77D", alpha=0.8, edgecolor="black", label="Biomass"),
            Patch(facecolor="#208CB9", alpha=0.8, edgecolor="black", label="$CO_2$"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        fontsize=9,
        frameon=False,
        fancybox=False,
        edgecolor="black",
    )
    
    save_combined_figure(fig, "n_phase_consumed_combined.png")


def plot_total_combined(all_data: dict[str, pd.DataFrame]) -> None:
    """Plot total carbon balance for both reactors side by side with stacked outputs."""
    luke5_data = filter_stable_cycles(all_data["offgas_Luke5"], "offgas_Luke5") if "offgas_Luke5" in all_data else None
    luke6_data = filter_stable_cycles(all_data["offgas_Luke6"], "offgas_Luke6") if "offgas_Luke6" in all_data else None
    
    if luke5_data is None or luke6_data is None or luke5_data.empty or luke6_data.empty:
        return

    # Get averages
    l5_acetate_in = luke5_data["C_acetate_mCmol"].mean()
    l5_acetate_in_std = luke5_data["C_acetate_mCmol"].std()
    l5_pha = luke5_data["PHA_removed_mCmol"].mean()
    l5_pha_std = luke5_data["PHA_removed_mCmol"].std()
    l5_biomass = luke5_data["Biomass_total_grown_mCmol"].mean()
    l5_biomass_std = luke5_data["Biomass_total_grown_mCmol"].std()
    l5_co2 = (luke5_data["CO2_C_mmol"] + luke5_data["CO2_N_mmol"]).mean()
    l5_co2_std = (luke5_data["CO2_C_mmol"] + luke5_data["CO2_N_mmol"]).std()
    # Calculate carbon balance closure: all_out summed / all_IN * 100
    l5_closure_per_cycle = 100 * (luke5_data["PHA_removed_mCmol"] + luke5_data["Biomass_total_grown_mCmol"] + luke5_data["CO2_C_mmol"] + luke5_data["CO2_N_mmol"]) / luke5_data["C_acetate_mCmol"]
    l5_closure = l5_closure_per_cycle.mean()
    l5_closure_std = l5_closure_per_cycle.std()
    
    l6_acetate_in = luke6_data["C_acetate_mCmol"].mean()
    l6_acetate_in_std = luke6_data["C_acetate_mCmol"].std()
    l6_pha = luke6_data["PHA_removed_mCmol"].mean()
    l6_pha_std = luke6_data["PHA_removed_mCmol"].std()
    l6_biomass = luke6_data["Biomass_total_grown_mCmol"].mean()
    l6_biomass_std = luke6_data["Biomass_total_grown_mCmol"].std()
    l6_co2 = (luke6_data["CO2_C_mmol"] + luke6_data["CO2_N_mmol"]).mean()
    l6_co2_std = (luke6_data["CO2_C_mmol"] + luke6_data["CO2_N_mmol"]).std()
    # Calculate carbon balance closure: all_out summed / all_IN * 100
    l6_closure_per_cycle = 100 * (luke6_data["PHA_removed_mCmol"] + luke6_data["Biomass_total_grown_mCmol"] + luke6_data["CO2_C_mmol"] + luke6_data["CO2_N_mmol"]) / luke6_data["C_acetate_mCmol"]
    l6_closure = l6_closure_per_cycle.mean()
    l6_closure_std = l6_closure_per_cycle.std()

    fig, ax = plt.subplots(figsize=(5, 7.9))
    bar_width = 0.175  # Half the original width
    x = [0, 0.3, 0.85, 1.15]  # Adjusted for centered bars with tighter spacing
    
    # Luke 5 input bar
    ax.bar(x[0], l5_acetate_in, bar_width, yerr=l5_acetate_in_std, color="#666666", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_acetate_in > 0:
        ax.text(x[0], l5_acetate_in/2, f"{int(round(l5_acetate_in))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    
    # Luke 5 output bars (stacked: PHA at bottom, Biomass middle, CO2 on top)
    ax.bar(x[1], l5_pha, bar_width, yerr=l5_pha_std, color="#F18F01", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_pha > 0:
        ax.text(x[1], l5_pha/2, f"{int(round(l5_pha))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[1], l5_biomass, bar_width, bottom=l5_pha, yerr=l5_biomass_std, color="#06A77D", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_biomass > 0:
        ax.text(x[1], l5_pha + l5_biomass/2, f"{int(round(l5_biomass))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[1], l5_co2, bar_width, bottom=l5_pha+l5_biomass, yerr=l5_co2_std, color="#208CB9", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_co2 > 0:
        ax.text(x[1], l5_pha + l5_biomass + l5_co2/2, f"{int(round(l5_co2))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    # Add closure annotation
    closure_text = f"{l5_closure:.0f}±{l5_closure_std:.0f}%"
    if l5_pha + l5_biomass + l5_co2 > 0:
        ax.text(x[1], l5_pha + l5_biomass + l5_co2 + 3, closure_text, ha="center", va="bottom", fontsize=8, fontweight="bold")
    
    # Luke 6 input bar
    ax.bar(x[2], l6_acetate_in, bar_width, yerr=l6_acetate_in_std, color="#666666", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_acetate_in > 0:
        ax.text(x[2], l6_acetate_in/2, f"{int(round(l6_acetate_in))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    
    # Luke 6 output bars (stacked: PHA at bottom, Biomass middle, CO2 on top)
    ax.bar(x[3], l6_pha, bar_width, yerr=l6_pha_std, color="#F18F01", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_pha > 0:
        ax.text(x[3], l6_pha/2, f"{int(round(l6_pha))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[3], l6_biomass, bar_width, bottom=l6_pha, yerr=l6_biomass_std, color="#06A77D", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_biomass > 0:
        ax.text(x[3], l6_pha + l6_biomass/2, f"{int(round(l6_biomass))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[3], l6_co2, bar_width, bottom=l6_pha+l6_biomass, yerr=l6_co2_std, color="#208CB9", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_co2 > 0:
        ax.text(x[3], l6_pha + l6_biomass + l6_co2/2, f"{int(round(l6_co2))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    # Add closure annotation
    closure_text = f"{l6_closure:.0f}±{l6_closure_std:.0f}%"
    if l6_pha + l6_biomass + l6_co2 > 0:
        ax.text(x[3], l6_pha + l6_biomass + l6_co2 + 3, closure_text, ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Carbon (mmolC)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["IN", "OUT", "IN", "OUT"], fontsize=11, fontweight="bold")
    
    # Add vertical divider line between reactors
    ax.axvline(x=0.575, color="gray", linestyle="--", linewidth=2, alpha=0.6, zorder=0)
    ax.text(0.55, 0.98, "R3-CN40", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="normal", style="italic", color="black")
    ax.set_xlim(-0.25, 1.4)
    ax.set_ylim(0, 200)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.legend(
        handles=[
            Patch(facecolor="#F18F01", alpha=0.8, edgecolor="black", label="PHA"),
            Patch(facecolor="#06A77D", alpha=0.8, edgecolor="black", label="Biomass"),
            Patch(facecolor="#208CB9", alpha=0.8, edgecolor="black", label="$CO_2$"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        fontsize=9,
        frameon=False,
        fancybox=False,
        edgecolor="black",
    )
    
    save_combined_figure(fig, "total_combined.png")


def plot_n_phase_combined(all_data: dict[str, pd.DataFrame]) -> None:
    """Plot N-phase carbon balance for both reactors side by side with stacked outputs."""
    luke5_data = filter_stable_cycles(all_data["offgas_Luke5"], "offgas_Luke5") if "offgas_Luke5" in all_data else None
    luke6_data = filter_stable_cycles(all_data["offgas_Luke6"], "offgas_Luke6") if "offgas_Luke6" in all_data else None
    
    if luke5_data is None or luke6_data is None or luke5_data.empty or luke6_data.empty:
        return

    # Get averages
    l5_pha_available = luke5_data["PHA_available_N_mCmol"].mean()
    l5_pha_leftover = luke5_data["PHA_leftover"].mean()
    l5_biomass = luke5_data["Biomass_N_phase_growth_mCmol"].mean()
    l5_co2 = luke5_data["CO2_N_mmol"].mean()
    
    l6_pha_available = luke6_data["PHA_available_N_mCmol"].mean()
    l6_pha_leftover = luke6_data["PHA_leftover"].mean()
    l6_biomass = luke6_data["Biomass_N_phase_growth_mCmol"].mean()
    l6_co2 = luke6_data["CO2_N_mmol"].mean()
    
    # Standard deviations
    l5_pha_available_std = luke5_data["PHA_available_N_mCmol"].std()
    l5_pha_leftover_std = luke5_data["PHA_leftover"].std()
    l5_biomass_std = series_for(luke5_data, "Biomass_formed_mCmol", "Biomass_N_phase_growth_mCmol").std()
    l5_co2_std = luke5_data["CO2_N_mmol"].std()
    
    l6_pha_available_std = luke6_data["PHA_available_N_mCmol"].std()
    l6_pha_leftover_std = luke6_data["PHA_leftover"].std()
    l6_biomass_std = series_for(luke6_data, "Biomass_formed_mCmol", "Biomass_N_phase_growth_mCmol").std()
    l6_co2_std = luke6_data["CO2_N_mmol"].std()
    # Calculate carbon balance closure: all_out summed / all_IN * 100
    l5_closure_per_cycle = 100 * (luke5_data["PHA_leftover"] + luke5_data["Biomass_N_phase_growth_mCmol"] + luke5_data["CO2_N_mmol"]) / luke5_data["PHA_available_N_mCmol"]
    l5_closure = l5_closure_per_cycle.mean()
    l5_closure_std = l5_closure_per_cycle.std()
    l6_closure_per_cycle = 100 * (luke6_data["PHA_leftover"] + luke6_data["Biomass_N_phase_growth_mCmol"] + luke6_data["CO2_N_mmol"]) / luke6_data["PHA_available_N_mCmol"]
    l6_closure = l6_closure_per_cycle.mean()
    l6_closure_std = l6_closure_per_cycle.std()

    fig, ax = plt.subplots(figsize=(5, 7.9))
    bar_width = 0.175  # Half the original width
    x = [0, 0.3, 0.85, 1.15]  # Adjusted for centered bars with tighter spacing
    
    # Luke 5 input bar
    ax.bar(x[0], l5_pha_available, bar_width, yerr=l5_pha_available_std, color="#666666", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_pha_available > 0:
        ax.text(x[0], l5_pha_available/2, f"{int(round(l5_pha_available))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    
    # Luke 5 output bars (stacked: PHA_leftover at bottom, Biomass middle, CO2 on top)
    ax.bar(x[1], l5_pha_leftover, bar_width, yerr=l5_pha_leftover_std, color="#F18F01", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_pha_leftover > 0:
        ax.text(x[1], l5_pha_leftover/2, f"{int(round(l5_pha_leftover))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[1], l5_biomass, bar_width, bottom=l5_pha_leftover, yerr=l5_biomass_std, color="#06A77D", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_biomass > 0:
        ax.text(x[1], l5_pha_leftover + l5_biomass/2, f"{int(round(l5_biomass))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[1], l5_co2, bar_width, bottom=l5_pha_leftover+l5_biomass, yerr=l5_co2_std, color="#208CB9", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l5_co2 > 0:
        ax.text(x[1], l5_pha_leftover + l5_biomass + l5_co2/2, f"{int(round(l5_co2))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    # Add closure annotation
    closure_text = f"{l5_closure:.0f}±{l5_closure_std:.0f}%"
    if l5_pha_leftover + l5_biomass + l5_co2 > 0:
        ax.text(x[1], l5_pha_leftover + l5_biomass + l5_co2 + 3, closure_text, ha="center", va="bottom", fontsize=8, fontweight="bold")
    
    # Luke 6 input bar
    ax.bar(x[2], l6_pha_available, bar_width, yerr=l6_pha_available_std, color="#666666", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_pha_available > 0:
        ax.text(x[2], l6_pha_available/2, f"{int(round(l6_pha_available))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    
    # Luke 6 output bars (stacked: PHA_leftover at bottom, Biomass middle, CO2 on top)
    ax.bar(x[3], l6_pha_leftover, bar_width, yerr=l6_pha_leftover_std, color="#F18F01", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_pha_leftover > 0:
        ax.text(x[3], l6_pha_leftover/2, f"{int(round(l6_pha_leftover))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[3], l6_biomass, bar_width, bottom=l6_pha_leftover, yerr=l6_biomass_std, color="#06A77D", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_biomass > 0:
        ax.text(x[3], l6_pha_leftover + l6_biomass/2, f"{int(round(l6_biomass))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    ax.bar(x[3], l6_co2, bar_width, bottom=l6_pha_leftover+l6_biomass, yerr=l6_co2_std, color="#208CB9", alpha=0.8, edgecolor="black", linewidth=1.5, capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    if l6_co2 > 0:
        ax.text(x[3], l6_pha_leftover + l6_biomass + l6_co2/2, f"{int(round(l6_co2))}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")
    # Add closure annotation
    closure_text = f"{l6_closure:.0f}±{l6_closure_std:.0f}%"
    if l6_pha_leftover + l6_biomass + l6_co2 > 0:
        ax.text(x[3], l6_pha_leftover + l6_biomass + l6_co2 + 3, closure_text, ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Carbon (mmolC)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["IN", "OUT$_{N}$", "IN", "OUT$_{N}$"], fontsize=11, fontweight="bold")
    
    # Add vertical divider line between reactors
    ax.axvline(x=0.575, color="gray", linestyle="--", linewidth=2, alpha=0.6, zorder=0)
    
    # Add reactor labels
    ax.text(0.02, 0.98, "R2-CN24", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="normal", style="italic", color="black")
    ax.text(0.82, 0.98, "R3-CN40", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="normal", style="italic", color="black")
    ax.set_xlim(-0.25, 1.4)
    ax.set_ylim(0, 200)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.legend(
        handles=[
            Patch(facecolor="#F18F01", alpha=0.8, edgecolor="black", label="PHA"),
            Patch(facecolor="#06A77D", alpha=0.8, edgecolor="black", label="Biomass"),
            Patch(facecolor="#208CB9", alpha=0.8, edgecolor="black", label="$CO_2$"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        fontsize=9,
        frameon=False,
        fancybox=False,
        edgecolor="black",
    )
    
    save_combined_figure(fig, "n_phase_combined.png")


def compile_and_export_stable_cycles_to_excel(all_data: dict[str, pd.DataFrame]) -> None:
    
    # Molecular conversion factors (g per mol of carbon)
    MW_BIOMASS_PER_C = 24.6  # CH1.8O0.5N0.2
    MW_PHA_PER_C = 21.5      # C4H6O2: 86 g/mol / 4 C
    MW_ACETATE_PER_C = 30.0  # C2H4O2: 60 g/mol / 2 C
    
    data_summary = {}
    
    for sheet_name in SHEETS:
        if sheet_name not in all_data:
            continue
        
        reactor_name = "Luke 5" if "Luke5" in sheet_name else "Luke 6"
        df = filter_stable_cycles(all_data[sheet_name], sheet_name)
        
        if df.empty:
            continue
        
        # ============================================================================
        # C-PHASE CALCULATIONS
        # ============================================================================
        c_acetate_in = df['C_acetate_mCmol'].mean()
        c_acetate_in_std = df['C_acetate_mCmol'].std()
        
        co2_c_phase = df['CO2_C_mmol'].mean()
        co2_c_phase_std = df['CO2_C_mmol'].std()
        
        pha_new = df['PHA_new_mCmol'].mean()
        pha_new_std = df['PHA_new_mCmol'].std()
        
        # C-phase: Acetate yield on input (always 1.0 or reference)
        acetate_yield_c_phase = 1.0
        acetate_yield_c_phase_std = 0.0
        
        # C-phase: CO2 yield on C-load (mol CO2-C / mol Acetate-C)
        df['co2_yield_c_phase'] = df['CO2_C_mmol'] / df['C_acetate_mCmol']
        co2_yield_c_phase = df['co2_yield_c_phase'].mean()
        co2_yield_c_phase_std = df['co2_yield_c_phase'].std()
        
        # C-phase: PHA yield on C-load (mol PHA-C / mol Acetate-C)
        df['pha_yield_c_phase'] = df['PHA_new_mCmol'] / df['C_acetate_mCmol']
        pha_yield_c_phase = df['pha_yield_c_phase'].mean()
        pha_yield_c_phase_std = df['pha_yield_c_phase'].std()
        
        # PHA content in C-phase
        pha_pct_c = df['PHA_content_end_C_phase_%'].mean()
        pha_pct_c_std = df['PHA_content_end_C_phase_%'].std()
        pha_content_c_g_g = (pha_pct_c / 100.0) * (MW_PHA_PER_C / MW_BIOMASS_PER_C)
        pha_content_c_g_g_std = (pha_pct_c_std / 100.0) * (MW_PHA_PER_C / MW_BIOMASS_PER_C)
        
        # C-phase: Biomass growth
        biomass_c_phase = df["Biomass_C_phase_growth_mCmol"]
        biomass_c_formed = biomass_c_phase.mean()
        biomass_c_formed_std = biomass_c_phase.std()
        
        # C-phase: Biomass yield on C-load (mol Biomass-C / mol Acetate-C)
        df['biomass_yield_c_phase'] = biomass_c_phase / df['C_acetate_mCmol']
        biomass_yield_c_phase = df['biomass_yield_c_phase'].mean()
        biomass_yield_c_phase_std = df['biomass_yield_c_phase'].std()
        
        # C-phase: Convert biomass to mg and calculate yield on substrate
        biomass_c_mg = biomass_c_formed * MW_BIOMASS_PER_C
        biomass_c_mg_std = biomass_c_formed_std * MW_BIOMASS_PER_C
        biomass_c_yield_substrate = (biomass_c_mg) / c_acetate_in
        biomass_c_yield_substrate_std = np.sqrt((biomass_c_mg_std / c_acetate_in)**2 + 
                                                (biomass_c_mg * c_acetate_in_std / c_acetate_in**2)**2)
        
        # ============================================================================
        # N-PHASE CALCULATIONS
        # ============================================================================
        n_load = df['N_load_mNmol'].mean()
        n_load_std = df['N_load_mNmol'].std()
        
        pha_available_n = df['PHA_available_N_mCmol'].mean()
        pha_available_n_std = df['PHA_available_N_mCmol'].std()
        
        pha_consumed = df['PHA_consumed_mCmol'].mean()
        pha_consumed_std = df['PHA_consumed_mCmol'].std()
        
        co2_n_phase = df['CO2_N_mmol'].mean()
        co2_n_phase_std = df['CO2_N_mmol'].std()
        
        biomass_n_phase = df["Biomass_N_phase_growth_mCmol"]
        biomass_n_formed = biomass_n_phase.mean()
        biomass_n_formed_std = biomass_n_phase.std()
        
        pha_leftover = df['PHA_leftover'].mean()
        pha_leftover_std = df['PHA_leftover'].std()
        
        # N-phase: Yield on PHA consumed (mol product-C / mol PHA-C)
        df['co2_yield_n_phase'] = df['CO2_N_mmol'] / df['PHA_consumed_mCmol']
        co2_yield_n_phase = df['co2_yield_n_phase'].mean()
        co2_yield_n_phase_std = df['co2_yield_n_phase'].std()
        
        # N-phase: Biomass yield on PHA consumed (grams of biomass / mmol PHA-C)
        df['biomass_yield_n_phase'] = (biomass_n_phase * MW_BIOMASS_PER_C) / df['PHA_consumed_mCmol']
        biomass_yield_n_phase = df['biomass_yield_n_phase'].mean()
        biomass_yield_n_phase_std = df['biomass_yield_n_phase'].std()
        
        df['pha_leftover_yield_n_phase'] = df['PHA_leftover'] / df['PHA_consumed_mCmol']
        pha_leftover_yield_n_phase = df['pha_leftover_yield_n_phase'].mean()
        pha_leftover_yield_n_phase_std = df['pha_leftover_yield_n_phase'].std()
        
        # PHA content in N-phase
        pha_pct_n = df['PHA_content_end_N_phase_%'].mean()
        pha_pct_n_std = df['PHA_content_end_N_phase_%'].std()
        pha_content_n_g_g = (pha_pct_n / 100.0) * (MW_PHA_PER_C / MW_BIOMASS_PER_C)
        pha_content_n_g_g_std = (pha_pct_n_std / 100.0) * (MW_PHA_PER_C / MW_BIOMASS_PER_C)
        
        # N-phase: Convert biomass to mg
        biomass_n_mg = biomass_n_formed * MW_BIOMASS_PER_C
        biomass_n_mg_std = biomass_n_formed_std * MW_BIOMASS_PER_C
        
        # N-phase: PHA consumed in mg
        pha_consumed_mg = pha_consumed * MW_PHA_PER_C
        pha_consumed_mg_std = pha_consumed_std * MW_PHA_PER_C
        
        # ============================================================================
        # TOTAL CALCULATIONS
        # ============================================================================
        c_total_in = c_acetate_in
        c_total_in_std = c_acetate_in_std
        
        co2_total = (df['CO2_C_mmol'] + df['CO2_N_mmol']).mean()
        co2_total_std = (df['CO2_C_mmol'] + df['CO2_N_mmol']).std()
        
        pha_removed = df['PHA_removed_mCmol'].mean()
        pha_removed_std = df['PHA_removed_mCmol'].std()
        
        biomass_total_grown = df['Biomass_total_grown_mCmol'].mean()
        biomass_total_grown_std = df['Biomass_total_grown_mCmol'].std()
        
        # Total: Yield on C-load (mol product-C / mol Acetate-C)
        df['co2_yield_total'] = (df['CO2_C_mmol'] + df['CO2_N_mmol']) / df['C_acetate_mCmol']
        co2_yield_total = df['co2_yield_total'].mean()
        co2_yield_total_std = df['co2_yield_total'].std()
        
        df['pha_yield_total'] = df['PHA_removed_mCmol'] / df['C_acetate_mCmol']
        pha_yield_total = df['pha_yield_total'].mean()
        pha_yield_total_std = df['pha_yield_total'].std()
        
        df['biomass_yield_total'] = df['Biomass_total_grown_mCmol'] / df['C_acetate_mCmol']
        biomass_yield_total = df['biomass_yield_total'].mean()
        biomass_yield_total_std = df['biomass_yield_total'].std()
        
        # Biomass yield on substrate (gX/molC)
        # biomass_total_grown is in mmolC, need to convert: (mg) / (molC)
        # biomass_mg = biomass_total_grown * MW_BIOMASS_PER_C
        # yield = (biomass_mg/1000) / (c_in/1000) = biomass_mg / c_in = (biomass_mmolC * 24.6) / c_mmolC
        biomass_yield_substrate = (biomass_total_grown * MW_BIOMASS_PER_C) / c_acetate_in
        biomass_yield_substrate_std = np.sqrt((biomass_total_grown_std * MW_BIOMASS_PER_C / c_acetate_in)**2 + 
                                             (biomass_total_grown * MW_BIOMASS_PER_C * c_acetate_in_std / c_acetate_in**2)**2)
        
        # Convert biomass to mg for reporting
        biomass_total_grown_mg = biomass_total_grown * MW_BIOMASS_PER_C
        biomass_total_grown_mg_std = biomass_total_grown_std * MW_BIOMASS_PER_C
        
        # PHA content in Total (same as C-phase, since total PHA removed is from C-phase)
        pha_content_total_g_g = pha_content_c_g_g
        pha_content_total_g_g_std = pha_content_c_g_g_std
        
        # Store in dictionary by reactor
        data_summary[reactor_name] = {
            # C-phase
            'c_acetate_in': (c_acetate_in, c_acetate_in_std),
            'c_acetate_yield': (acetate_yield_c_phase, acetate_yield_c_phase_std),
            'c_co2': (co2_c_phase, co2_c_phase_std),
            'c_co2_yield': (co2_yield_c_phase, co2_yield_c_phase_std),
            'c_pha': (pha_new, pha_new_std),
            'c_pha_yield': (pha_yield_c_phase, pha_yield_c_phase_std),
            'c_pha_content': (pha_content_c_g_g, pha_content_c_g_g_std),
            'c_biomass': (biomass_c_formed, biomass_c_formed_std),
            'c_biomass_yield': (biomass_yield_c_phase, biomass_yield_c_phase_std),
            'c_biomass_mg': (biomass_c_mg, biomass_c_mg_std),
            'c_biomass_yield_substrate': (biomass_c_yield_substrate, biomass_c_yield_substrate_std),
            # N-phase
            'n_load': (n_load, n_load_std),
            'n_pha_available': (pha_available_n, pha_available_n_std),
            'n_pha_available_yield': (1.0, 0.0),
            'n_pha_consumed': (pha_consumed, pha_consumed_std),
            'n_pha_consumed_mg': (pha_consumed_mg, pha_consumed_mg_std),
            'n_co2': (co2_n_phase, co2_n_phase_std),
            'n_co2_yield': (co2_yield_n_phase, co2_yield_n_phase_std),
            'n_biomass': (biomass_n_formed, biomass_n_formed_std),
            'n_biomass_yield': (biomass_yield_n_phase, biomass_yield_n_phase_std),
            'n_biomass_mg': (biomass_n_mg, biomass_n_mg_std),
            'n_biomass_content': (pha_content_n_g_g, pha_content_n_g_g_std),
            'n_pha_leftover': (pha_leftover, pha_leftover_std),
            'n_pha_leftover_yield': (pha_leftover_yield_n_phase, pha_leftover_yield_n_phase_std),
            'n_pha_leftover_content': (pha_content_n_g_g, pha_content_n_g_g_std),
            # Total
            't_acetate_in': (c_total_in, c_total_in_std),
            't_acetate_yield': (1.0, 0.0),
            't_co2': (co2_total, co2_total_std),
            't_co2_yield': (co2_yield_total, co2_yield_total_std),
            't_pha_removed': (pha_removed, pha_removed_std),
            't_pha_removed_yield': (pha_yield_total, pha_yield_total_std),
            't_pha_removed_content': (pha_content_total_g_g, pha_content_total_g_g_std),
            't_biomass': (biomass_total_grown, biomass_total_grown_std),
            't_biomass_yield': (biomass_yield_total, biomass_yield_total_std),
            't_biomass_mg': (biomass_total_grown_mg, biomass_total_grown_mg_std),
            't_biomass_yield_substrate': (biomass_yield_substrate, biomass_yield_substrate_std),
        }
    
    # Create C-Phase dataframe
    c_phase_data = {
        'Compound': ['Acetate (IN)', 'CO2 (OUT)', 'PHA (OUT)', 'Biomass (OUT)'],
        'Absolute_L5_mmolC': [
            f"{data_summary['Luke 5']['c_acetate_in'][0]:.1f} ± {data_summary['Luke 5']['c_acetate_in'][1]:.1f}",
            f"{data_summary['Luke 5']['c_co2'][0]:.1f} ± {data_summary['Luke 5']['c_co2'][1]:.1f}",
            f"{data_summary['Luke 5']['c_pha'][0]:.1f} ± {data_summary['Luke 5']['c_pha'][1]:.1f}",
            f"{data_summary['Luke 5']['c_biomass_mg'][0]:.1f} ± {data_summary['Luke 5']['c_biomass_mg'][1]:.1f} mg",
        ],
        'Yield_L5': [
            f"{data_summary['Luke 5']['c_acetate_yield'][0]:.3f} ± {data_summary['Luke 5']['c_acetate_yield'][1]:.3f}",
            f"{data_summary['Luke 5']['c_co2_yield'][0]:.3f} ± {data_summary['Luke 5']['c_co2_yield'][1]:.3f}",
            f"{data_summary['Luke 5']['c_pha_yield'][0]:.3f} ± {data_summary['Luke 5']['c_pha_yield'][1]:.3f}",
            f"{data_summary['Luke 5']['c_biomass_yield_substrate'][0]:.3f} ± {data_summary['Luke 5']['c_biomass_yield_substrate'][1]:.3f} gX/molC",
        ],
        'PHA%_L5_gPHA_gX': [
            '—',
            '—',
            f"{data_summary['Luke 5']['c_pha_content'][0]:.3f} ± {data_summary['Luke 5']['c_pha_content'][1]:.3f}",
            '—',
        ],
        'Absolute_L6_mmolC': [
            f"{data_summary['Luke 6']['c_acetate_in'][0]:.1f} ± {data_summary['Luke 6']['c_acetate_in'][1]:.1f}",
            f"{data_summary['Luke 6']['c_co2'][0]:.1f} ± {data_summary['Luke 6']['c_co2'][1]:.1f}",
            f"{data_summary['Luke 6']['c_pha'][0]:.1f} ± {data_summary['Luke 6']['c_pha'][1]:.1f}",
            f"{data_summary['Luke 6']['c_biomass_mg'][0]:.1f} ± {data_summary['Luke 6']['c_biomass_mg'][1]:.1f} mg",
        ],
        'Yield_L6': [
            f"{data_summary['Luke 6']['c_acetate_yield'][0]:.3f} ± {data_summary['Luke 6']['c_acetate_yield'][1]:.3f}",
            f"{data_summary['Luke 6']['c_co2_yield'][0]:.3f} ± {data_summary['Luke 6']['c_co2_yield'][1]:.3f}",
            f"{data_summary['Luke 6']['c_pha_yield'][0]:.3f} ± {data_summary['Luke 6']['c_pha_yield'][1]:.3f}",
            f"{data_summary['Luke 6']['c_biomass_yield_substrate'][0]:.3f} ± {data_summary['Luke 6']['c_biomass_yield_substrate'][1]:.3f} gX/molC",
        ],
        'PHA%_L6_gPHA_gX': [
            '—',
            '—',
            f"{data_summary['Luke 6']['c_pha_content'][0]:.3f} ± {data_summary['Luke 6']['c_pha_content'][1]:.3f}",
            '—',
        ],
    }
    c_phase_df = pd.DataFrame(c_phase_data)
    
    # Create N-Phase dataframe
    n_phase_data = {
        'Compound': ['N-Load (IN)', 'PHA (IN)', 'PHA Consumed', 'CO2 (OUT)', 'Biomass (OUT)', 'PHA Leftover'],
        'Absolute_L5_mmolC': [
            f"{data_summary['Luke 5']['n_load'][0]:.1f} ± {data_summary['Luke 5']['n_load'][1]:.1f}",
            f"{data_summary['Luke 5']['n_pha_available'][0]:.1f} ± {data_summary['Luke 5']['n_pha_available'][1]:.1f}",
            f"{data_summary['Luke 5']['n_pha_consumed'][0]:.1f} ± {data_summary['Luke 5']['n_pha_consumed'][1]:.1f}",
            f"{data_summary['Luke 5']['n_co2'][0]:.1f} ± {data_summary['Luke 5']['n_co2'][1]:.1f}",
            f"{data_summary['Luke 5']['n_biomass_mg'][0]:.1f} ± {data_summary['Luke 5']['n_biomass_mg'][1]:.1f} mg",
            f"{data_summary['Luke 5']['n_pha_leftover'][0]:.1f} ± {data_summary['Luke 5']['n_pha_leftover'][1]:.1f}",
        ],
        'Yield_L5': [
            '—',
            f"{data_summary['Luke 5']['n_pha_available_yield'][0]:.3f} ± {data_summary['Luke 5']['n_pha_available_yield'][1]:.3f}",
            '—',
            f"{data_summary['Luke 5']['n_co2_yield'][0]:.3f} ± {data_summary['Luke 5']['n_co2_yield'][1]:.3f}",
            f"{data_summary['Luke 5']['n_biomass_yield'][0]:.3f} ± {data_summary['Luke 5']['n_biomass_yield'][1]:.3f} gX/molC",
            f"{data_summary['Luke 5']['n_pha_leftover_yield'][0]:.3f} ± {data_summary['Luke 5']['n_pha_leftover_yield'][1]:.3f}",
        ],
        'PHA%_L5_gPHA_gX': [
            '—',
            '—',
            '—',
            '—',
            f"{data_summary['Luke 5']['n_biomass_content'][0]:.3f} ± {data_summary['Luke 5']['n_biomass_content'][1]:.3f}",
            f"{data_summary['Luke 5']['n_pha_leftover_content'][0]:.3f} ± {data_summary['Luke 5']['n_pha_leftover_content'][1]:.3f}",
        ],
        'Absolute_L6_mmolC': [
            f"{data_summary['Luke 6']['n_load'][0]:.1f} ± {data_summary['Luke 6']['n_load'][1]:.1f}",
            f"{data_summary['Luke 6']['n_pha_available'][0]:.1f} ± {data_summary['Luke 6']['n_pha_available'][1]:.1f}",
            f"{data_summary['Luke 6']['n_pha_consumed'][0]:.1f} ± {data_summary['Luke 6']['n_pha_consumed'][1]:.1f}",
            f"{data_summary['Luke 6']['n_co2'][0]:.1f} ± {data_summary['Luke 6']['n_co2'][1]:.1f}",
            f"{data_summary['Luke 6']['n_biomass_mg'][0]:.1f} ± {data_summary['Luke 6']['n_biomass_mg'][1]:.1f} mg",
            f"{data_summary['Luke 6']['n_pha_leftover'][0]:.1f} ± {data_summary['Luke 6']['n_pha_leftover'][1]:.1f}",
        ],
        'Yield_L6': [
            '—',
            f"{data_summary['Luke 6']['n_pha_available_yield'][0]:.3f} ± {data_summary['Luke 6']['n_pha_available_yield'][1]:.3f}",
            '—',
            f"{data_summary['Luke 6']['n_co2_yield'][0]:.3f} ± {data_summary['Luke 6']['n_co2_yield'][1]:.3f}",
            f"{data_summary['Luke 6']['n_biomass_yield'][0]:.3f} ± {data_summary['Luke 6']['n_biomass_yield'][1]:.3f} gX/molC",
            f"{data_summary['Luke 6']['n_pha_leftover_yield'][0]:.3f} ± {data_summary['Luke 6']['n_pha_leftover_yield'][1]:.3f}",
        ],
        'PHA%_L6_gPHA_gX': [
            '—',
            '—',
            '—',
            '—',
            f"{data_summary['Luke 6']['n_biomass_content'][0]:.3f} ± {data_summary['Luke 6']['n_biomass_content'][1]:.3f}",
            f"{data_summary['Luke 6']['n_pha_leftover_content'][0]:.3f} ± {data_summary['Luke 6']['n_pha_leftover_content'][1]:.3f}",
        ],
    }
    n_phase_df = pd.DataFrame(n_phase_data)
    
    # Create Total dataframe
    total_data = {
        'Compound': ['Acetate (IN)', 'CO2 (OUT)', 'PHA (OUT)', 'Biomass (OUT)'],
        'Absolute_L5': [
            f"{data_summary['Luke 5']['t_acetate_in'][0]:.1f} ± {data_summary['Luke 5']['t_acetate_in'][1]:.1f}",
            f"{data_summary['Luke 5']['t_co2'][0]:.1f} ± {data_summary['Luke 5']['t_co2'][1]:.1f}",
            f"{data_summary['Luke 5']['t_pha_removed'][0]:.1f} ± {data_summary['Luke 5']['t_pha_removed'][1]:.1f}",
            f"{data_summary['Luke 5']['t_biomass_mg'][0]:.1f} ± {data_summary['Luke 5']['t_biomass_mg'][1]:.1f} mg",
        ],
        'Yield_L5': [
            f"{data_summary['Luke 5']['t_acetate_yield'][0]:.3f} ± {data_summary['Luke 5']['t_acetate_yield'][1]:.3f}",
            f"{data_summary['Luke 5']['t_co2_yield'][0]:.3f} ± {data_summary['Luke 5']['t_co2_yield'][1]:.3f}",
            f"{data_summary['Luke 5']['t_pha_removed_yield'][0]:.3f} ± {data_summary['Luke 5']['t_pha_removed_yield'][1]:.3f}",
            f"{data_summary['Luke 5']['t_biomass_yield_substrate'][0]:.3f} ± {data_summary['Luke 5']['t_biomass_yield_substrate'][1]:.3f} gX/molC",
        ],
        'PHA%_L5_gPHA_gX': [
            '—',
            '—',
            f"{data_summary['Luke 5']['t_pha_removed_content'][0]:.3f} ± {data_summary['Luke 5']['t_pha_removed_content'][1]:.3f}",
            '—',
        ],
        'Absolute_L6': [
            f"{data_summary['Luke 6']['t_acetate_in'][0]:.1f} ± {data_summary['Luke 6']['t_acetate_in'][1]:.1f}",
            f"{data_summary['Luke 6']['t_co2'][0]:.1f} ± {data_summary['Luke 6']['t_co2'][1]:.1f}",
            f"{data_summary['Luke 6']['t_pha_removed'][0]:.1f} ± {data_summary['Luke 6']['t_pha_removed'][1]:.1f}",
            f"{data_summary['Luke 6']['t_biomass_mg'][0]:.1f} ± {data_summary['Luke 6']['t_biomass_mg'][1]:.1f} mg",
        ],
        'Yield_L6': [
            f"{data_summary['Luke 6']['t_acetate_yield'][0]:.3f} ± {data_summary['Luke 6']['t_acetate_yield'][1]:.3f}",
            f"{data_summary['Luke 6']['t_co2_yield'][0]:.3f} ± {data_summary['Luke 6']['t_co2_yield'][1]:.3f}",
            f"{data_summary['Luke 6']['t_pha_removed_yield'][0]:.3f} ± {data_summary['Luke 6']['t_pha_removed_yield'][1]:.3f}",
            f"{data_summary['Luke 6']['t_biomass_yield_substrate'][0]:.3f} ± {data_summary['Luke 6']['t_biomass_yield_substrate'][1]:.3f} gX/molC",
        ],
        'PHA%_L6_gPHA_gX': [
            '—',
            '—',
            f"{data_summary['Luke 6']['t_pha_removed_content'][0]:.3f} ± {data_summary['Luke 6']['t_pha_removed_content'][1]:.3f}",
            '—',
        ],
    }
    total_df = pd.DataFrame(total_data)
    
    # Export to Excel with 3 sheets
    excel_path = OUTPUT_DIR / "stable_cycles_summary.xlsx"
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        c_phase_df.to_excel(writer, sheet_name='C-Phase', index=False)
        n_phase_df.to_excel(writer, sheet_name='N-Phase', index=False)
        total_df.to_excel(writer, sheet_name='Total', index=False)

def main() -> None:
    ensure_output_dir()
    all_data = load_workbook()
    # Individual reactor plots disabled - using combined plots only
    # plot_c_phase(all_data)
    # plot_c_phase_with_feast(all_data)
    # plot_n_phase_consumed(all_data)
    # plot_n_phase(all_data)
    # plot_total_prediction(all_data)
    
    # Combined plots
    plot_c_phase_combined(all_data)
    plot_c_phase_combined_with_feast(all_data)
    plot_n_phase_consumed_combined(all_data)
    plot_n_phase_combined(all_data)
    plot_total_combined(all_data)
    
    # Generate and save summary to Excel (single function, single calculation)
    compile_and_export_stable_cycles_to_excel(all_data)


if __name__ == "__main__":
    main()
