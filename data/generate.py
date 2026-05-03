"""
Test Data Generator for Load Shedding Optimizer - Flow 2
=========================================================
Generates CSV files that Flow 2's parser accepts.

FORMAT 1 - 2 columns (single area):
    Datetime,MW
    2023-01-01 00:00:00,1200.5

FORMAT 2 - 3 columns (multi area):
    area_id,Datetime,MW
    ZONE_A,2023-01-01 00:00:00,1200.5

Datetime accepted formats:
  - YYYY-MM-DD HH:MM:SS
  - MM/DD/YYYY H:MM
"""

import csv
import math
import random
from datetime import datetime, timedelta

random.seed(42)

# ─────────────────────────────────────────────
# Demand simulation: realistic hourly pattern
# ─────────────────────────────────────────────
def simulate_demand(hour, day_of_week, month, base_mw, noise=0.05):
    """
    Simulates realistic electricity demand.
    - Peak hours: 10:00 - 20:00
    - Weekend dip: ~15% lower
    - Summer peak: June-August higher
    - Winter peak: Dec-Feb higher
    """
    # Hour-of-day curve (normalized 0-1)
    hour_curve = (
        0.60 + 0.40 * math.sin(math.pi * (hour - 6) / 14.0)
        if 6 <= hour <= 22
        else 0.55
    )

    # Weekend factor
    weekend_factor = 0.85 if day_of_week >= 5 else 1.0

    # Season factor
    if month in [6, 7, 8]:      # summer (AC load)
        season_factor = 1.25
    elif month in [12, 1, 2]:   # winter (heating load)
        season_factor = 1.15
    elif month in [3, 4, 5]:    # spring
        season_factor = 0.90
    else:                        # fall
        season_factor = 0.95

    # Random noise
    noise_factor = 1.0 + random.uniform(-noise, noise)

    return round(base_mw * hour_curve * weekend_factor * season_factor * noise_factor, 1)


# ─────────────────────────────────────────────
# FORMAT 1: 2-column single-area CSV
# ─────────────────────────────────────────────
def generate_single_area(
    filename="test_single_area.csv",
    start_date="2023-01-01",
    days=90,
    base_mw=5000.0
):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    rows = []

    for d in range(days):
        for h in range(24):
            dt = start + timedelta(days=d, hours=h)
            mw = simulate_demand(h, dt.weekday(), dt.month, base_mw)
            rows.append({
                "Datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "MW": mw
            })

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Datetime", "MW"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] {filename} — {len(rows)} rows, {days} days, base={base_mw} MW")
    print(f"     Peak  MW : {max(r['MW'] for r in rows)}")
    print(f"     Median MW: {sorted(r['MW'] for r in rows)[len(rows)//2]}")
    print(f"     Avg   MW : {sum(r['MW'] for r in rows)/len(rows):.1f}")


# ─────────────────────────────────────────────
# FORMAT 2: 3-column multi-area CSV
# ─────────────────────────────────────────────
def generate_multi_area(
    filename="test_multi_area.csv",
    start_date="2023-01-01",
    days=90,
    areas=None   # dict of {area_id: base_mw}
):
    if areas is None:
        areas={
    "NORTH":  8000.0,
    "SOUTH":  5500.0,
    "EAST":   3200.0,
    "WEST":   7000.0,
    "CENTRAL":6000.0,
    "ZONE_A": 4500.0,
    "ZONE_B": 3800.0,
    "ZONE_C": 3400.0,
    "ZONE_D": 1800.0,

}

    start = datetime.strptime(start_date, "%Y-%m-%d")
    rows = []

    for d in range(days):
        for h in range(24):
            dt = start + timedelta(days=d, hours=h)
            for area_id, base_mw in areas.items():
                mw = simulate_demand(h, dt.weekday(), dt.month, base_mw)
                rows.append({
                    "area_id": area_id,
                    "Datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "MW": mw
                })

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["area_id", "Datetime", "MW"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] {filename} — {len(rows)} rows, {days} days, {len(areas)} areas")
    for area_id in areas:
        area_rows = [r["MW"] for r in rows if r["area_id"] == area_id]
        print(f"     {area_id}: peak={max(area_rows):.0f} MW, "
              f"median={sorted(area_rows)[len(area_rows)//2]:.0f} MW")


# ─────────────────────────────────────────────
# FORMAT 1 (alternate datetime): MM/DD/YYYY H:MM
# ─────────────────────────────────────────────
def generate_us_format(
    filename="test_us_datetime.csv",
    start_date="2023-01-01",
    days=30,
    base_mw=3000.0
):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    rows = []

    for d in range(days):
        for h in range(24):
            dt = start + timedelta(days=d, hours=h)
            mw = simulate_demand(h, dt.weekday(), dt.month, base_mw)
            rows.append({
                # US format: MM/DD/YYYY H:MM  (matches your parse_datetime)
                "Datetime": dt.strftime("%m/%d/%Y %H:%M"),
                "MW": mw
            })

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Datetime", "MW"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] {filename} — {len(rows)} rows (US date format MM/DD/YYYY H:MM)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Flow 2 Test Data Generator")
    print("=" * 55)

    # 1. Single area, 90 days, moderate load
    generate_single_area(
        filename="test_single_area.csv",
        start_date="2023-01-01",
        days=90,
        base_mw=5000.0
    )

    # 2. Multi area, 90 days, 3 zones
    generate_multi_area(
    filename="test_multi_area.csv",
    start_date="2023-01-01",
    days=360,
    areas={
        "NORTH":   8000.0,
        "SOUTH":   5500.0,
        "EAST":    3200.0,
        "WEST":    7000.0,
        "CENTRAL": 6000.0,
        "ZONE_A":  4500.0,
        "ZONE_B":  3800.0,
        "ZONE_C":  3400.0,
        "ZONE_D":  1800.0,
        "ZONE_E":  2600.0,
    }
)

    # 3. Single area, US date format
    generate_us_format(
        filename="test_us_datetime.csv",
        start_date="2023-06-01",
        days=30,
        base_mw=3000.0
    )

    print("\n" + "=" * 55)
    print("  Done! Upload any of these files to Flow 2.")
    print("  Recommended: test_single_area.csv to start.")
    print("=" * 55)