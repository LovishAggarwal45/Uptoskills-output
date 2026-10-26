"""
Violation Analytics & HTML Dashboard Generator
Reads violations CSV and produces formatted terminal analytics
as well as a standalone interactive HTML inspection report.
"""

import argparse
import csv
import os
import sys
from collections import Counter
from typing import Dict, List

def analyze_and_generate_report(csv_path: str = "output/violations_real.csv", report_path: str = "output/report.html"):
    if not os.path.exists(csv_path):
        print(f"[!] Error: CSV file '{csv_path}' not found.", file=sys.stderr)
        return

    records: List[Dict[str, str]] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    total_violations = len(records)
    if total_violations == 0:
        print("[*] No violations found in CSV.")
        return

    # Aggregate stats
    types_counter = Counter(r["violation_type"] for r in records)
    classes_counter = Counter(r["vehicle_type"] for r in records)
    lanes_counter = Counter(r["lane"] for r in records)
    
    confs = [float(r["confidence"]) for r in records if r["confidence"]]
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    print("=" * 65)
    print("        AI TRAFFIC VIOLATION AUDIT SUMMARY REPORT        ")
    print("=" * 65)
    print(f"Total Violations Recorded: {total_violations}")
    print(f"Average Detection Confidence: {avg_conf:.1%}")
    print("\n--- Breakdown by Violation Type ---")
    for vtype, count in types_counter.items():
        pct = (count / total_violations) * 100
        print(f"  • {vtype}: {count} ({pct:.1f}%)")

    print("\n--- Breakdown by Vehicle Type ---")
    for vclass, count in classes_counter.items():
        pct = (count / total_violations) * 100
        print(f"  • {vclass.capitalize()}: {count} ({pct:.1f}%)")

    print("\n--- Breakdown by Lane ---")
    for lane, count in lanes_counter.items():
        pct = (count / total_violations) * 100
        print(f"  • {lane}: {count} ({pct:.1f}%)")
    print("=" * 65)

    # Generate Standalone Interactive HTML Report
    table_rows = []
    for r in records:
        vtype_badge = "badge-danger" if "Wrong-Way" in r.get("violation_type", "") else "badge-warning"
        
        table_rows.append(f"""
        <tr>
            <td><strong>{r.get('violation_id', '')}</strong></td>
            <td>#{r.get('vehicle_id', '')}</td>
            <td><span class='badge-vehicle'>{r.get('vehicle_type', '').capitalize()}</span></td>
            <td><span class='{vtype_badge}'>{r.get('violation_type', '')}</span></td>
            <td>{r.get('lane', '')}</td>
            <td><code>{r.get('timestamp', '')}</code></td>
            <td>{float(r.get('confidence', 0)):.1%}</td>
            <td>{r.get('heading_deg', '0')}°</td>
            <td>{r.get('deviation_deg', '0')}°</td>
        </tr>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Traffic Violation Audit Dashboard</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --accent: #38bdf8;
            --danger: #ef4444;
            --warning: #f59e0b;
            --success: #10b981;
            --border: #334155;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 30px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 1.8rem;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
        }}
        .kpi-title {{
            color: var(--text-dim);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 2rem;
            font-weight: 700;
        }}
        .table-container {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.95rem;
        }}
        th {{
            background: #111827;
            padding: 14px 18px;
            color: var(--text-dim);
            font-weight: 600;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 14px 18px;
            border-bottom: 1px solid var(--border);
        }}
        tr:hover {{
            background: rgba(255,255,255,0.03);
        }}
        .badge-danger {{
            background: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.4);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .badge-warning {{
            background: rgba(245, 158, 11, 0.2);
            color: #fcd34d;
            border: 1px solid rgba(245, 158, 11, 0.4);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .badge-vehicle {{
            background: rgba(56, 189, 248, 0.15);
            color: #7dd3fc;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.85rem;
        }}
        .btn-clip {{
            display: inline-block;
            background: #2563eb;
            color: #fff;
            padding: 6px 12px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
            transition: background 0.2s;
        }}
        .btn-clip:hover {{
            background: #1d4ed8;
        }}
        .no-clip {{
            color: var(--text-dim);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚨 AI Traffic Monitoring Violation Audit</h1>
        <div style="color: var(--text-dim); font-size: 0.9rem;">Source: <code>{os.path.basename(csv_path)}</code></div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Total Violations</div>
            <div class="kpi-value" style="color: var(--accent);">{total_violations}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Wrong-Way Driving</div>
            <div class="kpi-value" style="color: var(--danger);">{types_counter.get('Wrong-Way Driving', 0)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Lane Discipline (Straddling)</div>
            <div class="kpi-value" style="color: var(--warning);">{types_counter.get('Lane Discipline Violation (Straddling)', 0)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Avg Detection Confidence</div>
            <div class="kpi-value" style="color: var(--success);">{avg_conf:.1%}</div>
        </div>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Violation ID</th>
                    <th>Vehicle</th>
                    <th>Type</th>
                    <th>Infraction</th>
                    <th>Lane</th>
                    <th>Timestamp</th>
                    <th>Confidence</th>
                    <th>Heading</th>
                    <th>Deviation</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n[*] Standalone Interactive HTML Report generated at: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Traffic Violations CSV and Generate Report")
    parser.add_argument("--csv", default="output/violations_real.csv", help="Path to violations CSV")
    parser.add_argument("--report", default="output/report.html", help="Path to output HTML report")
    args = parser.parse_args()

    analyze_and_generate_report(csv_path=args.csv, report_path=args.report)

if __name__ == "__main__":
    main()
