import json
import os
import sys

nb_path = r"C:\Users\91636\.gemini\antigravity\scratch\crowd_surveillance_pipeline\Adverse_Weather_Crowd_Surveillance_Pipeline.ipynb"

print(f"Verifying notebook at: {nb_path}")
if not os.path.exists(nb_path):
    print("[ERROR] Notebook file does not exist!")
    sys.exit(1)

file_size_kb = os.path.getsize(nb_path) / 1024
print(f"Notebook file size: {file_size_kb:.2f} KB")

with open(nb_path, "r", encoding="utf-8") as f:
    try:
        nb_data = json.load(f)
        print("[SUCCESS] Notebook JSON structure is 100% valid.")
    except Exception as e:
        print(f"[ERROR] Invalid JSON: {e}")
        sys.exit(1)

cells = nb_data.get("cells", [])
print(f"Total cells: {len(cells)}")

code_cells = [c for c in cells if c.get("cell_type") == "code"]
md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
print(f"Markdown cells: {len(md_cells)}")
print(f"Code cells    : {len(code_cells)}")

errors = []
for idx, c in enumerate(code_cells):
    code_text = "".join(c.get("source", []))
    try:
        compile(code_text, f"<cell_{idx}>", "exec")
    except SyntaxError as se:
        errors.append((idx, str(se), code_text[:120]))

if errors:
    print(f"[ERROR] Found {len(errors)} syntax errors in code cells:")
    for idx, err, snippet in errors:
        print(f"  Cell {idx}: {err}\n  Snippet: {snippet}...\n")
    sys.exit(1)
else:
    print("[SUCCESS] All code cells compiled with zero Python syntax errors!")
