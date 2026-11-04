
import ast
from pathlib import Path
from collections import Counter, defaultdict
import re

# Load main.py as text without executing the video pipeline.
source = Path("main.py").read_text(encoding="utf-8")
tree = ast.parse(source)

# Extract only choose_final_plate().
function_node = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "choose_final_plate"
)

function_code = ast.unparse(function_node)

def normalize_plate(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def validate_plate(text):
    text = normalize_plate(text)
    return bool(
        re.fullmatch(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}", text)
        or re.fullmatch(r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}", text)
    )

namespace = {
    "normalize_plate": normalize_plate,
    "validate_plate": validate_plate,
    "Counter": Counter,
    "defaultdict": defaultdict,
}

exec(function_code, namespace)
choose_final_plate = namespace["choose_final_plate"]

tests = [
    (
        "Repeated valid Indian plate",
        [
            ("TN09AB1234", 0.90, 1.0),
            ("TN09AB1234", 0.95, 2.0),
            ("R183JF", 0.80, 3.0),
        ],
    ),
    (
        "Invalid-only OCR observations",
        [
            ("R183JF", 0.80, 1.0),
            ("N894", 0.70, 2.0),
        ],
    ),
    ("No observations", []),
]

for name, votes in tests:
    print(f"\n{name}:")
    print(choose_final_plate(votes))
