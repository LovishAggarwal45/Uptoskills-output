from plate_utils import validate_indian_plate

tests = {
    "TN10AB1234": True,
    "UP32HX7289": True,
    "22BH1234AA": True,
    "INVALID123": False,
    "": False,
}

for plate, expected in tests.items():
    actual = validate_indian_plate(plate)
    print(
        f"{plate or '[empty]'} | "
        f"Expected={expected} | Actual={actual}"
    )