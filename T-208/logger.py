import csv
import os
from datetime import datetime

LOG_FILE = "aggression_alerts.csv"

def initialize_logger():
    """Creates the CSV file and writes the header if it doesn't exist."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists or os.stat(LOG_FILE).st_size == 0:
            writer.writerow(["Timestamp", "Location", "Condition", "Risk_Score"])

def log_event(location_id, condition, score):
    """Appends a new aggression event to the CSV log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, location_id, condition, round(score, 2)])