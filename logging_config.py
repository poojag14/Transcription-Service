#logging_config.py
import logging
import os
import time
from datetime import datetime, timedelta


LOG_FILE = "transcription.log"

# --- CLEANUP FUNCTION ---
def clean_old_logs(filepath, days=7):
    """
    Retains only log lines with timestamps within the last 'days' days.
    Lines without valid timestamps are preserved.
    """
    if not os.path.exists(filepath):
        return

    threshold = datetime.now() - timedelta(days=days)
    new_lines = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                timestamp_str = line.split(" - ")[0].strip()
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                if timestamp >= threshold:
                    new_lines.append(line)
            except Exception:
                # Keep malformed lines just in case
                new_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

# --- LOGGER SETUP ---
logger = logging.getLogger("transcription_logger")
logger.setLevel(logging.INFO)

# Avoid adding duplicate handlers if re-imported
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter(
        '%(asctime)s - jobId=%(jobId)s - fileUrl=%(fileUrl)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Optional: automatically clean old logs on import
clean_old_logs(LOG_FILE, days=7)
