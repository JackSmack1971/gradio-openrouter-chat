from __future__ import annotations  # Fixed: double underscores, not asterisks
import re, time, json, csv, threading
from typing import Dict, List, Generator, Iterable, Any
from collections import deque, defaultdict
from pathlib import Path

SANITIZE_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def sanitize_text(s: str, max_chars: int) -> str:
    s = SANITIZE_RE.sub("", s or "")
    return s[:max_chars]

def trim_history(messages: List[dict], max_messages: int) -> List[dict]:
    # keep the latest N messages (system included if present)
    if len(messages) <= max_messages:
        return messages
    # preserve first system message if exists
    system = [m for m in messages if m["role"] == "system"][:1]
    rest = [m for m in messages if m["role"] != "system"]
    return (system + rest[-max_messages:]) if system else rest[-max_messages:]

# --- Simple token-bucket per-IP limiter (in-memory) ---
class RateLimiter:
    def __init__(self, capacity_per_min: int):  # Fixed: double underscores, not asterisks
        self.capacity = capacity_per_min
        self.allowance: Dict[str, float] = defaultdict(lambda: float(capacity_per_min))
        self.last_check: Dict[str, float] = defaultdict(time.time)
        self.lock = threading.Lock()  # Correct usage per Python docs

    def check(self, key: str) -> bool:
        now = time.time()
        with self.lock:  # Using context manager - correct per Python threading docs
            last = self.last_check[key]
            elapsed = now - last
            self.last_check[key] = now
            # Token bucket algorithm: replenish tokens based on elapsed time
            self.allowance[key] = min(self.capacity, self.allowance[key] + elapsed * (self.capacity / 60.0))
            if self.allowance[key] < 1.0:
                return False
            self.allowance[key] -= 1.0
            return True

# --- Persistence / Analytics ---
DATA_DIR = Path(".data")
DATA_DIR.mkdir(exist_ok=True)
LOG_CSV = DATA_DIR / "usage.csv"

def log_usage(row: dict) -> None:
    # Initialize CSV with headers if it doesn't exist
    if not LOG_CSV.exists():
        LOG_CSV.write_text("ts,ip,model,input_tokens,output_tokens,latency_ms,cost_estimate\n", encoding="utf-8")
    
    # Append usage data to CSV
    with LOG_CSV.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=["ts","ip","model","input_tokens","output_tokens","latency_ms","cost_estimate"]).writerow(row)

def export_conversation(history: List[dict]) -> str:
    # Export conversation history as timestamped JSON file
    path = DATA_DIR / f"chat_{int(time.time())}.json"
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)