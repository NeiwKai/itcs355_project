from __future__ import annotations

import json
import os
import smtplib
import sys
import time
from datetime import datetime, timezone
from email.message import EmailMessage
import urllib.request

HF_TOKEN = os.getenv("HF_TOKEN")
HF_NAMESPACE = os.getenv("HF_NAMESPACE", "NeiwKai")
TELEMETRY_URL = f"https://huggingface.co/datasets/{HF_NAMESPACE}/itcs355-metrics/raw/main/telemetry.jsonl"

# Configured Alert Thresholds
THRESHOLDS = {
    "drift.psi.volatile_acidity": 0.20,
    "drift.ks.volatile_acidity": 0.10,
}


def fetch_telemetry_jsonl() -> list[dict]:
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    req = urllib.request.Request(TELEMETRY_URL, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            lines = response.read().decode("utf-8").strip().splitlines()
            return [json.loads(line) for line in lines if line.strip()]
    except Exception as e:
        print(f"Error downloading telemetry.jsonl from HF: {e}")
        sys.exit(1)


def acknowledge_alerts_in_hf(ack_items: list[tuple[str, float]]):
    """Appends alert_acknowledged records to HF telemetry.jsonl for all dispatched breaches."""
    if not HF_TOKEN:
        print("⚠️ HF_TOKEN missing. Skipping automated acknowledgment commit.")
        return

    try:
        from huggingface_hub import HfApi, HfFileSystem

        fs = HfFileSystem(token=HF_TOKEN)
        repo_file = f"datasets/{HF_NAMESPACE}/itcs355-metrics/telemetry.jsonl"

        # Read current content to append safely
        current_content = ""
        if fs.exists(repo_file):
            with fs.open(repo_file, "r", encoding="utf-8") as f:
                current_content = f.read()

        new_lines = []
        for metric_name, target_ts in ack_items:
            ack_record = {
                "event": "alert_acknowledged",
                "metric": metric_name,
                "target_timestamp": target_ts,
                "timestamp": time.time(),
            }
            new_lines.append(json.dumps(ack_record))

        updated_content = current_content.rstrip("\n") + "\n" + "\n".join(new_lines) + "\n"

        api = HfApi(token=HF_TOKEN)
        repo_id = f"{HF_NAMESPACE}/itcs355-metrics"
        api.upload_file(
            path_or_fileobj=updated_content.encode("utf-8"),
            path_in_repo="telemetry.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="ack: automated drift alert notification recorded",
        )
        print("✓ Recorded alert acknowledgment events directly in Hugging Face telemetry repo.")

    except Exception as e:
        print(f"⚠️ Failed to record alert acknowledgment in HF: {e}")


def send_email_alert(breaches: list[str], fired_at_utc: str) -> bool:
    sender = os.getenv("ALERT_SENDER_EMAIL")
    password = os.getenv("ALERT_SENDER_PASSWORD")
    recipient = os.getenv("ALERT_RECIPIENT_EMAIL")

    if not (sender and password and recipient):
        print("⚠️ Email secrets missing (ALERT_SENDER_EMAIL, etc.). Printing alerts to console:\n")
        print(f"Fired At: {fired_at_utc}")
        print("\n".join(breaches))
        return False

    msg = EmailMessage()
    msg["Subject"] = f"🚨 MLOps Alert: Feature Drift Breached ({fired_at_utc})"
    msg["From"] = sender
    msg["To"] = recipient

    body = (
        f"MLOps Monitoring Alert — Drift Threshold Breached\n"
        f"Fired At (UTC): {fired_at_utc}\n\n"
        f"Detected Metric Breaches:\n"
        + "\n".join(f"• {b}" for b in breaches)
        + f"\n\nTelemetry Repo: https://huggingface.co/datasets/{HF_NAMESPACE}/itcs355-metrics\n"
        + "Action Required: Review feature distributions or trigger model retraining pipeline."
    )
    msg.set_content(body)

    try:
        with smtplib.SMTP(
            os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            int(os.getenv("SMTP_PORT", "587")),
        ) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        print(f"✓ Alert email sent successfully to {recipient} at {fired_at_utc}")
        return True
    except Exception as e:
        print(f"Failed to send alert email: {e}")
        return False


def main():
    records = fetch_telemetry_jsonl()
    if not records:
        print("No telemetry records found.")
        return

    # 1. Parse all previously acknowledged event keys (metric:target_timestamp)
    acknowledged_events = set()
    for r in records:
        if r.get("event") == "alert_acknowledged":
            ack_key = f"{r.get('metric')}:{r.get('target_timestamp')}"
            acknowledged_events.add(ack_key)

    breaches_to_alert = []
    ack_items_to_commit = []
    fired_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 2. Check for unacknowledged threshold breaches
    for metric_name, threshold in THRESHOLDS.items():
        metric_entries = [r for r in records if r.get("metric") == metric_name]
        if not metric_entries:
            continue

        latest = metric_entries[-1]
        val = float(latest.get("value", 0))
        logged_ts = float(latest.get("timestamp", 0))

        ack_key = f"{metric_name}:{logged_ts}"

        if val > threshold:
            if ack_key in acknowledged_events:
                print(
                    f"ℹ️ Breach on {metric_name} ({val:.4f}) already notified for timestamp {logged_ts}. Skipping."
                )
            else:
                ts_str = (
                    datetime.fromtimestamp(logged_ts, timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S UTC"
                    )
                    if logged_ts
                    else fired_at_utc
                )
                breaches_to_alert.append(
                    f"{metric_name}: current={val:.4f} exceeds threshold={threshold} (logged at: {ts_str})"
                )
                ack_items_to_commit.append((metric_name, logged_ts))

    # 3. Send email and record acknowledgment if new breaches were found
    if breaches_to_alert:
        sent = send_email_alert(breaches_to_alert, fired_at_utc)
        if sent and ack_items_to_commit:
            acknowledge_alerts_in_hf(ack_items_to_commit)
    else:
        print("✓ All metrics are within thresholds or already acknowledged.")


if __name__ == "__main__":
    main()
