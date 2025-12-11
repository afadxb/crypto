"""Pushover alerting utility."""
import os
from typing import Optional

import requests


def send_alert(title: str, message: str) -> Optional[requests.Response]:
    user = os.getenv("PUSHOVER_USER_KEY")
    token = os.getenv("PUSHOVER_API_TOKEN")
    if not user or not token:
        return None

    return requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": token,
            "user": user,
            "title": title,
            "message": message,
            "priority": 0,
        },
        timeout=10,
    )
