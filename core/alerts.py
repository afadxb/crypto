import os
import requests

def send_alert(title, message):
    user = os.getenv("PUSHOVER_USER_KEY")
    token = os.getenv("PUSHOVER_API_TOKEN")

    requests.post("https://api.pushover.net/1/messages.json", data={
        "token": token,
        "user": user,
        "title": title,
        "message": message,
        "priority": 0
    })
