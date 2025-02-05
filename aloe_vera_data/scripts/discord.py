import requests

def send_to_discord(webhook: str, content: str):
    data = {
        "content": content
    }

    response = requests.post(webhook, json=data)

    if response.status_code == 204:
        print("Message sent to Discord successfully.")
    else:
        print(f"Failed to send message. Status code: {response.status_code}")