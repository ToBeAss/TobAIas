import aloe.sensor as sensor
import aloe.message as message
import scripts.discord as discord
import my_secrets.webhooks as webhooks

def main():
    data = sensor.read_data()
    content = message.create_message(data)
    discord.send_to_discord(webhooks.aloe, content)