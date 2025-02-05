import scripts.timeout as timeout
import aloe.main as aloe
import scripts.discord as discord
import my_secrets.webhooks as webhooks

def main():
    # Test function
    discord.send_to_discord(webhooks.aloe, "rebooting")
    while True:
        # Wait until next whole hour before running the rest of the program
        timeout.next_hour()

        # Run projects
        aloe.main()

main()