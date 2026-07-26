import requests
import os
from dotenv import load_dotenv

load_dotenv()

headers = {"X-Auth-Token": os.getenv("FOOTBALL_DATA_KEY")}

response = requests.get(
    "https://api.football-data.org/v4/competitions/PL/scorers",
    headers=headers,
    params={"season": "2025"}
)

data = response.json()
scorers = data.get("scorers", [])
if scorers:
    print(scorers[0])