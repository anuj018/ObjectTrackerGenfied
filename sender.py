# sender.py
import aiohttp
from config import config
from typing import List

DETECTION_ENDPOINT = config["detection_endpoint"]

async def send_detection_data(detections: List[dict]):
    async with aiohttp.ClientSession() as session:
        async with session.put(DETECTION_ENDPOINT, json=detections) as response:
            if response.status == 200:
                print("Data sent successfully.")
            else:
                print(f"Failed to send data. Status: {response.status}")
                # Optionally, handle retries or logging here
