# azure_client.py
from azure.storage.blob import BlobServiceClient
from config import config

AZURE_CONNECTION_STRING = config["azure_connection_string"]
AZURE_CONTAINER_NAME = config["azure_container_name"]

# Initialize Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
