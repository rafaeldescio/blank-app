from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER")
BLOB_NAME = os.getenv("BLOB_NAME")

def get_blobservice():    
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    return blob_service_client

def get_blobclient():
    blob_service_client = get_blobservice()
    # Get a blob client
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER, blob=BLOB_NAME)
    return blob_client