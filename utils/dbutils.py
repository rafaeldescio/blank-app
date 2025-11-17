from pymongo import MongoClient
from dotenv import load_dotenv
import os
from pymongoarrow.api import find_pandas_all

load_dotenv()  # take environment variables from .env.

CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_COLLECTION = os.getenv("DB_COLLECTION")

def get_database():
    # Create a connection using MongoClient
    client = MongoClient(CONNECTION_STRING)
    
    # Specify the database you want to access
    db_name = DB_DATABASE # Replace with your database name
    return client[db_name]

def get_dataframe():
    client = None
    try:
        dbname = get_database()
        
        # Access a specific collection within your database
        collection_name = DB_COLLECTION # Replace with your collection name
        collection = dbname[collection_name]
        
        # Find all documents in the collection
        # You can add a query filter as a dictionary to find specific documents
        # For example: documents = collection.find({"name": "John Doe"})
        # documents = collection.find({}) 
        
        # print(f"Documents in '{collection_name}':")
        # for doc in documents:
        #     print(doc)
        
        # # Find a single document
        # # For example: single_doc = collection.find_one({"_id": "some_id"})
        # single_doc = collection.find_one({}) # Finds the first document
        # if single_doc:
        #     print("\nFirst document found:")
        #     print(single_doc)
        # else:
        #     print("\nNo documents found.")
        
        # Close the connection (optional, as client will be garbage collected)
        client = dbname.client
        df = find_pandas_all(collection, {}, projection={'_id': 0})
    finally:
        if client:
            client.close()
    return df