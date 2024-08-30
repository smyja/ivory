import os
from together import Together
from dotenv import load_dotenv
import httpx
import json


load_dotenv()
client = Together(api_key=os.environ["api_key"])

response = client.embeddings.create(
  model="BAAI/bge-large-en-v1.5",
  input="Our solar system orbits the Milky Way galaxy at about 515,000 mph",
)
print(response.data[0].embedding)

