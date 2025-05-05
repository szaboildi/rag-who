import os
# from dotenv import load_dotenv

try:
    import tomllib # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse

from ragwho.qdrant_pipeline import rag_setup_qdrant, rag_query_once_qdrant


# load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
QDRANT_CLOUD_API_KEY = os.environ.get("QDRANT_CLOUD_API_KEY", "")


with open(os.path.join(
        "parameters_remote.toml"), mode="rb") as fp:
    config = tomllib.load(fp)

vector_db_client, encoder, gen_api_client = rag_setup_qdrant(
    config["remote"], from_scratch=False)

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@api.get("/")
def root():
    return dict(greeting = "Is this thing on?")

@api.get("/generate")
def generate_answer(query: str) -> dict:
    q, a = rag_query_once_qdrant(
        query, vector_db_client, encoder, gen_api_client, config["remote"])

    return dict(query=q, answer=a)
