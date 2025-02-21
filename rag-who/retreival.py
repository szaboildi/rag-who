from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

from preprocessing import process_text
import os
# import json

def setup_vector_db():
    # Set up encoder and client
    encoder = SentenceTransformer("intfloat/e5-base")
    client = QdrantClient(":memory:")

    # Set up the passages
    input_passages_dict = process_text(
            os.path.join("data", "alcohol-use.txt"),
            length=100, words_overlap=15, return_format="dict")

    client.create_collection(
        collection_name="who_guidelines",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance=models.Distance.COSINE,
        ),
    )

    client.upload_points(
        collection_name="who_guidelines",
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(
                    doc["text"], normalize_embeddings=True).tolist(),
                payload=doc)
            for idx, doc in enumerate(input_passages_dict)
        ],
    )

    return client, encoder

def query_vector_db_once(client, encoder, question:str):
    # Set up queries
    # with open(os.path.join("data", "sample_qa.json")) as f:
    #     d = json.load(f)
    # input_queries = [item["question"] for item in d]

    raw_answer = client.query_points(
        collection_name="who_guidelines",
        query=encoder.encode(question, normalize_embeddings=True).tolist(),
        limit=4).points

    processed_answer = {"question": question,
    "answers": [{
        "text": hit.payload["text"],
        "cosine": hit.score} for hit in raw_answer]
    }

    return processed_answer

def query_vector_db_list(client, encoder, question_list:list[str]):
    answer_list = [
        query_vector_db_once(client, encoder, q) for q in question_list]
    return answer_list

# # Set up queries
# with open(os.path.join("data", "sample_qa.json")) as f:
#     d = json.load(f)
# input_queries = [item["question"] for item in d]

# raw_answers = [client.query_points(
#     collection_name="who_guidelines", query=encoder.encode(q).tolist(),
#     limit=5).points for q in input_queries]

# processed_answers = [{"question": input_queries[i],
# "answers": [{
#     "text": hit.payload["text"],
#     "cosine": hit.score} for hit in a]
# } for i, a in enumerate(raw_answers)]

# print(processed_answers)

# TODO: parametrize in config
# client source
# embedding model
# distance type
# inputs:
#   location
#   chunk length
#   chunk overlap (words)
