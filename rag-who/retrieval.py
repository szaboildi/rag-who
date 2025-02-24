from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

from preprocessing import process_text
import glob


def setup_vector_db(
    encoder_name:str="intfloat/e5-base", client_source:str=":memory:",
    input_folder:str="data/raw_input_files",
    chunk_length:int=200, chunk_overlap_words:int=20,
    collection_name:str="dummy_name", dist_name:str="COSINE",
    input_folder_qa:str="data",
    relevance_score_file_prefix:str="sample_qa_passage_lvl",
    sample_qa_file:str="sample_qa.json"):
    # Set up encoder and client
    encoder = SentenceTransformer(encoder_name)
    client = QdrantClient(client_source)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance=getattr(models.Distance, dist_name.upper()),
        ),
    )

    file_names = glob.glob(f"{input_folder}/*")
    for file_name in file_names:
        # Set up the passages
        input_passages_dict = process_text(
            file_name, length=chunk_length, words_overlap=chunk_overlap_words,
            return_format="ls_dict", input_folder_qa=input_folder_qa,
            relevance_score_file_prefix=relevance_score_file_prefix,
            sample_qa_file=sample_qa_file)

        print(f"Clean chunks created for {file_name}")

        client.upload_points(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=idx, vector=encoder.encode(
                        doc["text"], normalize_embeddings=True).tolist(),
                    payload=doc)
                for idx, doc in enumerate(input_passages_dict)
            ],
        )

    print("Vector database created")
    return client, encoder


def query_vector_db_once(
    client, encoder, question:str, collection_name:str="dummy_name", k:int=5,
    dist_name:str="COSINE"):

    raw_answer = client.query_points(
        collection_name=collection_name,
        query=encoder.encode(question, normalize_embeddings=True).tolist(),
        limit=k).points

    processed_answer = {"question": question,
    "answers": [{
        "text": hit.payload["text"],
        dist_name.lower(): hit.score} for hit in raw_answer]
    }

    return processed_answer


def query_vector_db_list(
    client, encoder, question_list:list[str],
    collection_name:str="dummy_name", k:int=5):
    answer_list = [
        query_vector_db_once(
            client, encoder, q, collection_name, k) for q in question_list]

    return answer_list
