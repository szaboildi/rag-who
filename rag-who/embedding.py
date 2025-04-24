from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder

from preprocessing import process_text
import glob


def setup_vector_db(
    encoder_name:str="intfloat/e5-base", client_source:str=":memory:",
    qdrant_cloud_api_key:str="None", from_scratch:bool=False,
    input_folder:str="data/raw_input_files",
    chunk_length:int=200, chunk_overlap_words:int=20,
    collection_name:str="dummy_name", dist_name:str="COSINE",
    input_folder_qa:str="data",
    relevance_score_file_prefix:str="sample_qa_passage_lvl",
    sample_qa_file:str="sample_qa.json", mode="qdrant"):
    if mode=="qdrant":

        # Set up encoder and client
        encoder = SentenceTransformer(encoder_name)
        if client_source == ":memory":
            client = QdrantClient(client_source)
        else:
            client = QdrantClient(
                url=client_source, api_key=qdrant_cloud_api_key)

        if not from_scratch:
            print("Vector database loaded")
            return client, encoder

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

    elif mode=="haystack":
        if dist_name.lower()=="cosine":
            dist_name = dist_name.lower()
        elif dist_name.lower()=="dot" or dist_name.lower()=="dotproduct" or\
            dist_name.lower()=="dot_product":
            dist_name = "dot_product"
        else:
            raise ValueError(
                """
                Invalid similarity type provided for haystack.
                You can use \"cosine\" or \"dot_product\"""")

        doc_embedder = SentenceTransformersDocumentEmbedder(model=encoder_name)
        doc_embedder.warm_up()

        document_store = InMemoryDocumentStore(
            embedding_similarity_function=dist_name)

        file_names = glob.glob(f"{input_folder}/*")
        for file_name in file_names:
            # Set up the passages
            input_passages = process_text(
                file_name, length=chunk_length, words_overlap=chunk_overlap_words,
                return_format="ls_haystack_doc", input_folder_qa=input_folder_qa,
                relevance_score_file_prefix=relevance_score_file_prefix,
                sample_qa_file=sample_qa_file)

            print(f"Clean chunks created for {file_name}")

            embedded_input_passages = doc_embedder.run(input_passages)
            document_store.write_documents(embedded_input_passages["documents"])

        query_embedder = SentenceTransformersTextEmbedder(model=encoder_name)

        return document_store, query_embedder
