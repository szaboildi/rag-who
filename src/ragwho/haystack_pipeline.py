import argparse

from ragwho.embedding import setup_vector_db

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from haystack.components.joiners import DocumentJoiner
# from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack import Pipeline


try:
    import tomllib # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib


def retrieval_pipeline_haystack(config):
    vector_db, embedder = setup_vector_db(
        encoder_name=config["encoder_name"],
        client_source=config["client_source"],
        input_folder=config["input_text_folder"],
        chunk_length=config["chunk_length"],
        chunk_overlap_words=config["chunk_overlap"],
        collection_name=config["collection_name"],
        dist_name=config["distance_type"],
        input_folder_qa=config["input_folder_qa"],
        relevance_score_file_prefix=config["relevance_score_file_prefix"],
        sample_qa_file=config["sample_qa_file"],
        mode="haystack")

    # Retriever
    k = config["retrieve_k"]
    if config["sparse_retriever"] == "BM25":
        k = config["retrieve_k_pre_rank"]
        sparse_retriever = InMemoryBM25Retriever(vector_db, top_k=k)
    embedding_retriever = InMemoryEmbeddingRetriever(vector_db, top_k=k)

    if config["sparse_retriever"] == "BM25":
        document_joiner = DocumentJoiner(
            top_k=config["retrieve_k"],
            join_mode="reciprocal_rank_fusion")

    # Build a pipeline
    retriever_pipeline = Pipeline()
    retriever_pipeline.add_component("text_embedder", embedder)
    retriever_pipeline.add_component("embedding_retriever", embedding_retriever)
    if config["sparse_retriever"] == "BM25":
        retriever_pipeline.add_component("sparse_retriever", sparse_retriever)
        retriever_pipeline.add_component("document_joiner", document_joiner)

    # Connect the components to each other
    retriever_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    if config["sparse_retriever"] == "BM25":
        retriever_pipeline.connect("embedding_retriever", "document_joiner")
        retriever_pipeline.connect("sparse_retriever", "document_joiner")

    return retriever_pipeline


def query_vector_db_once_haystack(
    retrieval_pipeline:Pipeline, query:str, config):
    if config["sparse_retriever"] == "BM25":
        raw_answer = retrieval_pipeline.run(
            {"text_embedder": {"text": query},
             "sparse_retriever": {"query": query}})
    elif config["sparse_retriever"].lower() == "none":
        raw_answer = retrieval_pipeline.run({
            "text_embedder": {"text": query}})
    else:
        raise ValueError('Invalid sparse_retriever provided in config. Use "BM25" or "None".')

    processed_answer = {"question": query,
    "answers": [{
        "text": doc.content,
        config["distance_type"].lower(): doc.score
        } for doc in raw_answer["document_joiner"]["documents"]]
    }

    return processed_answer


def query_vector_db_list_haystack(
    retrieval_pipeline:Pipeline, query_list:list[str], config):
    answer_list = [
        query_vector_db_once_haystack(
            retrieval_pipeline, q, config)
        for q in query_list]

    return answer_list


def rag_pipeline_haystack(
    config, api_key_variable:str="OPENAI_API_KEY"):
    vector_db, embedder = setup_vector_db(
        encoder_name=config["encoder_name"],
        client_source=config["client_source"],
        input_folder=config["input_text_folder"],
        chunk_length=config["chunk_length"],
        chunk_overlap_words=config["chunk_overlap"],
        collection_name=config["collection_name"],
        dist_name=config["distance_type"],
        input_folder_qa=config["input_folder_qa"],
        relevance_score_file_prefix=config["relevance_score_file_prefix"],
        sample_qa_file=config["sample_qa_file"],
        mode="haystack")

    # Retriever
    k = config["retrieve_k"]
    if config["sparse_retriever"] == "BM25":
        k = config["retrieve_k_pre_rank"]
        sparse_retriever = InMemoryBM25Retriever(vector_db, top_k=k)
    embedding_retriever = InMemoryEmbeddingRetriever(vector_db, top_k=k)

    if config["sparse_retriever"] == "BM25":
        document_joiner = DocumentJoiner(
            top_k=config["retrieve_k"],
            join_mode="reciprocal_rank_fusion")

    # Prompt builder
    with open(config["llm_system_prompt_path"], "r") as f:
        system_prompt = f.read()
    with open(config["llm_user_prompt_template"], "r") as f:
        user_template = f.read()
    template = [
        ChatMessage.from_system(system_prompt),
        ChatMessage.from_user(user_template)
    ]
    prompt_builder = ChatPromptBuilder(template=template)

    # LLM
    chat_generator = OpenAIChatGenerator(
        model="gpt-4o-mini", api_key=Secret.from_env_var(api_key_variable))

    # Make it into a pipeline
    rag_pipeline = Pipeline()

    rag_pipeline.add_component("text_embedder", embedder)
    rag_pipeline.add_component("embedding_retriever", embedding_retriever)
    if config["sparse_retriever"] == "BM25":
        rag_pipeline.add_component("sparse_retriever", sparse_retriever)
        rag_pipeline.add_component("document_joiner", document_joiner)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", chat_generator)
    # Connect the components to each other
    rag_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    if config["sparse_retriever"] == "BM25":
        rag_pipeline.connect("embedding_retriever", "document_joiner")
        rag_pipeline.connect("sparse_retriever", "document_joiner")
        rag_pipeline.connect("document_joiner", "prompt_builder")
    elif config["sparse_retriever"] == "None":
        rag_pipeline.connect("embedding_retriever", "prompt_builder")
    rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

    # rag_pipeline.draw("rag_pipeline.png")
    return rag_pipeline


def rag_query_once_haystack(rag_pipeline:Pipeline, query:str, config):
    if config["sparse_retriever"] == "BM25":
        response = rag_pipeline.run(
            {"text_embedder": {"text": query},
             "sparse_retriever": {"query": query},
             "prompt_builder": {"query": query}})
    elif config["sparse_retriever"].lower() == "none":
        response = rag_pipeline.run({
            "text_embedder": {"text": query}, "prompt_builder": {"question": query}})
    else:
        raise ValueError('Invalid sparse_retriever provided in config. Use "BM25" or "None".')

    return query, response["llm"]["replies"][0].text


def rag_query_list_haystack(
    rag_pipeline:Pipeline, queries:list[str], config):
    responses = [rag_query_once_haystack(rag_pipeline, q, config)[1] for q in queries]

    return queries, responses


if __name__ == "__main__":
    with open("parameters.toml", mode="rb") as fp:
        config = tomllib.load(fp)

    parser=argparse.ArgumentParser(description="argument parser for rag-who")
    parser.add_argument("--config_name", nargs='?', default="default")
    args=parser.parse_args()

    # # print(args.config_name)

    # question = "How long do rabbits live?"
    # question = "How many deaths does alcoholism cause a year in the European Region?"
    # question = "How many deaths does alcoholism cause a year in the world?"
    # question = "How much should a child exercise?"
    question = "How much should an eight year-old exercise?"
    # question = "How much should a 50 year-old exercise?"
    # question = "How much should my grandmother exercise?"


    # pipeline = rag_pipeline_haystack(config=config[args.config_name])
    # query, response = rag_query_once_haystack(
    #     pipeline, question, config[args.config_name])
    # print(query, response, sep="\n")

    pipeline = retrieval_pipeline_haystack(config=config[args.config_name])
    result = query_vector_db_once_haystack(
        pipeline, question, config[args.config_name])
    print(result)
