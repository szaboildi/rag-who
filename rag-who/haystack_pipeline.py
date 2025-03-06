import os
import json
import argparse
from openai import OpenAI
import pandas as pd
import time

from retrieval import setup_vector_db

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack import Pipeline


try:
    import tomllib # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib



def rag_pipeline(config_name:str="default", api_key_variable:str="OPENAI_API_KEY"):
    vector_db, embedder = setup_vector_db(
        encoder_name=config[config_name]["encoder_name"],
        client_source=config[config_name]["client_source"],
        input_folder=config[config_name]["input_text_folder"],
        chunk_length=config[config_name]["chunk_length"],
        chunk_overlap_words=config[config_name]["chunk_overlap"],
        collection_name=config[config_name]["collection_name"],
        dist_name=config[config_name]["distance_type"],
        input_folder_qa=config[config_name]["input_folder_qa"],
        relevance_score_file_prefix=config[config_name]["relevance_score_file_prefix"],
        sample_qa_file=config[config_name]["sample_qa_file"],
        mode="haystack")

    # Retriever
    retriever = InMemoryEmbeddingRetriever(vector_db)

    # Prompt builder
    with open(config[config_name]["llm_system_prompt_path"], "r") as f:
        system_prompt = f.read()
    with open(config[config_name]["llm_user_prompt_template"], "r") as f:
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
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", chat_generator)
    # Now, connect the components to each other
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder")
    rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

    return rag_pipeline


def rag_query_once_haystack(rag_pipeline:Pipeline, query:str):
    response = rag_pipeline.run({"text_embedder": {"text": query}, "prompt_builder": {"question": query}})

    return query, response["llm"]["replies"][0].text


def rag_query_list_qdrant(
    rag_pipeline:Pipeline, queries:list[str]):
    responses = [rag_query_once_haystack(rag_pipeline, q)[1] for q in queries]

    return queries, responses


if __name__ == "__main__":
    with open("rag-who.toml", mode="rb") as fp:
        config = tomllib.load(fp)

    parser=argparse.ArgumentParser(description="argument parser for rag-who")
    parser.add_argument("--config_name", nargs='?', default="default")
    args=parser.parse_args()

    # print(args.config_name)
    # retrieve_and_eval(config_name=args.config_name)
    pipeline = rag_pipeline(config_name="default")
    # question = "How long do rabbits live?"
    # question = "How many deaths does alcoholism cause a year in the European Region?"
    question = "How many deaths does alcoholism cause a year in the world?"

    query, response = rag_query_once_haystack(pipeline, question)
    print(query, response, sep="\n")
