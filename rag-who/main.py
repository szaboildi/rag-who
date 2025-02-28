import os
import json
import argparse
from openai import OpenAI
import pandas as pd
import time

from retrieval import setup_vector_db, query_vector_db_list, query_vector_db_once
from eval import eval_recall_sentence, eval_recall_passage, eval_mrr_sentence, ndcg_scorer_manual
from generation import create_qa_string, api_call #, QuestionAnswering


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

def retrieve_and_eval(config_name:str="default"):
    with open(os.path.join(config[config_name]["input_folder_qa"],
                           config[config_name]["sample_qa_file"])) as f:
        correct_answers = json.load(f)
    input_queries = [item["question"] for item in correct_answers]\

    client, encoder = setup_vector_db(
        encoder_name=config[config_name]["encoder_name"],
        client_source=config[config_name]["client_source"],
        input_folder=config[config_name]["input_text_folder"],
        chunk_length=config[config_name]["chunk_length"],
        chunk_overlap_words=config[config_name]["chunk_overlap"],
        collection_name=config[config_name]["collection_name"],
        dist_name=config[config_name]["distance_type"],
        input_folder_qa=config[config_name]["input_folder_qa"],
        relevance_score_file_prefix=config[config_name]["relevance_score_file_prefix"],
        sample_qa_file=config[config_name]["sample_qa_file"])
    results = query_vector_db_list(
        client, encoder, input_queries,
        collection_name = config[config_name]["collection_name"],
        k=config[config_name]["retrieve_k"])
    print("Queries completed")

    with open(os.path.join(
        config[config_name]["input_folder_qa"],
        f'{config[config_name]["relevance_score_file_prefix"]}_length' + \
        f'{config[config_name]["chunk_length"]}_overlap' + \
        f'{config[config_name]["chunk_overlap"]}.json',)) as f:
        correct_answers_passage_rel = json.load(f)

    rc = eval_recall_sentence(results, correct_answers)
    print(f"\nRecall (sentence-level): {sum(rc[0]) / len(rc[0]):.2%}")
    print(f"    Item-level: {[round(recall, 4) for recall in rc[0]]}\n")
    rc_passages = eval_recall_passage(results, correct_answers_passage_rel)
    print(f"Recall (passage-level): {sum(rc_passages[0]) / len(rc_passages[0]):.2%}")
    print(f"    Item-level: {[round(recall, 4) for recall in rc_passages[0]]}\n")

    mrr = eval_mrr_sentence(results, correct_answers)
    print(f"MRR (mean): {(sum(mrr)/len(mrr)):.2}")
    print(f"    Item-level: {[round(score,2) for score in mrr]}\n")

    # ndcg = ndcg_scorer(results, correct_answers, correct_answers_passage_rel)
    # print(f"{ndcg}\n\n")
    # print(f"NDCG: {(sum(ndcg)/len(ndcg)):.2}")

    ndcg = ndcg_scorer_manual(results, correct_answers, correct_answers_passage_rel)
    print(f"nDCG (mean): {(sum(ndcg)/len(ndcg)):.2}")
    print(f"    Item-level: {ndcg}")



def rag_setup(config_name:str="default", api_key_variable:str="OPENAI_API_KEY"):
    vector_db_client, encoder = setup_vector_db(
        encoder_name=config[config_name]["encoder_name"],
        client_source=config[config_name]["client_source"],
        input_folder=config[config_name]["input_text_folder"],
        chunk_length=config[config_name]["chunk_length"],
        chunk_overlap_words=config[config_name]["chunk_overlap"],
        collection_name=config[config_name]["collection_name"],
        dist_name=config[config_name]["distance_type"],
        input_folder_qa=config[config_name]["input_folder_qa"],
        relevance_score_file_prefix=config[config_name]["relevance_score_file_prefix"],
        sample_qa_file=config[config_name]["sample_qa_file"])

    api_client = OpenAI(api_key=os.environ.get(api_key_variable))

    print("RAG setup complete")
    return vector_db_client, encoder, api_client


def rag_query_once(
    query:str, vector_db, encoder, api_client, config_name:str="default"):
    retrieved_doc_dict = query_vector_db_once(
        vector_db, encoder, query,
        collection_name = config[config_name]["collection_name"],
        k=config[config_name]["retrieve_k"])
    print("Documents retrieved")

    user_prompt = create_qa_string(query, retrieved_doc_dict["answers"])

    response = api_call(
        client=api_client, user_prompt=user_prompt,
        system_propmt_path=config[config_name]["llm_system_prompt_path"],
        model=config[config_name]["llm_model"],
        temperature=config[config_name]["llm_temperature"])

    return query, response


def rag_query_list(
    queries:list[str], vector_db, encoder, api_client, config_name:str="default"):
    responses = [rag_query_once(
        q, vector_db, encoder, api_client, config_name) for q in queries]

    return queries, responses


def export_qa_lists(queries:list[str], responses:list[str],
                    model:str, temperature:float, export_folder:str):
    qa_df = pd.DataFrame({"Query": queries, "Response": responses})

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    qa_df.to_csv(
        os.path.join(
            export_folder, f"QA_{model}_temp{temperature}_{timestamp}.csv"),
        index=False)

    return


if __name__ == "__main__":
    with open("rag-who.toml", mode="rb") as fp:
        config = tomllib.load(fp)

    parser=argparse.ArgumentParser(description="argument parser for rag-who")
    parser.add_argument("--config_name", nargs='?', default="default")
    args=parser.parse_args()

    # print(args.config_name)
    # retrieve_and_eval(config_name=args.config_name)
    vector_db_client, encoder, api_client = rag_setup()
    query, response = rag_query_once(
        "How long do rabbits live?",
        vector_db_client, encoder, api_client)
    print("\n", query, response, sep="\n")

    # print("#########################################")

    # response = rag_query_once(
    #     "How many deaths does alcoholism cause a year in the European Region?",
    #     vector_db_client, encoder, api_client)
    # print(response)
