import os
# import json
import argparse
from openai import OpenAI

from ragwho.embedding import setup_vector_db

# from pydantic import BaseModel


try:
    import tomllib # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib



def create_qa_string(question:str, answers:list[str])->str:
    user_prompt = f"Answer the following question:\n<Question>{question}\n</Question>\n\n"
    user_prompt += "<Context>\n"

    for i, answer in enumerate(answers):
        user_prompt += f"<Document{i+1}>{answer}</Document{i+1}>\n\n"

    user_prompt += "</Context>"

    return user_prompt


# class QuestionAnswering(BaseModel):
#     answer_to_question: str


def api_call(client:OpenAI, user_prompt:str,
             system_propmt_path:str, model="gpt-4o-mini",
             temperature:float=0):
    # client = OpenAI(
    #     api_key=os.environ.get("OPENAI_API_KEY"))

    with open(system_propmt_path, "r") as f:
        system_prompt = f.read()

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        # response_format=QuestionAnswering,
        model=model,
        temperature=temperature
    )

    return chat_completion.choices[0].message.content


def query_vector_db_once_qdrant(
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


def query_vector_db_list_qdrant(
    client, encoder, question_list:list[str],
    collection_name:str="dummy_name", k:int=5):
    answer_list = [
        query_vector_db_once_qdrant(
            client, encoder, q, collection_name, k) for q in question_list]

    return answer_list


def rag_setup_qdrant(
    config_name:str="default", api_key_variable:str="OPENAI_API_KEY",
    qdrant_cloud_api_key_variable:str="QDRANT_CLOUD_API_KEY",
    from_scratch:bool=False):
    vector_db_client, encoder = setup_vector_db(
        encoder_name=config[config_name]["encoder_name"],
        client_source=config[config_name]["client_source"],
        qdrant_cloud_api_key=os.environ.get(qdrant_cloud_api_key_variable),
        from_scratch=from_scratch,
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


def rag_query_once_qdrant(
    query:str, vector_db, encoder, api_client, config_name:str="default"):
    retrieved_doc_dict = query_vector_db_once_qdrant(
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


def rag_query_list_qdrant(
    queries:list[str], vector_db, encoder, api_client, config_name:str="default"):
    responses = [rag_query_once_qdrant(
        q, vector_db, encoder, api_client, config_name)[1] for q in queries]

    return queries, responses



if __name__ == "__main__":
    with open("parameters.toml", mode="rb") as fp:
        config = tomllib.load(fp)

    parser=argparse.ArgumentParser(description="argument parser for rag-who")
    parser.add_argument("--config_name", nargs='?', default="default")
    args=parser.parse_args()
    print("Arguments parsed, parameters loaded")

    # print(args.config_name)
    # retrieve_and_eval(config_name=args.config_name)
    # vector_db_client, encoder, gen_api_client = rag_setup_qdrant(
    #     config_name=args.config_name, from_scratch=True)

    vector_db_client, encoder, gen_api_client = rag_setup_qdrant(
        config_name=args.config_name, from_scratch=False)
    query, response = rag_query_once_qdrant(
        "How long do rabbits live?",
        vector_db_client, encoder, gen_api_client)
    print("\n", query, response, sep="\n")

    # print("#########################################")

    # response = rag_query_once_qdrant(
    #     "How many deaths does alcoholism cause a year in the European Region?",
    #     vector_db_client, encoder, api_client)
    # print(response)
