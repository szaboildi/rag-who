
import pandas as pd
import time
import os

try:
    import tomllib # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib



def export_qa_lists(queries:list[str], responses:list[str],
                    model:str, temperature:float, export_folder:str):
    qa_df = pd.DataFrame({"Query": queries, "Response": responses})

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    qa_df.to_csv(
        os.path.join(
            export_folder, f"QA_{model}_temp{temperature}_{timestamp}.csv"),
        index=False)

    return


def is_relevant_sentence_str(result_str:str, correct_answers_sent:dict[str,str]) -> int:
    return int(any([
        corr_a in result_str for corr_a in correct_answers_sent["answers"]]))


def is_relevant_sentence_dict(
    result:dict[str,str], correct_answers_sent:dict[str,str]) -> dict[str,str]:
    # retreived_passages = [a["text"] for a in result["answers"]]
    for a in result["answers"]:
        # Iterating through the retrieved passages, find the first one
        # that's relevant, and add it's RR to the the accumulator list
        a["is_relevant"] = is_relevant_sentence_str(a["text"], correct_answers_sent)

    return result


def read_config(toml_path:str, config_name:str):
    with open(toml_path, "rb") as bf:
        config = tomllib.load(bf)

    """
    match config[config_name]:
        case {
            "input_text_folder": str(),
            "input_folder_qa": str(),
            "sample_qa_file": str(),
            "relevance_score_file_prefix": str(),
            "chunk_length": int(),
            "chunk_overlap": int(),
            "client_source": str() ":memory:"
            encoder_name                = "intfloat/e5-base"
            distance_type               = "COSINE"
            collection_name             = "who_guidelines"

            retrieve_k_pre_rank         = 8
            retrieve_k                  = 4
            sparse_retriever            = "BM25"

            llm_model                   = "gpt-4o-mini"
            llm_temperature             = 0.4
            llm_system_prompt_path      = "prompts/llm_system_prompt.txt"
            llm_user_prompt_template    = "prompts/llm_user_prompt_template_haystack.txt"
            QA_export_folder "
            }:
                pass
            case _:
                raise ValueError(f"invalid configuration: {config}")
    """

    return config
