
import pandas as pd
import time
import os



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
