import os
import json
from retreival import setup_vector_db, query_vector_db_list

def eval_recall_sentence(
    results:list[dict[str,str]], correct_answers_sent:list[dict[str,str]]):
    recalls = []
    not_found_answers = []

    for i in range(len(results)):
        retreived_passages = [a["text"] for a in results[i]["answers"]]
        not_found_answers_i = [
            corr_a for corr_a in correct_answers_sent[i]["answers"]
            if not any([corr_a in r for r in retreived_passages])]

        recall = sum([
            any([corr_a in r
                 for r in retreived_passages])
            for corr_a in correct_answers_sent[i]["answers"]]) / \
                len(correct_answers_sent[i]["answers"])

        recalls.append(recall)
        not_found_answers.append(not_found_answers_i)

    return recalls, not_found_answers


def eval_recall_passage(
    results:list[dict[str,str]], correct_answers_passages:list[dict[str,str]]):
    recalls = []
    not_found_answers = []

    for i in range(len(results)):
        retreived_passages = [a["text"] for a in results[i]["answers"]]
        not_found_answers_i = [
            corr_a for corr_a in correct_answers_passages[i]["answer_passages"]
            if not corr_a in retreived_passages]

        recall = len([
            corr_a for corr_a in correct_answers_passages[i]["answer_passages"]
            if corr_a in retreived_passages]) / \
                len(correct_answers_passages[i]["answer_passages"])

        recalls.append(recall)
        not_found_answers.append(not_found_answers_i)

    return recalls, not_found_answers


def eval_mrr_sentence(
    results:list[dict[str,str]], correct_answers_sent:list[dict[str,str]]):
    mrr = []
    for i in range(len(correct_answers_sent)):
        retreived_passages = [a["text"] for a in results[i]["answers"]]
        for j in range(len(retreived_passages)):
            # Iterating through the retrieved passages, find the first one
            # that's relevant, and add it's RR to the the accumulator list
            if any([
                corr_a in retreived_passages[j]
                for corr_a in correct_answers_sent[i]["answers"]]):
                mrr.append(1/(j+1))
                break
        # If none of the retreived passages were relevant
        # (if we haven't added an RR value to the list yet)
        if len(mrr) < i+1:
            mrr.append(0)
    return mrr

def is_relevant_sentence_str(result_str:str, correct_answers_sent:dict[str,str]) -> int:
    return int(any([
        corr_a in result_str for corr_a in correct_answers_sent["answers"]]))
