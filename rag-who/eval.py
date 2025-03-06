import math
import os
import json
# from sklearn.metrics import ndcg_score

from retrieval import setup_vector_db
from qdrant_pipeline import query_vector_db_list_qdrant
from haystack_pipeline import retrieval_pipeline_haystack, query_vector_db_list_haystack
from utils import is_relevant_sentence_dict

try:
    import tomllib # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib



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


# def ndcg_scorer(
#     results:list[dict[str,str]], correct_answers_sent:list[dict[str,str]],
#     correct_answers_passage_rel:list[dict[str,str]]):
#     ndcg_scores = []
#     for i in range(len(results)):
#         y_returned_relevance = np.asanyarray([[
#             a["is_relevant"] for a in is_relevant_sentence_dict(
#                 results[i], correct_answers_sent[i])["answers"]]])

#         n_passages_retrieved = len(y_returned_relevance[0])
#         y_ideal_relevance = \
#             np.asanyarray([
#                 correct_answers_passage_rel[i]["ideal_relevance"][:n_passages_retrieved]])

#         print(f"relevance scores: {y_returned_relevance}\n" +
#               f"ideal relevance: {y_ideal_relevance}\n" +
#               f"ndcg: {round(ndcg_score(y_ideal_relevance, y_returned_relevance, k=n_passages_retrieved), 5)}\n#################")
#         ndcg_scores.append(round(ndcg_score(
#             y_ideal_relevance, y_returned_relevance, k=n_passages_retrieved), 5))

#     return ndcg_scores


def ndcg_manual(
    returned_rel:list[int], ideal_rel:list[int]):
    dc_gain = 0
    idc_gain = 0
    for idx, value in enumerate(returned_rel):
        dc_gain += value / math.log2(idx + 2)
        # if idx + 1 == k:
        #     break
    for idx, value in enumerate(ideal_rel):
        idc_gain += value / math.log2(idx + 2)
        # if idx + 1 == k:
        #     break
    return round(dc_gain / idc_gain, 5), round(dc_gain, 5), round(idc_gain, 5)


def ndcg_scorer_manual(
    results:list[dict[str,str]], correct_answers_sentences:list[dict[str,str]],
    correct_answers_passage_rel:list[dict[str,str]]):
    # dcg_scores = []
    ndcg_scores = []
    for i in range(len(results)):
        y_ideal_relevance = correct_answers_passage_rel[i]["ideal_relevance"]
        y_returned_relevance = [a["is_relevant"] for a in
           is_relevant_sentence_dict(results[i], correct_answers_sentences[i])["answers"]]
        n_passages_retrieved = len(y_returned_relevance)
        # # Pad the true relevance to have 0 for passages beyond the retrieveal limit
        # # (non-retrieved passages)
        # y_true_relevance += [0] * (len(y_ideal_relevance) - n_passages_retrieved)

        ndcg, dcg, idcg = \
            ndcg_manual(y_returned_relevance, y_ideal_relevance[:n_passages_retrieved])
        # print(f"relevance scores: {y_returned_relevance}\n" +
        #       f"ideal relevance: {y_ideal_relevance[:n_passages_retrieved]}\n" +
        #       f"ndcg: {ndcg}\n#################")
        ndcg_scores.append(ndcg)
        # dcg_scores.append(dcg)

    return ndcg_scores



def retrieve_and_eval(config_name:str="default", mode="qdrant"):
    with open("rag-who.toml", mode="rb") as fp:
        config = tomllib.load(fp)

    with open(os.path.join(config[config_name]["input_folder_qa"],
                           config[config_name]["sample_qa_file"])) as f:
        correct_answers = json.load(f)
    input_queries = [item["question"] for item in correct_answers]\

    if mode=="qdrant":
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
            sample_qa_file=config[config_name]["sample_qa_file"], mode="qdrant")
        results = query_vector_db_list_qdrant(
            client, encoder, input_queries,
            collection_name = config[config_name]["collection_name"],
            k=config[config_name]["retrieve_k"])
        print("Queries completed")

    elif mode=="haystack":
        retrieval_pipeline = retrieval_pipeline_haystack(
            config, config_name=config_name)
        results = query_vector_db_list_haystack(
            retrieval_pipeline, input_queries,
            dist_name=config[config_name]["distance_type"])

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

if __name__ == "__main__":
    retrieve_and_eval(mode="haystack")
    retrieve_and_eval(mode="qdrant")
