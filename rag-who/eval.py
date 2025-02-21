import os
import json
from retreival import setup_vector_db, query_vector_db_list

def eval_recall(results:list[dict[str,str]],
                correct_answers:list[dict[str,str]]):
    recalls = []
    not_found_answers = []

    for i in range(len(results)):
        retreived_passages = [a["text"] for a in results[i]["answers"]]
        not_found_answers_i = [
            corr_a for corr_a in correct_answers[i]["answers"]
            if not any([corr_a in r for r in retreived_passages])]

        recall = sum([
            any([corr_a in r
                 for r in retreived_passages])
            for corr_a in correct_answers[i]["answers"]]) / \
                len(correct_answers[i]["answers"])

        recalls.append(recall)
        not_found_answers.append(not_found_answers_i)

    return recalls, not_found_answers

def eval_mrr(
    results:list[dict[str,str]], correct_answers:list[dict[str,str]]):
    mrr = []
    for i in range(len(correct_answers)):
        retreived_passages = [a["text"] for a in results[i]["answers"]]
        for j in range(len(retreived_passages)):
            # Iterating through the retrieved passages, find the first one
            # that's relevant, and add it's RR to the the accumulator list
            if any([
                corr_a in retreived_passages[j]
                for corr_a in correct_answers[i]["answers"]]):
                mrr.append(1/(j+1))
                break
        # If none of the retreived passages were relevant
        # (if we haven't added an RR value to the list yet)
        if len(mrr) < i+1:
            mrr.append(0)
    return mrr

if __name__ == "__main__":
    with open(os.path.join("data", "sample_qa.json")) as f:
        correct_answers = json.load(f)
    input_queries = [item["question"] for item in correct_answers]

    client, encoder = setup_vector_db()
    results = query_vector_db_list(client, encoder, input_queries)

    rc = eval_recall(results, correct_answers)
    print(f"Recall: {sum(rc[0]) / len(rc[0]):.2%}")

    mrr = eval_mrr(results, correct_answers)
    print(mrr)
    print(f"MRR: {(sum(mrr)/len(mrr)):.2}")
