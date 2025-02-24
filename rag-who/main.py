import os
import json
from retrieval import setup_vector_db, query_vector_db_list
from eval import eval_recall_sentence, eval_recall_passage, eval_mrr_sentence, ndcg_scorer_manual

if __name__ == "__main__":
    with open(os.path.join("data", "sample_qa.json")) as f:
        correct_answers = json.load(f)
    input_queries = [item["question"] for item in correct_answers]

    client, encoder = setup_vector_db()
    print("Vector database set up")
    results = query_vector_db_list(client, encoder, input_queries)
    print("Quesries completed")

    with open(os.path.join("data", "sample_qa_passage_lvl_length150_overlap15.json")) as f:
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
