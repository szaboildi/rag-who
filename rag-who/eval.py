import math
# from sklearn.metrics import ndcg_score



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


def is_relevant_sentence_dict(
    result:dict[str,str], correct_answers_sent:dict[str,str]) -> dict[str,str]:
    # retreived_passages = [a["text"] for a in result["answers"]]
    for a in result["answers"]:
        # Iterating through the retrieved passages, find the first one
        # that's relevant, and add it's RR to the the accumulator list
        a["is_relevant"] = is_relevant_sentence_str(a["text"], correct_answers_sent)

    return result


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
