import os
import json
from eval import is_relevant_sentence_str


def read_clean_text(path):
    with open(path, "r") as f:
        raw_text = f.read()

    text = raw_text.strip().replace("\n\n", "\n")
    return text

def chunk_text(input_text:str, length:int, words_overlap:int):
    words = input_text.split(" ")
    # Chunking
    buffer = []
    chunks = []
    for i in range(0, len(words), length-words_overlap):
        for word in words[i:]:
            buffer.append(word)
            if ('!' in word or '.' in word or "?" in word) and len(buffer) > length:
                chunks.append(' '.join(buffer))
                buffer = []
                break

    # chunks = [" ".join(words[i:i+length])
    #           for i in range(0, len(words), length-words_overlap)]

    return chunks


def process_text(path, length:int=200, words_overlap:int=20,
                 return_format:str="ls", force_write_qa_passages:bool=False):
    clean_text = read_clean_text(path)
    chunks = chunk_text(clean_text, length, words_overlap)

    # Check if the relevance scores and chunk-level answers have already been exported,
    # if so, no need to recalculate relevance scores (unless forced)
    json_out_location = os.path.join(
        "data", f"sample_qa_passage_lvl_length{length}_overlap{words_overlap}.json")
    if force_write_qa_passages or not os.path.exists(json_out_location):
        print(f"Relevance scores don't exist for these chunks, calculating them now")
        corr_answer_ls_dict = []
        json_location = os.path.join("data", "sample_qa.json")
        with open(json_location) as f:
            correct_answers = json.load(f)

        for corr_answer_dict in correct_answers:
            relevant_passages = [
                chunk for chunk in chunks
                if is_relevant_sentence_str(chunk, corr_answer_dict)]

            corr_answer_dict_passages = {
                "question": corr_answer_dict["question"],
                "answer_passages": relevant_passages,
                "ideal_relevance": [1] * len(relevant_passages) +\
                    [0] * (len(chunks)-len(relevant_passages))
            }
            corr_answer_ls_dict.append(corr_answer_dict_passages)

        with open(json_out_location, "w", encoding="utf-8") as f:
            json.dump(corr_answer_ls_dict, f, indent=4)

    if return_format=="ls":
        return chunks
    # Return a list of dictionaries
    if return_format=="ls_dict":
        return [{"text": chunk} for chunk in chunks]


def main():
    import os

    chunks = process_text(os.path.join("data", "alcohol-use.txt"))
    # print(chunks[-3])
    # print(f"#################\n{chunks[-2]}")
    # print(f"#################\n{chunks[-1]}")

    print(len(chunks))

if __name__ == "__main__":
    main()



# TODO parametrize
# QA json input
# QA json output
