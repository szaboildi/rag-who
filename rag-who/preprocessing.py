import re

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


def process_text(path, length:int=200, words_overlap:int=20):
    clean_text = read_clean_text(path)
    chunks = chunk_text(clean_text, length, words_overlap)

    return chunks


def main():
    import os

    chunks = process_text(os.path.join("data", "alcohol-use.txt"))
    # print(chunks[-3])
    # print(f"#################\n{chunks[-2]}")
    # print(f"#################\n{chunks[-1]}")

    print(len(chunks))

if __name__ == "__main__":
    main()
