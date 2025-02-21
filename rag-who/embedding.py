
from sentence_transformers import SentenceTransformer

from preprocessing import process_text
import os
import json


# Create input texts (queries & passages)
### Each input text should start with "query: " or "passage: ".
### For tasks other than retrieval, you can simply use the "query: " prefix.
input_passages = [
    f"passage: {t}"
    for t in process_text(
        os.path.join("data", "alcohol-use.txt"),
        length=100, words_overlap=15)]

with open(os.path.join("data", "sample_qa.json")) as f:
    d = json.load(f)
input_queries = [f"query: {item['question']}" for item in d]

input_texts = input_queries + input_passages


# Encode input texts
encoder = SentenceTransformer('intfloat/e5-base')
embeddings = encoder.encode(input_texts, normalize_embeddings=True)

# Print scores
scores = (embeddings[:6] @ embeddings[6:].T) * 100

# print(scores.tolist())
print([max(s) for s in scores])



# def average_pool(
#     last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
#     last_hidden = \
#         last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
# tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
# model = AutoModel.from_pretrained('intfloat/e5-base-v2')

# # Tokenize the input texts
# batch_dict = tokenizer(
#     input_texts, max_length=512,
#     padding=True, truncation=True, return_tensors='pt')
# # QUESTION what is return_tensors arg and what other options are there? PyTorch
# # # print(batch_dict)
# # print(len(input_passages), len(input_texts))
# # print(max([len(i.split(" ")) for i in input_texts]))
# # print(batch_dict["input_ids"].shape)
# # # QUESTION what are token type ids?
# # print(batch_dict["token_type_ids"].shape)
# # print(batch_dict["attention_mask"].shape)


# outputs = model(**batch_dict)
# embeddings = average_pool(
#     outputs.last_hidden_state, batch_dict['attention_mask'])

# # normalize embeddings
# embeddings = F.normalize(embeddings, p=2, dim=1)
# scores = (embeddings[:2] @ embeddings[2:].T) * 100
# print(scores.tolist())
# print(scores.shape)
