import numpy as np
import pandas as pd
import json
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fixed_token_chunker import FixedTokenChunker


#Embedding Function
def calculate_embeddings(text_batches, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    "Calculate embeddings for a batch of texts"
    model = SentenceTransformer(model_name)
    return model.encode(text_batches, show_progress_bar=False)


#Tokenizer length function
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
length_fn = lambda x: len(tokenizer.encode(x))


#Precision & Recall
def precision_recall(predicted_chunks, golden_text):
    """Compute precision and recall at token level"""
    pred_tokens = set(tokenizer.encode(predicted_chunks))
    golden_tokens = set(tokenizer.encode(golden_text))
    if not pred_tokens or not golden_tokens:
        return 0.0, 0.0

    intersection = pred_tokens & golden_tokens
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(golden_tokens)
    return precision, recall


#Main Pipeline
def evaluate_pipeline(corpus, queries, labels, chunk_size, chunk_overlap, top_k,
                      model_name='sentence-transformers/all-MiniLM-L6-v2'):
    # Chunk corpus
    chunker = FixedTokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_fn,
        keep_separator=False,
        add_start_index=False,
        strip_whitespace=True,
    )
    chunked_text = chunker.split_text(corpus)

    # Embeddings
    chunk_embeddings = calculate_embeddings(chunked_text, model_name)
    query_embeddings = calculate_embeddings(queries, model_name)

    # Retrieval
    cosine_sim = cosine_similarity(query_embeddings, chunk_embeddings)
    top_k_indices = np.argsort(cosine_sim, axis=1)[:, -top_k:][:, ::-1]  # descending order

    precision_list, recall_list = [], []

    for i, indices in enumerate(top_k_indices):
        retrieved_chunks = " ".join([chunked_text[idx] for idx in indices])
        precision, recall = precision_recall(retrieved_chunks, labels[i])
        precision_list.append(precision)
        recall_list.append(recall)

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "num_chunks": len(chunked_text)
    }


#%% Load Data

df = pd.read_csv('questions_df.csv')
df = df[df['corpus_id'] == 'wikitexts']
queries = df['question'].tolist()

labels = []
for ref in df['references'].apply(json.loads):
    combined = " ".join([x['content'] for x in ref])
    labels.append(combined)

with open("wikitexts.md", "r", encoding="utf-8") as file:
    corpus = file.read()

#%% Hyperparameter Sweeping

chunk_sizes = [200, 400]
overlaps = [50, 100]
top_ks = [1, 5, 10]

results = []

for size in chunk_sizes:
    for overlap in overlaps:
        for top_k in top_ks:
            result = evaluate_pipeline(
                corpus=corpus,
                queries=queries,
                labels=labels,
                chunk_size=size,
                chunk_overlap=overlap,
                top_k=top_k
            )
            print(result)
            results.append(result)

#%% Save Results

results_df = pd.DataFrame(results)
results_df.to_csv("retrieval_results.csv", index=False)
print("\nFinal Results:\n", results_df.sort_values(by="avg_precision", ascending=False))
