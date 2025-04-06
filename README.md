# RAG Pipeline Evaluation

This project implements a Retrieval-Augmented Generation (RAG) pipeline using open-source embedding models, chunking strategies, and evaluation metrics. It is built on the **Wikitext** dataset and evaluates how well the pipeline retrieves relevant context using metrics like **Precision**, **Recall**, and **IoU**.


## Project Overview

The main steps of the pipeline include:

1. **Dataset Preparation**  
   Using the Wikitext corpus along with a set of labeled queries and their relevant excerpts.

2. **Chunking**  
   Implemented with the `FixedTokenChunker`, splitting the corpus into overlapping token-based chunks.

3. **Embedding**  
   Embeddings are generated for both user queries and corpus chunks using the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model.

4. **Retrieval**  
   For each query, the top-k most similar chunks are retrieved using cosine similarity on embedding vectors.

5. **Evaluation**  
   Precision, Recall, and IoU are calculated to assess retrieval quality against the gold-labeled relevant excerpts.


