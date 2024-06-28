# src/index.py

import faiss
import numpy as np

def create_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

if __name__ == "__main__":
    lecture_embeddings = np.load('data/lecture_embeddings.npy')
    papers_embeddings = np.load('data/papers_embeddings.npy')

    lecture_index = create_index(lecture_embeddings)
    papers_index = create_index(papers_embeddings)

    faiss.write_index(lecture_index, 'data/lecture_index.faiss')
    faiss.write_index(papers_index, 'data/papers_index.faiss')
