# src/query_handler.py

from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

def query_embedding(query, model, tokenizer):
    query_inputs = tokenizer(query, return_tensors="pt")
    query_outputs = model(**query_inputs)
    query_embedding = query_outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return query_embedding

def search_index(query_embedding, index, top_k=5):
    D, I = index.search(query_embedding, top_k)
    return I

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    lecture_index = faiss.read_index('data/lecture_index.faiss')
    papers_index = faiss.read_index('data/papers_index.faiss')

    query = "What are some milestone model architectures and papers in the last few years?"
    query_emb = query_embedding(query, model, tokenizer)

    lecture_results = search_index(query_emb, lecture_index)
    papers_results = search_index(query_emb, papers_index)

    np.save('data/lecture_results.npy', lecture_results)
    np.save('data/papers_results.npy', papers_results)

    print("Lecture results:", lecture_results)
    print("Papers results:", papers_results)
