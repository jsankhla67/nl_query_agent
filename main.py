import os

# Set environment variable to handle OpenMP runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.data_preparation import preprocess_text, load_text
from src.embeddings import generate_embeddings
from src.index import create_index
from src.query_handler import query_embedding, search_index
from src.response_generator import generate_response

import faiss
from transformers import AutoTokenizer, AutoModel
import numpy as np

def main():
    # Load and preprocess text
    lecture_notes = preprocess_text(load_text('data/lecture_notes.txt'))
    milestone_papers = preprocess_text(load_text('data/milestone_papers.txt'))

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    # Generate embeddings
    lecture_embeddings = generate_embeddings(lecture_notes, model, tokenizer)
    papers_embeddings = generate_embeddings(milestone_papers, model, tokenizer)

    # Create and save indexes
    lecture_index = create_index(lecture_embeddings)
    papers_index = create_index(papers_embeddings)

    faiss.write_index(lecture_index, 'data/lecture_index.faiss')
    faiss.write_index(papers_index, 'data/papers_index.faiss')

    # Query handling
    query = "What are some milestone model architectures and papers in the last few years?"
    query_emb = query_embedding(query, model, tokenizer)

    lecture_results = search_index(query_emb, lecture_index)
    papers_results = search_index(query_emb, papers_index)

    np.save('data/lecture_results.npy', lecture_results)
    np.save('data/papers_results.npy', papers_results)

    # Generate response
    response = generate_response(lecture_results, lecture_notes.splitlines(), milestone_papers.splitlines())
    print("Response:", response)

if __name__ == "_main_":
    main()
