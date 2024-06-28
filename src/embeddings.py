# src/embeddings.py

from transformers import AutoTokenizer, AutoModel
import numpy as np

def generate_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    with open('data/processed_lecture_notes.txt', 'r') as file:
        lecture_notes = file.read()

    with open('data/processed_milestone_papers.txt', 'r') as file:
        milestone_papers = file.read()

    lecture_embeddings = generate_embeddings(lecture_notes, model, tokenizer)
    papers_embeddings = generate_embeddings(milestone_papers, model, tokenizer)

    np.save('data/lecture_embeddings.npy', lecture_embeddings)
    np.save('data/papers_embeddings.npy', papers_embeddings)
