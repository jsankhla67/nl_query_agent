# src/response_generator.py

import numpy as np

def generate_response(results, lecture_text, papers_text):
    lecture_snippets = [lecture_text[i] for i in results[0]]
    papers_snippets = [papers_text[i] for i in results[1]]
    response = " ".join(lecture_snippets + papers_snippets)
    return response

if __name__ == "__main__":
    with open('data/processed_lecture_notes.txt', 'r') as file:
        lecture_text = file.read().splitlines()

    with open('data/processed_milestone_papers.txt', 'r') as file:
        papers_text = file.read().splitlines()

    lecture_results = np.load('data/lecture_results.npy')
    papers_results = np.load('data/papers_results.npy')

    response = generate_response(lecture_results, lecture_text, papers_text)
    print("Response:", response)
