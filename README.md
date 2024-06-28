# Natural Language Query Agent

## Overview
This project builds a Natural Language Query Agent to answer simple natural language queries over lecture notes and a table of LLM architectures.

## Project Structure
- `data/`: Contains input data files.
- `src/`: Contains the source code.
- `requirements.txt`: Lists the dependencies.
- `main.py`: The main script to run the project.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Prepare data:
   - Place `lecture_notes.txt` and `milestone_papers.txt` in the `data/` directory.
4. Run the main script: `python main.py`.

## Description
- `data_preparation.py`: Loads and preprocesses text data.
- `embeddings.py`: Generates embeddings for the text data using a pre-trained model.
- `index.py`: Creates and saves FAISS indexes for embeddings.
- `query_handler.py`: Processes queries and retrieves relevant information from indexes.
- `response_generator.py`: Generates a conversational response based on retrieved information.


# nl_query_agent
