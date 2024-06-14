"""
This script allows you to search a Hugging Face dataset using either keyword or embedding-based search. 
The script can be run directly from the command line or imported into another script to call the `run_search` function.

Usage:
    Command Line:
        python script.py dataset query --search_key search_key --output_keys output_key1 output_key2 --search_type search_type --top_k top_k --chunk_size chunk_size

    Import and Call:
        from your_script import run_search
        results = run_search('username/dataset_name', 'search query', 'search_key', ['output_key1', 'output_key2'], 'embedding', top_k=3)

Arguments:
    dataset (str): 
        Hugging Face dataset location (e.g., username/dataset_name).

    query (str, optional):
        Search query. If not provided, the script will print the available keys from the dataset.

    --search_key (str):
        Key to search against in the dataset.

    --output_keys (list of str):
        List of output keys (space-separated) to include in the search results.

    --search_type (str):
        Type of search to perform. Must be either 'embedding' or 'keyword'.

    --top_k (int, default=5):
        Number of top results to return.

    --chunk_size (int, default=100):
        Chunk size for text chunking when performing embedding search.

Examples:
    Command Line:
        python script.py zhengyun21/PMC-Patients "Chest pain" --search_key "patient" --output_keys "patient" "title" "age" --search_type keyword --top_k 3

    Import and Call:
        from your_script import run_search
        results = run_search('zhengyun21/PMC-Patients', 'Chest pain', 'patient', ['patient', 'title', 'age'], 'keyword', top_k=3)

Functions:
    load_cached_dataset(cache_file):
        Load cached dataset from a file.

    save_cached_dataset(dataset, cache_file):
        Save dataset to a cache file.

    keyword_search(dataset, query, search_key, top_k):
        Perform keyword search on the dataset.

    cosine_search(embedded_dataset, user_embedding, top_k):
        Perform embedding-based search on the dataset.

    format_output(schema, output_keys):
        Format the output results based on specified keys.

    chunk_text(text, chunk_size):
        Chunk the text into smaller pieces based on the specified chunk size.

    run_search(dataset_location, query, search_key, output_keys, search_type, top_k=5, chunk_size=100):
        Main function to perform the search on the specified dataset.

"""

import os
import json
import pickle
from datasets import load_dataset
from gpt4all import Embed4All
from scipy.spatial.distance import cosine
import tiktoken
from tqdm import tqdm

# Static value for initial embedding size
INITIAL_EMBEDDING_SIZE = 16

# Global or higher scope initialization
tokenizer = tiktoken.get_encoding("cl100k_base")

def load_cached_dataset(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as file:
            return json.load(file)
    else:
        return None

def save_cached_dataset(dataset, cache_file):
    with open(cache_file, 'w') as file:
        json.dump(dataset, file)

def keyword_search(dataset, query, search_key, top_k):
    query_keywords = query.lower().split()
    keyword_counts = []

    for example in dataset:
        if search_key in example:
            value = example[search_key].lower()
            keyword_count = sum(keyword in value for keyword in query_keywords)
            keyword_counts.append((keyword_count, example[search_key], example))

    keyword_counts.sort(reverse=True)
    top_results = keyword_counts[:top_k]

    return top_results

def cosine_search(embedded_dataset, user_embedding, top_k):
    cosine_distances = []

    for value, data in embedded_dataset.items():
        distance = cosine(user_embedding, data['embedding'])
        cosine_distances.append((distance, value, data['full_schema']))

    cosine_distances.sort()
    top_results = cosine_distances[:top_k]

    return top_results

def format_output(schema, output_keys, chunk_size=100):
    output = ""
    for key in output_keys:
        if key in schema:
            text = schema[key]
            tokens = tokenizer.encode(text)
            chunks = [tokenizer.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
            for chunk in chunks:
                output += f"{key.capitalize()}: {chunk}\n"
            output += "\n"  # Add a newline for separation between different keys
    return output

def chunk_text(text, chunk_size):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    
    while start < len(tokens):
        if start + chunk_size >= len(tokens):
            # If remaining tokens are fewer than the chunk size, take all remaining tokens
            end = len(tokens)
        else:
            # Otherwise, try to find the best place to split
            end = start + chunk_size
            while end > start and not tokenizer.decode(tokens[end - 1:end]).isspace():
                if tokenizer.decode(tokens[end - 1:end]) in ',.?!':
                    # If a punctuation mark is found, break after this point
                    end += 1
                    break
                end -= 1

            # If no punctuation is found, and we're back at start, enforce the chunk size limit
            if end == start:
                end = start + chunk_size

        # Decode the chunk and add it to the list
        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        start = end

    return chunks


def run_search(dataset_location, query=None, search_key=None, output_keys=None, search_type=None, top_k=5, chunk_size=100):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(script_dir, f"{dataset_location.replace('/', '_')}_cached_dataset.json")
    embedded_cache_file = os.path.join(script_dir, f"{dataset_location.replace('/', '_')}_{search_key}_embedded_dataset.pkl")

    # Check if the cached dataset exists
    cached_dataset = load_cached_dataset(cache_file)

    if cached_dataset is None:
        print("Loading dataset...")
        # Load the dataset
        dataset = load_dataset(dataset_location)
        # Convert the dataset to a list for caching
        dataset_list = list(dataset['train'])
        # Save the cached dataset
        print("Saving cached dataset...")
        save_cached_dataset(dataset_list, cache_file)
    else:
        dataset_list = cached_dataset

    if not search_key or not output_keys or not search_type:
        # Print the available keys if search_key, output_keys, or search_type is not provided
        print("Available keys:")
        keys = list(dataset_list[0].keys())
        for key in keys:
            print(f"- {key}")
        return

    if search_type.lower() == 'embedding':
        # Check if the embedded dataset exists
        if not os.path.exists(embedded_cache_file):
            # Create an instance of the embedder
            embedder = Embed4All()

            # Embed the values in the dataset and create a dictionary
            embedded_dataset = {}
            print("Embedding dataset...")
            for example in tqdm(dataset_list, desc="Embedding"):
                if search_key in example:
                    value = example[search_key]
                    chunks = chunk_text(value, chunk_size)  # Use the specified chunk size
                    for chunk in chunks:
                        chunk_embedding = embedder.embed(chunk)
                        embedded_dataset[chunk] = {
                            'embedding': chunk_embedding,
                            'full_schema': example
                        }

            # Save the embedded dataset as a pickle file
            print("Saving embedded dataset...")
            with open(embedded_cache_file, 'wb') as file:
                pickle.dump(embedded_dataset, file)
        else:
            # Load the embedded dataset from the pickle file
            print("Loading embedded dataset...")
            with open(embedded_cache_file, 'rb') as file:
                embedded_dataset = pickle.load(file)

        # Create an instance of the embedder
        embedder = Embed4All(device='gpu')

    if search_type.lower() == 'keyword':
        print("Performing keyword search...")
        top_results = keyword_search(dataset_list, query, search_key, top_k)
    elif search_type.lower() == 'embedding':
        print("Performing embedding search...")
        top_results = []
        query_chunks = chunk_text(query, chunk_size)
        for query_chunk in tqdm(query_chunks, desc="Searching"):
            user_embedding = embedder.embed(query_chunk)
            chunk_results = cosine_search(embedded_dataset, user_embedding, top_k)
            top_results.extend(chunk_results)
        top_results = sorted(top_results, key=lambda x: x[0])[:top_k]
    else:
        raise ValueError("Invalid search type. Please enter 'embedding' or 'keyword'.")

    if top_results:
        print("Search Results:")
        for i, result in enumerate(top_results, start=1):
            if search_type.lower() == 'keyword':
                keyword_count, matched_value, matched_schema = result
            elif search_type.lower() == 'embedding':
                distance, matched_value, matched_schema = result

            print(f"Top {i} Result:")
            print("Matched Value:")
            print(matched_value)
            print()

            print("Full Schema:")
            print(json.dumps(matched_schema, indent=2))
            print()

            print("Formatted Output:")
            output = format_output(matched_schema, output_keys, chunk_size)  # Ensure chunk_size is passed correctly
            print(output)
            print()

    else:
        print("No matching results found.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Search Hugging Face dataset')
    parser.add_argument('dataset', help='Hugging Face dataset location (e.g., username/dataset_name)')
    parser.add_argument('query', nargs='?', default=None, help='Search query')
    parser.add_argument('--search_key', help='Key to search against')
    parser.add_argument('--output_keys', nargs='+', help='Output keys (space-separated)')
    parser.add_argument('--search_type', choices=['embedding', 'keyword'], help='Search type (embedding/keyword)')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to return')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for embedding resolution and output formatting')
    args = parser.parse_args()

    run_search(args.dataset, args.query, args.search_key, args.output_keys, args.search_type, args.top_k, args.chunk_size)
