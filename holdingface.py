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
import argparse
import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init(autoreset=True)

# Custom color definitions
LIGHT_ORANGE = Fore.YELLOW + Style.BRIGHT
ORANGE = Fore.YELLOW
DARK_ORANGE = Fore.RED + Fore.YELLOW
DARKER_ORANGE = Fore.RED + Style.DIM
BURNT_ORANGE = Fore.RED

# Static value for initial embedding size
INITIAL_EMBEDDING_SIZE = 16

# Global tokenizer instance
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
            value = str(example[search_key]).lower()  # Convert to string to handle non-string types
            keyword_count = sum(keyword in value for keyword in query_keywords)
            keyword_counts.append((keyword_count, value, example))

    keyword_counts.sort(reverse=True, key=lambda x: x[0])  # Sort by keyword_count
    top_results = keyword_counts[:top_k]

    return top_results

def cosine_search(embedded_dataset, user_embedding, top_k):
    cosine_distances = []

    for key, data in embedded_dataset.items():
        distance = cosine(user_embedding, data['embedding'])
        cosine_distances.append((distance, key, data['full_schema']))

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
    start_idx = 0
    current_char_position = 0

    while start_idx < len(tokens):
        if start_idx + chunk_size >= len(tokens):
            end_idx = len(tokens)
        else:
            end_idx = start_idx + chunk_size
            while end_idx > start_idx and not tokenizer.decode(tokens[end_idx - 1:end_idx]).isspace():
                if tokenizer.decode(tokens[end_idx - 1:end_idx]) in ',.?!':
                    end_idx += 1
                    break
                end_idx -= 1
            if end_idx == start_idx:
                end_idx = start_idx + chunk_size

        chunk = tokenizer.decode(tokens[start_idx:end_idx])
        end_char_position = current_char_position + len(chunk)
        chunks.append({
            'text': chunk,
            'start_pos': current_char_position,
            'end_pos': end_char_position - 1
        })

        current_char_position = end_char_position
        start_idx = end_idx

    return chunks

def run_search(dataset_location, query=None, search_key=None, output_keys=None, search_type=None, top_k=5, chunk_size=100):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(script_dir, f"{dataset_location.replace('/', '_')}_cached_dataset.json")
    embedded_cache_file = os.path.join(script_dir, f"{dataset_location.replace('/', '_')}_{search_key}_embedded_dataset.pkl")

    cached_dataset = load_cached_dataset(cache_file)

    if cached_dataset is None:
        print(ORANGE + "Loading dataset...")
        dataset = load_dataset(dataset_location)
        dataset_list = list(dataset['train'])
        print(ORANGE + "Saving cached dataset...")
        save_cached_dataset(dataset_list, cache_file)
    else:
        dataset_list = cached_dataset

    if not search_key or not output_keys or not search_type:
        print(LIGHT_ORANGE + "Available keys:")
        keys = list(dataset_list[0].keys())
        for key in keys:
            print(ORANGE + f"- {key}")
        return

    if search_type.lower() == 'embedding':
        if not os.path.exists(embedded_cache_file):
            embedder = Embed4All()
            embedded_dataset = {}
            print(ORANGE + "Embedding dataset...")
            for index, example in enumerate(tqdm(dataset_list, desc="Embedding")):
                if search_key in example:
                    value = example[search_key]
                    chunks = chunk_text(value, chunk_size)
                    for i, chunk in enumerate(chunks):
                        chunk_embedding = embedder.embed(chunk['text'])
                        embedded_dataset[f"{index}_{i}"] = {
                            'embedding': chunk_embedding,
                            'full_schema': example
                        }
            print(ORANGE + "Saving embedded dataset...")
            with open(embedded_cache_file, 'wb') as file:
                pickle.dump(embedded_dataset, file)
        else:
            print(ORANGE + "Loading embedded dataset...")
            with open(embedded_cache_file, 'rb') as file:
                embedded_dataset = pickle.load(file)

        embedder = Embed4All(device='gpu')

    if search_type.lower() == 'keyword':
        print(ORANGE + "Performing keyword search...")
        top_results = keyword_search(dataset_list, query, search_key, top_k)
    elif search_type.lower() == 'embedding':
        print(ORANGE + "Performing embedding search...")
        top_results = []
        query_chunks = chunk_text(query, chunk_size)
        for query_chunk in tqdm(query_chunks, desc="Searching"):
            user_embedding = embedder.embed(query_chunk['text'])
            chunk_results = cosine_search(embedded_dataset, user_embedding, top_k)
            top_results.extend(chunk_results)
        top_results = sorted(top_results, key=lambda x: x[0])[:top_k]
    else:
        raise ValueError("Invalid search type. Please enter 'embedding' or 'keyword'.")

    if top_results:
        #print(LIGHT_ORANGE + "Search Results:")
        for i, result in enumerate(top_results, start=1):
            distance, key, matched_schema = result
            print(ORANGE + Style.BRIGHT + f"Top {i} Result:")
            #print(DARK_ORANGE + f"Matched Value: {key}")
            print(DARKER_ORANGE + f"Distance/Count: {distance}")
            print(BURNT_ORANGE + "Full Schema:")
            print(Style.DIM + json.dumps(matched_schema, indent=2))
            print(LIGHT_ORANGE + "Formatted Output:")
            output = format_output(matched_schema, output_keys, chunk_size)
            print(ORANGE + output)
            print()
    else:
        print(BURNT_ORANGE + "No matching results found.")

if __name__ == '__main__':
    print(LIGHT_ORANGE + Style.BRIGHT + "Dataset Search Tool")
    print(ORANGE + "-------------------")

    parser = argparse.ArgumentParser(description='Search Hugging Face dataset')
    parser.add_argument('dataset', help='Hugging Face dataset location (e.g., username/dataset_name)')
    parser.add_argument('query', nargs='?', default=None, help='Search query')
    parser.add_argument('--search_key', help='Key to search against')
    parser.add_argument('--output_keys', nargs='+', help='Output keys (space-separated)')
    parser.add_argument('--search_type', choices=['embedding', 'keyword'], help='Search type (embedding/keyword)')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to return')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for embedding resolution and output formatting')
    args = parser.parse_args()

    print(DARK_ORANGE + f"Searching dataset: {args.dataset}")
    print(DARK_ORANGE + f"Query: {args.query}")
    print(DARK_ORANGE + f"Search type: {args.search_type}")
    print()

    run_search(args.dataset, args.query, args.search_key, args.output_keys, args.search_type, args.top_k, args.chunk_size)
