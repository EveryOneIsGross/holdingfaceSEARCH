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
import re
from pygments import highlight
from pygments.lexers import JsonLexer, XmlLexer, HtmlLexer
from pygments.formatters import TerminalFormatter
import numpy as np

# Initialize colorama
colorama.init(autoreset=True)

# CGA-style color definitions
CYAN = Fore.CYAN + Style.BRIGHT
MAGENTA = Fore.MAGENTA + Style.BRIGHT
WHITE = Fore.WHITE + Style.BRIGHT
YELLOW = Fore.YELLOW + Style.BRIGHT
GREEN = Fore.GREEN + Style.BRIGHT
RED = Fore.RED + Style.BRIGHT
BLUE = Fore.BLUE + Style.BRIGHT

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

def keyword_search(dataset, query, search_keys, top_k):
    query_keywords = query.lower().split()
    keyword_counts = []

    for example in dataset:
        total_count = 0
        for search_key in search_keys:
            if search_key in example:
                value = str(example[search_key]).lower()
                keyword_count = sum(keyword in value for keyword in query_keywords)
                total_count += keyword_count
        keyword_counts.append((total_count, example))

    keyword_counts.sort(reverse=True, key=lambda x: x[0])
    top_results = keyword_counts[:top_k]

    return top_results

def truncate_text(text, max_tokens):
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)
    
    last_sentence_end = max(
        truncated_text.rfind('.'),
        truncated_text.rfind('!'),
        truncated_text.rfind('?')
    )
    
    if last_sentence_end != -1 and last_sentence_end > len(truncated_text) * 0.5:
        return truncated_text[:last_sentence_end + 1]
    else:
        last_space = truncated_text.rfind(' ')
        if last_space != -1:
            return truncated_text[:last_space]
        return truncated_text

def detect_and_highlight(text):
    try:
        if isinstance(text, dict):
            text = json.dumps(text, indent=2)
        
        json.loads(text)
        return highlight(text, JsonLexer(), TerminalFormatter())
    except (json.JSONDecodeError, TypeError):
        pass

    if isinstance(text, str) and text.strip().startswith('<') and text.strip().endswith('>'):
        if re.match(r'<\?xml', text):
            return highlight(text, XmlLexer(), TerminalFormatter())

    if isinstance(text, str) and re.search(r'<(!DOCTYPE|html|body|head)', text, re.IGNORECASE):
        return highlight(text, HtmlLexer(), TerminalFormatter())

    return text

def format_output(schema, output_keys, max_tokens=100):
    output = ""
    for key in output_keys:
        if key in schema:
            text = schema[key]
            truncated_text = truncate_text(text, max_tokens)
            highlighted_text = detect_and_highlight(truncated_text)
            output += f"{YELLOW}{key.capitalize()}: {WHITE}{highlighted_text}\n\n"
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

def cosine_search_multi_key(embedded_dataset, user_embedding, top_k, weights=None):
    cosine_distances = []

    for key, data in embedded_dataset.items():
        # Check if 'embedding' or 'embeddings' key exists
        if 'embedding' in data:
            distances = [cosine(user_embedding, data['embedding'])]
        elif 'embeddings' in data:
            distances = [cosine(user_embedding, emb) for emb in data['embeddings']]
        else:
            continue  # Skip this item if no embedding is found

        if weights:
            weighted_distance = sum(d * w for d, w in zip(distances, weights))
        else:
            weighted_distance = sum(distances) / len(distances)
        cosine_distances.append((weighted_distance, key, data['full_schema']))

    cosine_distances.sort()
    top_results = cosine_distances[:top_k]

    return top_results

def run_search(dataset_location, query=None, search_keys=None, output_keys=None, search_type=None, top_k=5, chunk_size=100, max_output_tokens=200):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(script_dir, f"{dataset_location.replace('/', '_')}_cached_dataset.json")
    embedded_cache_file = os.path.join(script_dir, f"{dataset_location.replace('/', '_')}_{'_'.join(search_keys)}_embedded_dataset.pkl")

    cached_dataset = load_cached_dataset(cache_file)

    if cached_dataset is None:
        print(CYAN + "Loading dataset...")
        dataset = load_dataset(dataset_location)
        dataset_list = list(dataset['train'])
        print(CYAN + "Saving cached dataset...")
        save_cached_dataset(dataset_list, cache_file)
    else:
        dataset_list = cached_dataset

    if not search_keys or not output_keys or not search_type:
        print(MAGENTA + "Available keys:")
        keys = list(dataset_list[0].keys())
        for key in keys:
            print(WHITE + f"- {key}")
        return

    if search_type.lower() == 'embedding':
        embedder = Embed4All()
        if not os.path.exists(embedded_cache_file):
            embedded_dataset = {}
            print(CYAN + "Embedding dataset...")
            for index, example in enumerate(tqdm(dataset_list, desc="Embedding")):
                embedded_dataset[str(index)] = {'full_schema': example}
                for search_key in search_keys:
                    if search_key in example:
                        value = example[search_key]
                        chunks = chunk_text(value, chunk_size)
                        if len(chunks) == 1:
                            embedded_dataset[str(index)]['embedding'] = embedder.embed(chunks[0]['text'])
                        else:
                            embedded_dataset[str(index)]['embeddings'] = [embedder.embed(chunk['text']) for chunk in chunks]
            print(CYAN + "Saving embedded dataset...")
            with open(embedded_cache_file, 'wb') as file:
                pickle.dump(embedded_dataset, file)
        else:
            print(CYAN + "Loading embedded dataset...")
            with open(embedded_cache_file, 'rb') as file:
                embedded_dataset = pickle.load(file)

    if search_type.lower() == 'keyword':
        print(CYAN + "Performing keyword search...")
        top_results = keyword_search(dataset_list, query, search_keys, top_k)
    elif search_type.lower() == 'embedding':
        print(CYAN + "Performing embedding search...")
        query_embedding = embedder.embed(query)
        weights = [1.0 / len(search_keys)] * len(search_keys)  # Equal weights for now
        top_results = cosine_search_multi_key(embedded_dataset, query_embedding, top_k, weights)
    else:
        raise ValueError("Invalid search type. Please enter 'embedding' or 'keyword'.")

    if top_results:
        for i, result in enumerate(top_results, start=1):
            if search_type.lower() == 'keyword':
                count, matched_schema = result
                print(MAGENTA + f"Top {i} Result:")
                print(GREEN + f"Keyword Count: {count}")
            else:
                distance, key, matched_schema = result
                print(MAGENTA + f"Top {i} Result:")
                print(GREEN + f"Distance: {distance}")
            
            print(YELLOW + "Full Schema:")
            schema_json = json.dumps(matched_schema, indent=2)
            highlighted_schema = detect_and_highlight(schema_json)
            print(highlighted_schema)
            
            print(BLUE + "Formatted Output:")
            truncated_output = format_output(matched_schema, output_keys, max_output_tokens)
            print(truncated_output)
            print()
    else:
        print(RED + "No matching results found.")

    return top_results

if __name__ == '__main__':
    print(CYAN + Style.BRIGHT + "Dataset Search Tool")
    print(MAGENTA + "-------------------")

    parser = argparse.ArgumentParser(description='Search Hugging Face dataset')
    parser.add_argument('dataset', help='Hugging Face dataset location (e.g., username/dataset_name)')
    parser.add_argument('query', nargs='?', default=None, help='Search query')
    parser.add_argument('--search_keys', nargs='+', help='Keys to search against (space-separated)')
    parser.add_argument('--output_keys', nargs='+', help='Output keys (space-separated)')
    parser.add_argument('--search_type', choices=['embedding', 'keyword'], help='Search type (embedding/keyword)')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to return')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for embedding resolution')
    parser.add_argument('--max_output_tokens', type=int, default=200, help='Maximum number of tokens for each output field')
    args = parser.parse_args()

    print(GREEN + f"Searching dataset: {args.dataset}")
    print(GREEN + f"Query: {args.query}")
    print(GREEN + f"Search type: {args.search_type}")
    print(GREEN + f"Search keys: {', '.join(args.search_keys)}")
    print()

    run_search(args.dataset, args.query, args.search_keys, args.output_keys, args.search_type, args.top_k, args.chunk_size, args.max_output_tokens)
