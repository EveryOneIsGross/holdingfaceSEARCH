import argparse
import os
import json
import re
from datetime import datetime
from openai import OpenAI
from holdingface import run_search

def parse_arguments():
    parser = argparse.ArgumentParser(description='Interactive dataset search and LLM conversation')
    parser.add_argument('dataset', help='Hugging Face dataset location (e.g., username/dataset_name)')
    parser.add_argument('--search_keys', nargs='+', required=True, help='Keys to search against (space-separated)')
    parser.add_argument('--output_keys', nargs='+', required=True, help='Output keys (space-separated)')
    parser.add_argument('--search_type', choices=['embedding', 'keyword'], required=True, help='Search type (embedding/keyword)')
    parser.add_argument('--top_k', type=int, default=1, help='Number of top results to return')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for embedding resolution')
    parser.add_argument('--max_output_tokens', type=int, default=200, help='Maximum number of tokens for each output field')
    return parser.parse_args()

def capture_console_output(func):
    import io
    import sys

    def wrapper(*args, **kwargs):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        result = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return result, captured_output.getvalue()

    return wrapper

def extract_output_keys_content(output, output_keys):
    results = []
    matches = re.findall(r"(Top \d+ Result:\n)(.*?)(?=Top \d+ Result:|$)", output, re.DOTALL)
    for match in matches:
        content = {}
        for key in output_keys:
            pattern = rf"{key.capitalize()}: (.*?)(?:\n\n|\Z)"
            key_matches = re.findall(pattern, match[1], re.DOTALL)
            if key_matches:
                content[key] = key_matches[0].strip()
        results.append(content)
    return results

def prepare_context(user_input, extracted_contents, search_type, search_keys, output_keys):
    context = f"""User Query: {user_input}

Search Information:
- Search Type: {search_type}
- Search Keys: {', '.join(search_keys)}
- Output Keys: {', '.join(output_keys)}

Relevant Information from Dataset:
"""
    for i, extracted_content in enumerate(extracted_contents, 1):
        context += f"\nResult {i}:\n"
        for key in output_keys:
            if key in extracted_content:
                context += f"{key.capitalize()}: {extracted_content[key]}\n\n"
    return context

def main():
    args = parse_arguments()

    print("Debug - Parsed arguments:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Search keys: {args.search_keys}")
    print(f"  Output keys: {args.output_keys}")
    print(f"  Search type: {args.search_type}")
    print(f"  Top K: {args.top_k}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Max output tokens: {args.max_output_tokens}")

    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )

    conversation_log = []
    log_filename = f"conversation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print("\nWelcome to the interactive dataset search and LLM conversation!")
    print("Type 'exit' to end the conversation.")

    captured_run_search = capture_console_output(run_search)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        print(f"Debug - Search query: {user_input}")
        print(f"Debug - Using top_k: {args.top_k}")

        _, search_output = captured_run_search(
            args.dataset,
            user_input,
            args.search_keys,
            args.output_keys,
            args.search_type,
            args.top_k,
            args.chunk_size,
            args.max_output_tokens
        )

        extracted_contents = extract_output_keys_content(search_output, args.output_keys)
        context = prepare_context(user_input, extracted_contents, args.search_type, args.search_keys, args.output_keys)

        print("Debug - Extracted content:", json.dumps(extracted_contents, indent=2))
        print("Debug - Prepared context:", context)

        messages = [
            {"role": "system", "content": """You are a helpful assistant with access to a specific dataset. 
Your task is to answer the user's question based on the provided context from the dataset search.
Follow these guidelines:
1. Use the information from the dataset to formulate your response.
2. If the dataset doesn't contain relevant information, say so and offer a general response.
3. Cite specific parts of the dataset when answering, referencing the specific output keys.
4. If multiple output keys are relevant, synthesize the information from all of them.
5. Always maintain a helpful and informative tone.

Here's the context from the dataset search:"""},
            {"role": "system", "content": context},
            {"role": "user", "content": user_input}
        ]

        response = client.chat.completions.create(
            model="gemma2:latest",
            messages=messages
        )

        assistant_response = response.choices[0].message.content
        print("\nAssistant:", assistant_response)

        conversation_turn = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "extracted_content": extracted_contents,
            "context": context,
            "assistant_response": assistant_response
        }
        conversation_log.append(conversation_turn)

        with open(log_filename, 'w') as f:
            json.dump(conversation_log, f, indent=2)

    print(f"\nConversation ended. Log saved to {log_filename}")

if __name__ == "__main__":
    main()
