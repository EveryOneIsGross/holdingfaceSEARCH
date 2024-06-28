
![monochrome_threshold_transparent](https://github.com/EveryOneIsGross/holdingfaceSEARCH/assets/23621140/e3efb20e-ef22-41b1-99c6-a7577ea4d135)

This script allows you to search a Hugging Face dataset using either keyword or embedding-based search. 
The script can be run directly from the command line or imported into another script to call the `run_search` function.

# Dataset Search Tool

## Overview

The Dataset Search Tool is a powerful Python script designed to search and analyze Hugging Face datasets. It supports both keyword-based and embedding-based searches across multiple fields, with the ability to cache datasets and embeddings for improved performance. The tool features syntax highlighting for JSON, XML, and HTML content, as well as a retro CGA-style color scheme for improved readability.

## Features

- Search Hugging Face datasets using keyword or embedding-based methods
- Support for searching across multiple fields with customizable weights
- Cache datasets and embeddings for faster subsequent searches
- Truncate output to a specified number of tokens
- Syntax highlighting for JSON, XML, and HTML content in search results
- Customizable search parameters including chunk size and number of results

## Requirements

- Python 3.6+
- Required Python packages:
  - datasets
  - gpt4all
  - scipy
  - tiktoken
  - tqdm
  - colorama
  - pygments
  - numpy

## Installation

1. Clone this repository or download the script.
2. Install the required packages:

```
pip install datasets gpt4all scipy tiktoken tqdm colorama pygments numpy
```

## Usage

Run the script from the command line with the following syntax:

```
python dataset_search_tool.py <dataset> <query> --search_keys <key1> <key2> ... --output_keys <key1> <key2> ... --search_type <type> --top_k <num> --chunk_size <size> --max_output_tokens <num>
```

### Arguments

- `<dataset>`: Hugging Face dataset location (e.g., username/dataset_name)
- `<query>`: Search query
- `--search_keys`: Space-separated list of keys to search against in the dataset
- `--output_keys`: Space-separated list of keys to include in the output
- `--search_type`: Type of search to perform (`embedding` or `keyword`)
- `--top_k`: Number of top results to return (default: 5)
- `--chunk_size`: Chunk size for embedding resolution (default: 100)
- `--max_output_tokens`: Maximum number of tokens for each output field (default: 200)

### Example

```
python dataset_search_tool.py glaiveai/RAG-v1 "What is value .csv table?" --search_keys answer question --output_keys question answer documents --search_type embedding --top_k 1 --chunk_size 2000 --max_output_tokens 100
```

## Output

The tool will display:
- Search parameters
- Progress of dataset loading and embedding (if applicable)
- Top results, including:
  - Combined relevance score
  - Full schema of the result (colorized JSON)
  - Formatted output with specified keys (truncated and syntax-highlighted)

## Multi-field Search

When searching across multiple fields:
- For embedding-based search, the tool combines embeddings from all specified fields and calculates a weighted average of cosine similarities.
- For keyword-based search, the tool performs individual searches on each field and combines the results, sorting by relevance.

Currently, equal weights are used for multiple fields. Future versions may allow custom weight specification.

## Caching

The tool caches datasets and embeddings to improve performance for subsequent searches. Cached files are stored in the same directory as the script.

## Note

This tool requires an internet connection to download datasets from Hugging Face. Ensure you have sufficient disk space for caching large datasets and their embeddings.
