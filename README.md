
![monochrome_threshold_transparent](https://github.com/EveryOneIsGross/holdingfaceSEARCH/assets/23621140/e3efb20e-ef22-41b1-99c6-a7577ea4d135)

This script allows you to search a Hugging Face dataset using either keyword or embedding-based search. 
The script can be run directly from the command line or imported into another script to call the `run_search` function.

```yaml
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
```
