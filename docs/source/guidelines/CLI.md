# Command Line Interface

* Successful installation will expose `docs-cli` entrypoint, a `Typer` application.
* Start with `generate-dataset` command to create JSON dataset containing source documents for retrieval.
* Follow it up with `generate-database` command to create vector database for document embeddings.
* Start querying the database with `answer-query` command.
* Use `--help` to know available flags and default settings.
* The response shall contain the following:
    * original input
    * model response
    * generation time
    * retrieved documents

## Example

### Dataset Generation

```console
$ docs-cli generate-dataset generative_ai --dataset-file "retrieval_dataset.json"
Dataset generation complete: '/full/path/to/working/directory/retrieval_dataset.json'.
```

### Database Generation

```console
$ docs-cli generate-database --dataset-file "retrieval_dataset.json" --embedding-model "all-mpnet-base-v2" --database-directory "vector_database"
Database generation complete: '/full/path/to/working/directory/vector_database'.
```

### Response Generation

#### Retrieval with Similarity using Flan T5

```console
$ docs-cli answer-query "How many modules are there in information retrieval sub-package?" --embedding-model "all-mpnet-base-v2" --database-directory "vector_database" --search-type "similarity" --number-of-documents 5 --language-model-type "standard_transformers" --standard-pipeline-type "text2text-generation" --standard-model-name "google/flan-t5-large"
Query: How many modules are there in information retrieval sub-package?
Answer: 5
Duration: 1.69 seconds
Source 1: 'information_retrieval' package has 5 many modules.
Source 2: Modules of 'information_retrieval' package are as follows: 1. orchestrate_retrieval 2. step_1_retrieval 3. step_2_retrieval 4. step_3_retrieval 5. utils_retrieval.
Source 3: 'information_retrieval' package has 22 many public exports.
Source 4: Hierarchy of 'information_retrieval' package is as follows: 1. generative_ai 2. information_retrieval.
Source 5: Hierarchy of 'utils_retrieval' module is as follows: 1. generative_ai 2. information_retrieval 3. utils_retrieval.
```

#### Retrieval with Similarity using Quantised Llama 2

```console
$ docs-cli answer-query "What are different types of retrieval supported by this package?" --embedding-model "all-mpnet-base-v2" --database-directory "vector_database" --search-type "similarity" --number-of-documents 5 --language-model-type "quantised_ctransformers" --quantised-model-name "TheBloke/Llama-2-7B-GGUF" --quantised-model-file "llama-2-7b.Q4_K_M.gguf" --quantised-model-type "llama"
Query: What are different types of retrieval supported by this package?
Answer: MMR and Similarity.

Duration: 22.97 seconds
Source 1: The following is the documentation of 'RetrievalType' object: 'Define supported retrieval types.'.
Source 2: 'information_retrieval' package has 5 many modules.
Source 3: Names of different members of 'RetrievalType' enum are as follows: 1. MMR 2. SIMILARITY.
Source 4: Names of different members of 'RetrievalType' enum are as follows: 1. MMR 2. SIMILARITY.
Source 5: The following is the documentation of 'information_retrieval' package: 'Define functionalities for information retrieval.'.
```

#### Retrieval with Maximum Marginal Reference using Quantised Mistral

```console
$ docs-cli answer-query "List public exports of dataset generation package." --embedding-model "all-mpnet-base-v2" --database-directory "vector_database" --search-type "mmr" --number-of-documents 5 --initial-number-of-documents 10 --language-model-type "quantised_ctransformers" --quantised-model-name "TheBloke/Mistral-7B-v0.1-GGUF" --quantised-model-file "mistral-7b-v0.1.Q4_K_M.gguf" --quantised-model-type "mistral"
Query: List public exports of dataset generation package.
Answer: The following is the list of public exports of 'dataset_generation' package: 1. JSONDataset 2. JSONDocument 3. generate_json_dataset 4. generate_member_dataset 5. generate_module_dataset 6. generate_package_dataset 7. generate_raw_datasets 8. get_all_member_details 9. get_all_module_contents 10. get_all_package_contents 11. load_json_dataset 12. store_json_dataset.
Duration: 57.91 seconds
Source 1: 'dataset_generation' package has 12 many public exports.
Source 2: Documentation of 'generate_package_dataset' function lacks any examples.
Source 3: The following is the documentation of 'generate_raw_datasets' object: 'Generate all retrieval and tuning documents for exploring documentation of a package.

Parameters
----------
package_name : str
    name of the root package to import with

Returns
-------
list[Dataset]
    all retrieval and tuning documents for root package and its contents'.
Source 4: 'dataset_generation' package exports following public members using __all__: 1. JSONDataset 2. JSONDocument 3. generate_json_dataset 4. generate_member_dataset 5. generate_module_dataset 6. generate_package_dataset 7. generate_raw_datasets 8. get_all_member_details 9. get_all_module_contents 10. get_all_package_contents 11. load_json_dataset 12. store_json_dataset.
Source 5: Modules of 'dataset_generation' package are as follows: 1. orchestrate_generation 2. step_1_generation 3. step_2_generation 4. utils_generation.
```
