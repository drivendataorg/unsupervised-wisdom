
Tested on a clean conda environment with python3.9 and faiss (installed from conda)

### Install
1. `conda create --name fall python=3.9`
2. `conda activate fall`
3. `pip install -r requirements.txt`
##### Ubuntu
4. `conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl`
##### windows  
4. `conda install -c conda-forge faiss` (using the other conda channel in Windows causes missing dlls)

### Specify data paths and OpenAI Key

To run the notebook ensure the following file paths and your openai key are specified. i.e

```
primary_path =  'path to primary_data.csv'
variable_mapping = 'path to variable_mapping.json'
embedding_path = 'path to \ openai_embeddings_primary_narratives.parquet.gzip'
openai_key_uk =   'your openai key'
```

This is the data structure used by default in the notebook.

```
data
├── interim
│   ├── openai_precipitating.json
│   └── response
│       └── response_0.json
└── raw
    ├── openai_embeddings_primary_narratives.parquet.gzip
    ├── primary_data.csv
    └── variable_mapping.json
```

Notes:
- `data/interim/response`: directory for storing Opeanai chatGpt3.5 responses
- `data/interim/openai_precipitating.json`: All Opeanai chatGpt3.5 responses assembled in a single json file.
- `notebooks/images`: contains saved graph images (interactive Graph image snapshots were saved and loaded; to render them the notebook needs to be run)
