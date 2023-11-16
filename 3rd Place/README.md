
Tested on a clean conda environment with python3.9 and faiss(installed from conda)

### Install
1. conda create --name fall  python=3.9
2. conda activate fall
3.   pip install -r requirements.txt
##### Ubuntu
4. conda install -c pytorch faiss-cpu
##### windows  
4. conda install -c conda-forge faiss 
   #(using the other conda channel in Windows causes missing dlls)

Interactive Graph image snapshots were saved and loaded. To render them the notebook needs to be run 
To run the notebook ensure file paths and your openai key are specified. i.e

primary_path =  'path to primary_data.csv'
variable_mapping = 'path to variable_mapping.json'
embedding_path = 'path to \ openai_embeddings_primary_narratives.parquet.gzip'
openai_key_uk =   'your openai key' 

##### Directory:
assets : contains saved graph images

response : directory for storing Opeanai chatGpt3.5 responses
##### File:
openai_precipitating.json : All Opeanai chatGpt3.5 responses assembled in a single json file.