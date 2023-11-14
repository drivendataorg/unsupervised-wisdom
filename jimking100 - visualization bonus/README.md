## README

1. Create and activate a conda environment
```
conda create -n unsupervised-wisdom python=3.10 -y
conda activate unsupervised-wisdom
```
2. Install the requirements using the provided requirements.txt file
```
pip install -r requirements.txt
```
3. Change the data directory to your local data directory in cell 1.
```
DATA_DIRECTORY = Path("YOUR DATA DIRECTORY HERE")
```
4. Run the first 6 cells of the notebook.
5. Spacy requires some configuration through the terminal the **first time** through the notebook.
- In the terminal perform the following:
- Navigate to the data directory and activate the environment.
- Ensure the provided `base_config.cfg` file is in the data directory.
- Run the following commands to configure and train the spacy model:
```
python -m spacy init fill-config base_config.cfg config.cfg
python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./val.spacy
```
6. You can now run the entire notebook as often as you'd like from top to bottom.

**Note:** This code has only been tested on a Mac. Spacy configuration may be machine dependent; more details on spacy config can be found at https://spacy.io/usage/training
