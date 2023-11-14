## Configuration for Jim King's Notebook:

1. Install the environment using the provided requirements.txt file
**conda install --file requirements.txt**
2. Change the data directory to your local data directory in cell 1
**DATA_DIRECTORY = Path("YOUR DATA DIRECTORY HERE")**
3. Run the first 6 cells of the notebook.
4. Spacy requires some configuration through the terminal the **first time** through the notebook.
- In the terminal perform the following:
- Navigate to the data directory and activate the environment.
- Copy the provided base_config.cfg file to the data directory.
- Run the following commands to configure and train the spacy model:
**python -m spacy init fill-config base_config.cfg config.cfg**
**python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./val.spacy**
5. You can now run the entire notebook as often as you'd like from top to bottom.
 
**Notes:**  I believe the spacy configuration is machine dependent, but have no way of testing this on other machines (I'm using a Mac).  If the above process does not work it may be possible to simply provide all the spacy files generated during config and copy them to your data directory - let me know if these steps don't work.  Details on spacy config can be found at https://spacy.io/usage/training



