# Large Language Models and Topic Modelling for Textual Information Extraction and Analysis

**This project achieved the 2nd place overall and the bonus prize for most helpful shared code on DrivenData's [Unsupervised Wisdom: Explore Medical Narratives on Older Adult Falls](https://www.drivendata.org/competitions/217/cdc-fall-narratives/) competition!!!**

This repository contains the code used for my submission for DrivenData's [Unsupervised Wisdom: Explore Medical Narratives on Older Adult Falls](https://www.drivendata.org/competitions/217/cdc-fall-narratives/) competition. The goal of the challenge was to identify effective methods of using unsupervised machine learning to extract insights about older adult falls from emergency department narratives.

My submission, titled "Large Language Models and Topic Modelling for Textual Information Extraction and Analysis" explored the usage of Text2Text Generation transformers with big narratives (a custom prompt in human-readable format that uses a combination of tabular data and the available narratives). The big narratives provide the transformers with a more complete understanding of what happened. The big narratives have six different questions appended to them, creating the final prompts used to automatically extract certain textual information present in the narratives, such as precipitating events, activity involved and a more thorough diagnosis of the event. Finally, I also employ topic modelling (with LDA) to create categories from the information that was extracted from the narratives, allowing for an easy way to explore the data and find correlations.

### Repository structure

- `submission.ipynb`: Jupyter notebook file containing my complete analysis thoroughly explained. I recommend using the following [nbviewer link](https://nbviewer.org/github/zysymu/unsupervised-wisdom/blob/main/submission.ipynb) to visualize it online.
- `Summary_Marcos_Tidball.pdf`: an executive summary explaining my analysis and demonstrating how it can be applied.
- `data`: directory containing all of the data used.
    - `official`: directory containing data provided by the competition hosts.
    - `intermediate`: directory containing intermediate data generated with `main.ipynb`.

### Instructions for running the code

I use the `pyenchant` library, which requires you to install the enchant C library. To run the code you first need to `git clone` this repository. After going inside the repository you can do the following:

Linux:
```
apt-get install python3-enchant
pip install -r requirements.txt
```

MacOS with Apple Silicon (note that after the `export` command you need to quit and reopen your terminal):
```
brew install enchant
export PYENCHANT_LIBRARY_PATH=/opt/homebrew/lib/libenchant-2.2.dylib
pip install -r requirements.txt
```