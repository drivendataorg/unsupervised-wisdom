# Unsupervised Wisdom Challenge Midpoint Bonus Prize

## Summary of my approach

**My approach uses ChatGPT and Co-occurrence Network.**
1. Extract actions from Narrative by ChatGPT.
2. Visualize the relationships between fall and preceding action using a co-occurrence network.
3. Infer the fundamental cause of the fall.

Our approach is based on the hypothesis that a preceding action caused the fall, a concept unique to our narrative. While we have information about the object that caused the fall, this alone cannot infer the fundamental cause of the fall, which is critical for devising fall prevention measures. Therefore, my focus was on the question, 'What action led to the fall?' To gather this information, I used the OpenAI API to extract actions through ChatGPT. With these results, I analyzed the primary data alongside other columns to gain deeper insights.

## Key findings
1. The action that caused the fall is useful as information specific to Narrative.
2. It became clear that using ChatGPT is more advantageous than conventional methods for extracting that action.
3. Based on the combination of this action information and other columns, there is a suggestion that a more detailed analysis for each target seems to be possible.

## Setup

My environment uses `Python 3.8.8`.

Install packages.
```bash 
pip install -r requirements.txt
```

Run the notebook.
```bash 
jupyter lab
```

Set your OpenAI API Key in `main.ipynb`
```python
import os
os.environ["OPENAI_API_KEY"] = "<API Key>"
```

## Directory structure
```bash
├── README.md
├── main.ipynb
├── dataset
│   ├── primary_data.csv
│   └── variable_mapping.json
└── output
    └── chatgpt_verbs_300.csv
```