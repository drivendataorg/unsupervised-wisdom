
Tested on PC using a renv restored environment using R version 4.3.1. See Session Info for more information.

### Windows instructions for rendering notebook
1. Install quarto https://quarto.org/docs/download/
2. Download raw data and place in "data/raw" (see below)
2. Navigate to the project's root directory ("4th Place")
3. `quarto render "notebooks/unsupervised-wisdom-from-medical-narrative-report.qmd"`
4. Notebook should render in "reports"

### Required data

This is the data structure used by default in the notebook. Note that there are commented lines of code that need to be uncommented to use raw data rather than interim files.

```
data
├── interim
│   ├── clean_narrative_data.csv
│   └── embeddings.csv
│   └── embeddings_pca.csv
│       
└── raw
    ├── openai_embeddings_primary_narratives.parquet.gzip
    ├── primary_data.csv
    └── variable_mapping.json
```

Notes:
- `renv.lock` `.Rprofile`, `renv/settings.json` and `renv/activate.R` are necessary to restore the environment
- `_quarto.yml` and `unsupervised-wisdom-xvii.Rproj` ensure that the notebook is executed in the project root directory and outputs into the reports directory
- `Session Info.txt` lists information about the environment used to render the notebook
