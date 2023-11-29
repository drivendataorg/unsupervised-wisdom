## Setup

To replicate the solution, follow these steps:

1.  Install the necessary Python packages by running the following command:

``` bash
pip install -r requirements.txt
```

2.  Configure the data directory in `notebooks/notebook.ipynb` using the WD variable.

3.  Download the required datasets and place them in the data directory.

4.  To label the data with ChatGPT, assign your OpenAI API key to the `openai.api_key` variable in the notebook file. You can find detailed information on obtaining and using the OpenAI key in the OpenAI API documentation <https://platform.openai.com/docs/api-reference>.

5.  Run the code in the `notebooks/notebook.ipynb` file. It contains three main parts:

-   **Preprocessing**: loading the full NIESS database of narratives and preprocessing it.

-   **Labeling**: labeling the sample with ChatGPT API, fitting the SVM model and predicting the labels for the full dataset.

-   **Analysis**: analysing the impact of obtained categories on the probability of falling and post-fall hospitalization.

## Required datasets

-   `primary_data.csv` - official dataset for the competition.

-   full dataset of NIESS narratives from 2013 to 2022 - available for download at <https://www.cpsc.gov/cgibin/NEISSQuery/>

-   `variable_mapping.json` - mapping of categories in the NIESS database (provided for the competition)

## Full NIESS database

folder `fulldb` contains all the narratives from the NIESS database for the 2013-2022 period.

## Intermediate outputs

After running the notebook, the folder `data/intermediate_outputs` will contain the following files:

1.  Samples labeled by ChatGPT:

*df_sample_full.obj* - sample of narratives from the full NIESS database labeled by ChatGPT.

*df_sample_home.obj* - sample of narratives from the NIESS database with 'HOME' location labeled by ChatGPT.

*df_sample_public.obj* - sample of narratives from the NIESS database with 'PUBLIC' location labeled by ChatGPT.

*df_lowest_activity.obj* - sample of narratives with the lowest predicted probability for the variable 'Activity' relabeled by ChatGPT.

*df_lowest_cause.obj* - sample of narratives with the lowest predicted probability for the variable 'Cause' relabeled by ChatGPT.

*df_lowest_facility.obj* - sample of narratives with the lowest predicted probability for the variable 'Facility' relabeled by ChatGPT.

*df_lowest_fall.obj* - sample of narratives with the lowest predicted probability for the variable 'Fall' relabeled by ChatGPT.

*df_lowest_homeloc.obj* - sample of narratives with the lowest predicted probability for the variable 'Home Location' relabeled by ChatGPT.

*df_lowest_position.obj* - sample of narratives with the lowest predicted probability for the variable 'Position' relabeled by ChatGPT.

2.  Samples сhecked by human coders:

*df_humancoded_sample_1.xlsx* - sample of labeled narratives сhecked by a human coder #1.

*df_humancoded_sample_2.xlsx* - sample of labeled narratives сhecked by a human coder #2.

3.  Estimates from the models:

*df_est_fall.obj* - estimates from the model 'probability of falling vs. encountering other incidents'.

*df_est_hospitd.obj* - estimates from the model 'post-falling risk of hospitalization'.

4.  Versions of the full dataset:

*df_embs.obj* - embeddings of narratives from the NIESS database.

*df_full_pp.obj* - full dataset of narratives from the NIESS database after the initial preprocessing.

*df_full_coded.obj* - full dataset of narratives with labeled variables 'Fall', 'Activity', 'Position', 'Cause'.

*df_home_coded.obj* - dataset of narratives with 'HOME' location with labeled variable 'Home Location'.

*df_home_coded.obj* - dataset of narratives with 'PUBLIC' location with labeled variable 'Facility'.

*df_full_imp.obj* - full dataset of narratives with all labeled variables and imputation of the 'not defined' categories.

---
Here is a tree of what the `data` directory looks like:
```
data
├── fulldb
│   ├── NEISS_2013.XLSX
│   ├── NEISS_2014.XLSX
│   ├── NEISS_2015.XLSX
│   ├── NEISS_2016.XLSX
│   ├── NEISS_2017.XLSX
│   ├── NEISS_2018.XLSX
│   ├── NEISS_2019.XLSX
│   ├── NEISS_2020.XLSX
│   ├── NEISS_2021.XLSX
│   └── NEISS_2022.XLSX
├── intermediate_outputs
│   ├── df_embs.obj
│   ├── df_est_fall.obj
│   ├── df_est_hospitd.obj
│   ├── df_full_coded.obj
│   ├── df_full_imp.obj
│   ├── df_full_pp.obj
│   ├── df_home_coded.obj
│   ├── df_humancoded_sample_1.xlsx
│   ├── df_humancoded_sample_2.xlsx
│   ├── df_lowest_activity.obj
│   ├── df_lowest_cause.obj
│   ├── df_lowest_facility.obj
│   ├── df_lowest_fall.obj
│   ├── df_lowest_homeloc.obj
│   ├── df_lowest_position.obj
│   ├── df_public_coded.obj
│   ├── df_sample_full.obj
│   ├── df_sample_home.obj
│   └── df_sample_public.obj
├── primary_data.csv
└── variable_mapping.json
```
