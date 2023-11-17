## Setup

To replicate the solution, follow these steps:

1.  Install the necessary Python packages by running the following command:

``` bash
pip install -r requirements.txt
```

2.  Configure the working directory in *notebook.ipynb* using the WD variable.

3.  Download the required datasets and place them in the working directory.

4.  To label the data with ChatGPT, assign your OpenAI API key to the openai.api_key variable in the *notebook.ipynb* file. You can find detailed information on obtaining and using the OpenAI key in the OpenAI API documentation <https://platform.openai.com/docs/api-reference>.

5.  Run the code in the *notebook.ipynb* file. It contains three main parts:

-   'Preprocessing': loading the full NIESS database of narratives and preprocessing it.

-   'Labeling': labeling the sample with ChatGPT API, fitting the SVM model and predicting the labels for the full dataset.

-   'Analysis': analysing the impact of obtained categories on the probability of falling and post-fall hospitalization.

## Required datasets

-   *primary_data.csv* - official dataset for the competition.

-   full dataset of NIESS narratives from 2013 to 2022 - available for download at <https://www.cpsc.gov/cgibin/NEISSQuery/>

-   *variable_mapping.json* - mapping of categories in the NIESS database.

## Intermediate outputs

folder *intermediate_outputs* contains the following files:

*df_sample_full.obj* - sample of narratives from the full NIESS database labeled by ChatGPT API.

*df_sample_home.obj* - sample of narratives from the NIESS database with 'HOME' location labeled by ChatGPT API.

*df_sample_facility.obj* - sample of narratives from the NIESS database with 'FACILITY' location labeled by ChatGPT API.

*df_lowest_activity.obj* - sample of narratives with the lowest predicted probability for the variable 'Activity' relabeled by ChatGPT API.

*df_lowest_cause.obj* - sample of narratives with the lowest predicted probability for the variable 'Cause' relabeled by ChatGPT API.

*df_lowest_facility.obj* - sample of narratives with the lowest predicted probability for the variable 'Facility' relabeled by ChatGPT API.

*df_lowest_fall.obj* - sample of narratives with the lowest predicted probability for the variable 'Fall' relabeled by ChatGPT API.

*df_lowest_homeloc.obj* - sample of narratives with the lowest predicted probability for the variable 'Home Location' relabeled by ChatGPT API.

*df_lowest_position.obj* - sample of narratives with the lowest predicted probability for the variable 'Position' relabeled by ChatGPT API.

*df_humancoded_sample_1.xlsx* - sample of labeled narratives сhecked by a human coder #1.

*df_humancoded_sample_2.xlsx* - sample of labeled narratives сhecked by a human coder #2.

*df_est_fall.obj* - estimates from the model 'probability of falling vs. encountering other incidents'.

*df_est_hospitd.obj* - estimates from the model 'post-falling risk of hospitalization'.
