#  Readme

Run `pip install -r requirements.txt` to install the required packages.

Note that notebook `n_optuna_11c_t2hosp__part4.ipynb` uses a series of intermediate files and assumes your data folder looks like the following:

```
data
├── LEALLA
│   ├── narrative_cleaned_n426691_emb6_d0_2023-10-04.pkl
│   ├── narrative_cleaned_n426691_emb6_d10_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d11_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d12_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d13_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d14_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d15_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d16_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d17_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d18_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d19_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d1_2023-10-04.pkl
│   ├── narrative_cleaned_n426691_emb6_d2_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d3_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d4_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d5_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d6_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d7_2023-10-05.pkl
│   ├── narrative_cleaned_n426691_emb6_d8_2023-10-05.pkl
│   └── narrative_cleaned_n426691_emb6_d9_2023-10-05.pkl
├── all_cpsc.pkl
├── all_embeddings_1.pkl
├── all_embeddings_2.pkl
├── all_embeddings_3.pkl
├── all_embeddings_4.pkl
├── decoded_df2_unique.csv
├── openai_embeddings_primary_narratives.parquet.gzip
├── primary_data.csv
├── supplementary_data.csv
└── variable_mapping.json
```

### Colab notebooks showing creation of intermediate files

- [Module 1: Load data files and clean narratives](https://colab.research.google.com/drive/1cJt-yfOVFhHqSow5zayMjllQBGjhh67g)
  - Will generate `all_cpsc.pkl` and `decoded_df2_unique.csv`
- [Module 2: Extract word embeddings using 5 different models](https://colab.research.google.com/drive/1chKtCLBwTJPfcQXJG6VMjK-c_iXJvQ9K)
  - Will generate `all_embeddings_*.pkl` and `narrative_clean_n426691*.pkl`
- [Module 3: Train and evaluate survival models](https://colab.research.google.com/drive/1rKEkwQaaUI71etntOM-nuT3rU6IVpbtg)
  - Note: this shows rendered output for the notebook included in this repo
  