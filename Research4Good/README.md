#  Readme

First run `pip install -r requirements.txt` to install the required packages.


---------

This demo was developed and last ran on Colab (in November 2023).

To run it, please download a [copy of the intermediate files](https://drive.google.com/drive/folders/1Tl-yM64qCMxI-dr7Ai9rReQkRu2VUASf) to your Google drive. The download requires ~10GB of storage space. The copy contains:
- ```requirements.txt```
- Pre-computed word embeddings (5 sets)

The demo will also download a couple of resources via command ```!wget```

## Steps to get this notebook running on your Colab folder...

1. Change runtime type to use GPU computing (the demo notebook was developed with T4)
2. Mount drive (under **Admin setup**)
3. If this is the first time running this notebook, the **output may print out a message that asks you to restart the session**
4. After installing all required packages, choose on the menu: 
    "Runtime" > "Run all"


## Shortening execution time

In order to shorten the execution time, we recommend only running the key results / trials, e.g.

**Option A**: XGB Optuna loop will only try the following input settings
```
for mid in ['xgb',]:
    for input_type in [25]: # rather than other combinations of inputs [19,21,22,23,24,26]
```
**Option B**: Run Cox regression only



## ```requirements```

```
!cat requirements.txt
```