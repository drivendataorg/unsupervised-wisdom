#!/usr/bin/env python
# coding: utf-8

# ## Overview of approach

# In this study, a novel data sampling approach was employed to ensure representative segmentation. The strategy focused on maximizing diversity among clusters to accurately capture the essence of the original dataset. The data was partitioned into 10,000 clusters, mirroring the sample dataset's size. To enhance diversity, the data point closest to each cluster's centroid was selected and added to the sample dataset, thus promoting a diverse representation.
# 
# The reliability of the sample dataset was validated through comprehensive data analysis, demonstrating similarities in insights between the sample and actual data.
# 
# As part of our modelling pipeline, the adopted approach is two folds:
# 
# **Creation of Gold Standard Data**: Various Large language models (LLMs) were employed to generate gold standard data (training data) for fine-tuning the classification model.
# 
# ![Creation of gold data](gold_data.png)
# 
# **Output on Complete Dataset**: The gold data created in step 1 was used to fine-tune a DistilBERT model (for classification), which was used to generate predictions on the complete primary data.
# 
# ![Output on complete dataset](output_on_complete_data.png)

# ## Importing Libraries

get_ipython().run_cell_magic('capture', '', "import pandas as pd\nimport numpy as np\nimport random\n\nfrom sklearn.feature_extraction.text import CountVectorizer\nfrom nltk.corpus import stopwords\n\nimport warnings\nfrom bertopic import BERTopic\n\nimport spacy\nimport json\nimport re\nimport openai\n\nimport seaborn as sns\nimport matplotlib.pyplot as plt\nimport plotly.express as px\nimport pyarrow.parquet as pq\n\nfrom sklearn.decomposition import PCA\n\nfrom sklearn.cluster import KMeans\n\n\nfrom huggingface_hub import hf_hub_download\nfrom llama_cpp import Llama\n\nimport time\nimport datetime\nfrom datetime import timedelta\nwarnings.filterwarnings('ignore')\npd.set_option('display.max_colwidth', None)\npd.set_option('display.max_columns', None)\nnlp = spacy.load('en_core_web_sm')\n")


# ## Reading the data

# Our analysis relies on primary data. We have meticulously mapped all variables using JSON files.

primary_data = pd.read_csv('data/Primary_data_mapped_cleaned.csv')
primary_data.shape


display(primary_data.describe(include='all'))


# ### Combining variables having multiple categories (for finding relationships in the data)

selected_body_parts = ['HEAD', 'LOWER TRUNK', 'FACE', 'UPPER TRUNK', 'SHOULDER', 'KNEE', 'UPPER LEG', 'UPPER ARM', 'WRIST', 'LOWER ARM', 'LOWER LEG', 'ANKLE', 'ELBOW', 'NECK', 'HAND', 'FOOT']
primary_data['body_part_modified']= primary_data['body_part'].apply(lambda x: x if x in selected_body_parts else 'Other')


selected_diagnosis = ['FRACTURE', 'INTERNAL INJURY', 'CONTUSIONS, ABR.', 'LACERATION', 'STRAIN, SPRAIN', 'HEMATOMA', 'AVULSION', 'DISLOCATION', 'CONCUSSION']
primary_data['diagnosis_modified']= primary_data['diagnosis'].apply(lambda x: x if x in selected_diagnosis else 'Other')


selected_body_parts = ['HEAD', 'LOWER TRUNK', 'FACE', 'UPPER TRUNK', 'SHOULDER', 'KNEE', 'UPPER LEG', 'UPPER ARM', 'WRIST', 'LOWER ARM', 'LOWER LEG', 'ANKLE', 'ELBOW', 'NECK', 'HAND', 'FOOT']
primary_data['body_part_2_modified']= primary_data['body_part_2'].apply(lambda x: x if x in selected_body_parts else 'Other')


selected_diagnosis = ['FRACTURE', 'INTERNAL INJURY', 'CONTUSIONS, ABR.', 'LACERATION', 'STRAIN, SPRAIN', 'HEMATOMA', 'AVULSION', 'DISLOCATION', 'CONCUSSION']
primary_data['diagnosis_2_modified']= primary_data['diagnosis_2'].apply(lambda x: x if x in selected_diagnosis else 'Other')


# ## Deep Exploratory Data Analysis

# ### Function to plot Using Matplotlib

def eda_matplotlib(data, var1, var2, title=None, xaxis_title=None, legend_title=None, loc='upper right', rotation=60):
    cross_tab = pd.crosstab(data[var1], data[var2])
    
    # Get unique categories for var2
    categories = cross_tab.columns
    
    # Create a list of colors for each category
#     colors = plt.cm.Paired(range(len(categories)))    #Can be used if we have less than 12 categories
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))   #Use if we have more than 12 categories
    
    # Initialize bottom values for stacking
    bottom = [0] * len(cross_tab.index)
    
    plt.style.use('seaborn-darkgrid') 
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Iterate through categories and plot bars
    for i, category in enumerate(categories):
        counts = cross_tab[category]
        bars = plt.bar(cross_tab.index, counts, bottom=bottom, label=category, color=colors[i])
        bottom += counts
    
    # Add labels and title
    plt.xlabel(var1)
    plt.ylabel('Frequency')
    plt.title(title)
    
    # Add legend with loc
    plt.legend(title=legend_title, loc= loc)
    
    # Set x-axis label if provided
    if xaxis_title:
        ax.set_xlabel(xaxis_title)
        
    plt.xticks(rotation=rotation)
    
    # Remove gridlines
    plt.grid(False)
    
    plt.show()


# ### Body Part wrt Diagnosis

eda_matplotlib(primary_data, 'body_part_modified', 'diagnosis_modified', "Body Part Vs Diagnosis", "Body Part", 'Diagnosis')


# ### Disposition wrt Diagnosis

eda_matplotlib(primary_data, 'disposition', 'diagnosis_modified', 'Disposition Vs Diagnosis', 'Disposition', 'Diagnosis', 'upper left')


# ### Diagnosis Trend over Time

eda_matplotlib(trend_df, 'month_year', 'diagnosis_modified', 'Diagnosis Trend Over Time', 'Month-Year', 'Diagnosis', 'upper left', 90)


# ## Processing and cleaning the narratives

#Reading the medical terms mapping file
file_path = 'data/medical_terms_mapping.json'

with open(file_path, 'r') as json_file:
    medical_terms_mapping = json.load(json_file)

print(medical_terms_mapping)


#Function to clean & process the narratives
def clean_narratives(text):
    text = text.lower()

    # Unglue DX (diagnosis)
    regex_dx = r"([ˆ\W]*(dx)[ˆ\W]*)"
    text = re.sub(regex_dx, r". dx: ", text)

    # remove age and sex identifications
    # regex to capture age and sex
    regex_age_sex = r"(\d+)\s*?(yof|yf|yo\s*female|yo\s*f|yom|ym|yo\s*male|yo\s*m)"
    age_sex_match = re.search(regex_age_sex, text)

    #format age and sex
    if age_sex_match:
        age = age_sex_match.group(1)
        sex = age_sex_match.group(2)

        if "f" in sex:
            text = text.replace(age_sex_match.group(0), f"{age} years old female patient")
        elif "m" in sex:
            text = text.replace(age_sex_match.group(0), f"{age} years old male patient")

    # translate medical terms
    for term, replacement in medical_terms_mapping.items():
        if term == "@" or term == ">>" or term == "&" or term == "***":
            pattern = fr"({re.escape(term)})"
            text = re.sub(pattern, f" {replacement} ", text) # add spaces around replacement

        else:
            pattern = fr"(?<!-)\b({re.escape(term)})\b(?!-)"
            text = re.sub(pattern, replacement, text)

    return text


primary_data['cleaned_narrative'] = np.nan


tqdm.pandas(desc="Processing column1")
start = time.time()
primary_data['cleaned_narrative'] = primary_data['narrative'].apply(lambda x: clean_narratives(x))
end = time.time()
elapsed = end - start
print('Overall Time Elapsed:',elapsed)
print(str(timedelta(seconds=elapsed)))


primary_data[['cpsc_case_number', 'narrative', 'cleaned_narrative']].head()


primary_data.to_csv(r'data\cleaned_narrative_primary_data.csv', index=False)


# ## Selecting Representative Cases - using KMeans

# ### Loading the openAI embeddings

file_path = 'data/openai_embeddings_primary_narratives.parquet.gzip'


# Open the Parquet file using pyarrow
table = pq.read_table(file_path)

# Convert the table to a Pandas DataFrame
embeddings_df = table.to_pandas()


display(embeddings_df.head(1))


cleaned_narrative_with_embeddings = pd.merge(cleaned_narrative_primary_data,embeddings_df,on='cpsc_case_number', how='inner')


#Shape of each embedding
cleaned_narrative_with_embeddings['embedding'][0].shape


# ### Dimensionality Reduction

embedding_data = np.array(cleaned_narrative_with_embeddings['embedding'].values.tolist())
embedding_data.shape


# - Choosing the n_components as 200 because we are able to explain ~85% of the total variance using 200 dimensions. (For 95% explained variance the number of components (n_components) have to be >300)

n_components = 200

pca = PCA(n_components=n_components)

pca_result = pca.fit_transform(embedding_data)

explained_variances = pca.explained_variance_ratio_

cumulative_explained_variance = np.cumsum(explained_variances)

print("Explained variances by each component:", explained_variances)
print("Cumulative explained variance:", cumulative_explained_variance)


#Reduced embeddings
pca_result.shape


# ### Applying KMeans with number of clusters = 10000
# 
# - We want to identify 10k clusters with minimum intra-cluster variance and maximizing the inter-cluster distance

start = time.time()
kmeans = KMeans(n_clusters=10000)
kmeans.fit(pca_result)
cleaned_narrative_with_embeddings['cluster_label'] = kmeans.labels_
cleaned_narrative_with_embeddings.to_csv("data/k_means_cleaned_narrative_10k_clusters_200_dim.csv", index=False)

end = time.time()

elapsed = end - start
print('Overall Time Elapsed:',elapsed)
print(str(timedelta(seconds=elapsed)))


clusters_data = pd.read_csv("data/k_means_cleaned_narrative_10k_clusters_200_dim.csv")


embedding_clusters = pd.merge(cleaned_narrative_with_embeddings[['cpsc_case_number', 'embedding']], clusters_data[['cluster_label', 'cpsc_case_number']] , on = 'cpsc_case_number')


def get_closest_sample(df_filtered):
    if len(df_filtered)==1:
        return df_filtered['cpsc_case_number'].values[0]
    df_filtered = df_filtered.reset_index(drop=True)
    
    mean_array = np.mean(df_filtered['embedding'].values, axis=0)
    
    cosine_similarities = np.dot(np.array(df_filtered['embedding'].values.tolist()), mean_array)

    # Find the index of the vector with the highest cosine similarity
    closest_index = np.argmax(cosine_similarities)
    
    return df_filtered['cpsc_case_number'].values[closest_index]
    


cpsc_sample_list = []
cluster_sample_list = []
for k,v in embedding_clusters.groupby('cluster_label'):
    cpsc = get_closest_sample(v)
    cpsc_sample_list.append(cpsc)
    cluster_sample_list.append(k)


df_sample = pd.DataFrame({'cpsc_case_number':cpsc_sample_list, 'cluster_label':cluster_sample_list})


sample_data_10k = pd.merge(df_sample, cleaned_narrative_with_embeddings, on='cpsc_case_number', how='left')


sample_data_10k.to_csv('data/sample_data_10k.csv', index=False)


# ## Creating golden/training dataset for severity (using Llama 2 LLM)

# ### Installing Llama 2 on GPU

# GPU llama-cpp-python
get_ipython().system('CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.78 --force-reinstall --upgrade --no-cache-dir --verbose')
# For download the models
get_ipython().system('pip install huggingface_hub')


# ### Downloading the llama 2 model

model_name_or_path = "TheBloke/Llama-2-7B-chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin" # the model is in bin format


model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)


# ### Loading the model on GPU

# Loading on GPU
lcpp_llm = None
lcpp_llm = Llama(
    model_path=model_path,
  n_threads=2, # CPU cores
    n_batch=516, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=10000, # Change this value based on your model and your GPU VRAM pool.
    n_ctx=2048
    )


data = pd.read_csv('data/sample_data_10k.csv')


# ### Defining the prompt for Llama 2 model

def create_prompt(prompt,incident):

    prompt_template=f'''SYSTEM: Classify the severity of falls based on narratives.
User: Please classify the severity of the reported injury for an elderly patient in the incident report into one of the following categories:
- "Very Severe": Life-threatening with long-term effects
- "Severe": Not life-threatening but requiring multiple days of hospitalization
- "Moderate": Requires a doctor's visit but no hospitalization
- "Minor": No hospitalization or doctor's visit needed
Severity Classification: [Very Severe/Severe/Moderate/Minor]

Fall Narrative: {incident}
Assistant:'''
    
    return prompt_template


# ### Getting the predictions from Llama 2 for representative data (10k)

results=[]
for i in range(data.shape[0]):
    try:
        prompt=create_prompt('',data.narrative.values[i]+'.')

        response=lcpp_llm(prompt=prompt, max_tokens=800, temperature=0, top_p=0.9,
                      repeat_penalty=1.2, top_k=1,
                      echo=False)
        result=response["choices"][0]["text"]

        results.append(result)
        
    except Exception as e:
        print(f"Error {e} occurred at count = {i}")
        results.append(None)
        
        

final_data=data[['cpsc_case_number']].copy()
final_data['severity']=results


final_list=[]
for i in final_data.severity:
    try:
        pattern = r'\b(Very Severe|Severe|Moderate|Minor)\b'
        matches = re.findall(pattern, i)
        final_list.append(matches[0])
    except:
        final_list.append('Not Found')


final_data['severity_of_fall']=final_list
final_data.drop(columns=['severity'], inplace=True)


final_data.to_csv('data/severity_data_10k.csv',index=False)


# ## Creating golden/training dataset for Reason of Fall - using OpenAI ChatGPT

# ### Installing OpenAI

get_ipython().system('pip install openai')


openai.api_key= 'INSERT_YOUR_API_KEY'


# ### Defining the User & System Prompts for Turbo 3.5 model

user_prompt = """Below are some common reasons associated with falls of elderly people - 

- Medical and Health Factors like chronic medical conditions (e.g., arthritis, stroke, Parkinson's, dementia, Alzheimer's, \
incontinence, postural hypotension), medications (sedatives, antidepressants, cardiovascular drugs, polypharmacy), cognitive \
and psychological factors (depression, cognitive impairment, memory loss).

- Demographic and Lifestyle Factors like age-related factors (older age, gender), living situation (living alone, \
housebound), reduced physical activity/exercise, history of falls.

- Sensory and Perception Factors like impaired vision (visual acuity, perceptual errors), impaired hearing.

- Environmental Factors like home hazards (for eg - wet floor, carpet, rug, item on floor) , environmental hazards (for eg - uneven roads, things on floor, etc).

- Behavioral Factors like risky behaviors, alcohol use with medication, bladder or bowel conditions.

- Physical Factors like Lower body weakness, reduced strength (knee, hip, ankle, grip), pain/arthritis (hip, knee), \
sensory impairments (visual acuity, depth perception, lower extremity sensory function), gait and balance issues.

### Narrative Text
{text}

Based on the narrative text above, can you choose the most appropriate reason of fall from the above options ? Format the output in the form of a json as
{<reason> : <explanation>}

Output only the json text."""

system_message = """You are a bot designed to identify the most appropriate reason of fall based on the Narrative Text."""


# ### Defining functions to call the Chat Completion API

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def get_message(user_message, system_message):
    messages =  [
        {'role':'system', 'content': system_message},
        {'role':'user', 'content':user_message}]
    return messages


# ### Getting predictions from OpenAI ChatGPT for representative data (10k)

json_extraction_pattern = r'\{[^{}]*\}'
reason_results=[]
explanation_results = []

for i in range(data.shape[0]):
    try:
        narrative = data.cleaned_narrative.values[i]
        user_message = user_prompt.replace('{text}', narrative)
        response = get_completion_from_messages(get_message(user_message, system_message))

        response_json = re.findall(json_extraction_pattern, response)[0]

        response_dict = json.loads(response_json)
        reason_of_fall = list(response_dict.keys())[0]
        explanation = list(response_dict.values())[0]

        reason_results.append(reason_of_fall)
        explanation_results.append(explanation)

    except Exception as e:
        print(f"Error {e} occurred at count = {i}")
        print(f"Narrative Text is \n{narrative}")
        reason_results.append("NA")
        explanation_results.append("NA")
        
    time.sleep(20)

temp_data=data[['cpsc_case_number']].copy()
temp_data['reason_of_fall'] = reason_results
# temp_data['reason_explanation'] = explanation_results
temp_data.to_csv('data/reason_of_fall_data_10k.csv',index=False)


# ## Creating golden/training dataset for Action before Fall - using OpenAI ChatGPT

# ### Defining the User & System Prompts for Turbo 3.5 model

user_prompt = """Below are different activities which elderly people tend to perform leading to their fall -

Assisted Mobility Activities : Activities that involve the use of different mobility aids or assistance such as using a \
walker, wheelchair , etc.

Walking and Gait Activities : Activities that involve walking, running and maintaining an upright posture.

Stair Mobility Activities : Activities that involve climbing or descending the stairs/steps.

Bathroom Activities : Activities that involve bathroom or using the washroom/toilet facilities.

Bedroom Activities : Activities that involve bed or are associated with the bedroom.

Household Activities : Activities that involve household, household chores or that took place inside the home.

Outdoor Activities : Activities that involve anything happening outside the house or outdoors.

Special Leisure Activities : Activities that involve unusual events and recreational/leisure activities.

Other Miscellaneous Activities : Activities that do not fall under any of the above categories.

Unknown Activities : If no activity can be identified from the narrative text.

### Narrative Text
{text}


Based on the narrative text given above, can you choose one of the activities the person was performing just before the \
fall. Respond in the form of a json as 

{<Activity before fall> : <explanation>}

Output only the json text."""

system_message = """You are an expert in identifying and classifying the action performed by the person just before the fall based on the Narrative Text given"""


# ### Getting predictions from OpenAI ChatGPT for representative data (10k)

json_extraction_pattern = r'\{[^{}]*\}'
action_before_fall_results=[]
explanation_results = []

for i in range(data.shape[0]):
    try:
        narrative = data.cleaned_narrative.values[i]
        user_message = user_prompt.replace('{text}', narrative)
        response = get_completion_from_messages(get_message(user_message, system_message))

        response_json = re.findall(json_extraction_pattern, response)[0]

        response_dict = json.loads(response_json)
        action_before_fall = list(response_dict.keys())[0]
        explanation = list(response_dict.values())[0]

        action_before_fall_results.append(action_before_fall)
        explanation_results.append(explanation)

    except Exception as e:
        print(f"Error {e} occurred at count = {i}")
        print(f"Narrative Text is \n{narrative}")
        action_before_fall_results.append("NA")
        explanation_results.append("NA")
        
    time.sleep(20)

temp_data=data[['cpsc_case_number']].copy()
temp_data['action_before_fall'] = action_before_fall_results
# temp_data['action_before_fall_explanation'] = explanation_results
temp_data.to_csv('data/action_before_fall_data_10k.csv',index=False)


# ## Building a Classifier - Training DistilBERT on Golden Data

# ### Installing & Loading DistilBERT

get_ipython().system('pip install numpy==1.23.5 --user')
get_ipython().system('pip install transformers[torch]')


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


from sklearn.model_selection import train_test_split


# ### Creating a Classifier for Severity
# 
# - Loading the severity training data

train_data = pd.read_csv('data/severity_data_10k.csv')


sample_data_10k = pd.read_csv('data/sample_data_10k.csv')
train_data = sample_data_10k.merge(train_data, on='cpsc_case_number', how='left')


train_data = train_data[['cleaned_narrative', 'severity_of_fall']]
train_data.columns=['text','label']


# #### Label Encoding

label2id = dict(zip(sorted(train_data.label.unique()), range(len(train_data.label.unique()))))
id2label = {v:k for k,v in label2id.items()}


train_data['label']=train_data['label'].apply(lambda x:label2id[x])


train_data.head()


X = train_data.drop('label', axis = 1)
y = train_data['label']


# #### Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify = y)


X_train['label'] = y_train
X_test['label'] = y_test


X_train.to_csv('data/Bert_train.csv', index=False)
X_test.to_csv('data/Bert_test.csv', index=False)


get_ipython().system(' pip install datasets')


from datasets import load_dataset


dataset = load_dataset('csv', data_files={'train': 'data/Bert_train.csv',
                                          'test': 'data/Bert_test.csv'})


dataset


# #### Preprocess the dataset

from transformers import DataCollatorWithPadding


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,padding=True)

tokenized_data = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# #### Function to compute F1-Score

from sklearn.metrics import f1_score

def compute_f1(pred):
    # Extract predicted labels and true labels from the prediction tuple
    predictions, labels = pred.predictions, pred.label_ids

    # Calculate F1 score
    f1 = f1_score(y_true=labels, y_pred=predictions.argmax(axis=1), average='macro')

    return {"f1": f1}


tokenized_data


# #### Training DistilBERT model for n epochs

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

#Number of labels
n_labels = len(list(label2id.keys()))

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=n_labels, id2label=id2label, label2id=label2id).to("cuda")

training_args = TrainingArguments(
    output_dir="data/models",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=4000,
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_f1,
)

trainer.train()


# #### Loading the last best checkpoint of the model

from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="data/models/checkpoint-20000/", device=0) #Setting device=0 to enable GPU


# #### Making predictions on the whole dataset

predict_data=pd.read_csv('data/cleaned_narrative_primary_data.csv')

predicted_outputs = []
# Define the batch size for processing multiple texts at once
batch_size = 1000

# Iterate over the dataframe in batches
for i in range(0, predict_data.shape[0], batch_size):
    print(i)
    batch_texts = predict_data['cleaned_narrative'].iloc[i:i+batch_size].tolist()

    results=classifier(batch_texts)     
    results=[i['label'] for i in results]
    predicted_outputs.extend(results)   
    
predict_data['severity']=predicted_outputs
predict_data[['cpsc_case_number', 'severity_of_fall']].to_csv('data/severity_whole_data.csv', index=False)


# - Similar to the classification model trained for severity of fall, we have trained two more classification models namely for classifying the action before fall and reason of fall.

# ## Collating the data and Generating Insights
# - Deep Exploratory Data Analysis

# ### Reading the data after making predictions

action_before_fall_10k = pd.read_csv("data/action_before_fall_data_10k.csv")
action_before_fall_100k = pd.read_csv("data/action_before_fall_whole_data.csv")

reason_of_fall_10k = pd.read_csv("data/reason_of_fall_data_10k.csv")
reason_of_fall_100k = pd.read_csv("data/reason_of_fall_whole_data.csv")

severity_of_fall_10k = pd.read_csv("data/severity_data_10k.csv")
severity_of_fall_100k = pd.read_csv("data/severity_whole_data.csv")


falls_valid_category = ["Assisted Mobility Activities",
"Walking and Gait Activities",
"Stair Mobility Activities",
"Bathroom Activities",
"Bedroom Activities",
"Household Activities",
"Outdoor Activities",
"Special Leisure Activities",
"Other Miscellaneous Activities",
"Unknown Activities"]

severity_valid_category = ['Severe', 'Very Severe', 'Moderate', 'Minor']


severity_of_fall_10k = severity_of_fall_10k[severity_of_fall_10k['severity_of_fall'].isin(severity_valid_category)]
severity_of_fall_10k.shape


action_before_fall_10k = action_before_fall_10k[action_before_fall_10k['action_before_fall'].isin(falls_valid_category)]
action_before_fall_10k.shape


action_before_fall_10k.shape, action_before_fall_100k.shape


reason_of_fall_10k.shape, reason_of_fall_100k.shape


severity_of_fall_10k.shape, severity_of_fall_100k.shape


action_before_fall_final = pd.concat([action_before_fall_10k, action_before_fall_100k[~action_before_fall_100k['cpsc_case_number'].isin(action_before_fall_10k['cpsc_case_number'].unique())]])


severity_of_fall_final = pd.concat([severity_of_fall_10k, severity_of_fall_100k[~severity_of_fall_100k['cpsc_case_number'].isin(severity_of_fall_10k['cpsc_case_number'].unique())]])


reason_of_fall_final = pd.concat([reason_of_fall_10k, reason_of_fall_100k[~reason_of_fall_100k['cpsc_case_number'].isin(reason_of_fall_10k['cpsc_case_number'].unique())]])


action_reason = pd.merge(action_before_fall_final, reason_of_fall_final, on='cpsc_case_number', how='left')


action_reason_severity = pd.merge(action_reason, severity_of_fall_final, on='cpsc_case_number', how='left')


final_data = pd.merge(cleaned_narrative_primary_data, action_reason_severity, on='cpsc_case_number', how='left')


final_data.head(10)


final_data.to_csv("data/final_combined_data.csv", index=False)


# ### Combining variables having multiple categories (for finding relationships in the data)

selected_body_parts = ['HEAD', 'LOWER TRUNK', 'FACE', 'UPPER TRUNK', 'SHOULDER', 'KNEE', 'UPPER LEG', 'UPPER ARM', 'WRIST', 'LOWER ARM', 'LOWER LEG', 'ANKLE', 'ELBOW', 'NECK', 'HAND', 'FOOT']
final_data['body_part_modified']= final_data['body_part'].apply(lambda x: x if x in selected_body_parts else 'Other')

selected_diagnosis = ['FRACTURE', 'INTERNAL INJURY', 'CONTUSIONS, ABR.', 'LACERATION', 'STRAIN, SPRAIN', 'HEMATOMA', 'AVULSION', 'DISLOCATION', 'CONCUSSION']
final_data['diagnosis_modified']= final_data['diagnosis'].apply(lambda x: x if x in selected_diagnosis else 'Other')

selected_body_parts = ['HEAD', 'LOWER TRUNK', 'FACE', 'UPPER TRUNK', 'SHOULDER', 'KNEE', 'UPPER LEG', 'UPPER ARM', 'WRIST', 'LOWER ARM', 'LOWER LEG', 'ANKLE', 'ELBOW', 'NECK', 'HAND', 'FOOT']
final_data['body_part_2_modified']= final_data['body_part_2'].apply(lambda x: x if x in selected_body_parts else 'Other')

selected_diagnosis = ['FRACTURE', 'INTERNAL INJURY', 'CONTUSIONS, ABR.', 'LACERATION', 'STRAIN, SPRAIN', 'HEMATOMA', 'AVULSION', 'DISLOCATION', 'CONCUSSION']
final_data['diagnosis_2_modified']= final_data['diagnosis_2'].apply(lambda x: x if x in selected_diagnosis else 'Other')


final_data.head()


# ### Creating Age Buckets

bins = list(range(65, 131, 5))

labels = [f'{start}-{end-1}' for start, end in zip(bins[:-1], bins[1:])]

final_data['age_group'] = pd.cut(final_data['age'], bins=bins, labels=labels, right=False)


# ### Analyzing Very Severe Cases of Fall wrt Age Buckets

temp_df = final_data[final_data['severity_of_fall']=='Very Severe']
temp_df.shape


severity_df = final_data[['severity_of_fall', 'age_group']]
severity_df.head()


severe_groups = severity_df.groupby('age_group')


severe_perc_df = pd.DataFrame()
for group,data in severe_groups:
    print(group)
    count_df = data.value_counts().to_frame().reset_index()
    count_df['percentage'] = np.nan
    count_df['percentage'] = count_df[0]/sum(list(count_df[0]))
    severe_perc_df = pd.concat([severe_perc_df, count_df])    


very_severe_df = severe_perc_df[severe_perc_df['severity_of_fall']=='Very Severe']
very_severe_df


# ### Analyzing Distribution of Very Severe Cases wrt Age Group

# Create a bar plot
plt.figure(figsize=(6, 4))
plt.bar(very_severe_df['age_group'], very_severe_df['percentage']*100, color='brown')
plt.xlabel('Age Group')
plt.ylabel('% of Very Severe Cases')
plt.title('Graph 1: Distribution of Very Severe Cases')

plt.xticks(rotation=45)
plt.show()


# ### Plotting Graphs using Matplotlib - Stacked Bar Chart (Bivariate/Cross Tab Analysis)

def eda_matplotlib(data, var1, var2, title=None, xaxis_title=None, legend_title=None, loc='upper right', rotation=60, calc_percentage = False):
    cross_tab = pd.crosstab(data[var1], data[var2])
    categories = cross_tab.columns
    
    # Create a list of colors for each category
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    
    # Initialize bottom values for stacking
    bottom = [0] * len(cross_tab.index)
    
    plt.style.use('seaborn-darkgrid') 
    
    fig, ax = plt.subplots(figsize=(5,3))
    
    if calc_percentage:
        cross_tab = cross_tab.apply(lambda x: x / x.sum(), axis=1)
        # Iterate through categories and plot bars
        for i, category in enumerate(categories):
            counts = cross_tab[category]
            bars = plt.bar(cross_tab.index, counts, bottom=bottom, label=category, color=colors[i])
            bottom += counts
        # Add legend with loc
        plt.legend(title=legend_title, loc= loc, bbox_to_anchor=(1, 1))

    else :  
        # Iterate through categories and plot bars
        for i, category in enumerate(categories):
            counts = cross_tab[category]
            bars = plt.bar(cross_tab.index, counts, bottom=bottom, label=category, color=colors[i])
            bottom += counts
        # Add legend with loc
        plt.legend(title=legend_title, loc= loc)
    
    # Add labels and title
    plt.ylabel('Frequency')
    plt.title(title)

    # Set x-axis label if provided
    if xaxis_title:
        ax.set_xlabel(xaxis_title)

    plt.xticks(rotation=rotation, ha = 'left')
    
    # Remove gridlines
    plt.grid(False)
    plt.show()


# ### Analyzing Body Part wrt Severity

eda_matplotlib(final_data, 'body_part_modified', 'severity_of_fall', 'Graph 2: Body Part Vs Severity', 'Body Part', legend_title='Severity', rotation=-45, loc='upper left', calc_percentage=True )


# ### Analyzing Age Group wrt Action Before Fall

eda_matplotlib(final_data, 'age_group', 'action_before_fall', 'Graph 3: Age Group Vs Action Before Fall', 'Age Group', legend_title='Action before Fall', rotation=90, loc='upper left', calc_percentage=True )


# ### Analyzing Action Before Fall wrt Severity

eda_matplotlib(final_data, 'action_before_fall', 'severity_of_fall', 'Graph 4: Action Before Fall Vs Severity', 'Action before Fall', legend_title='Severity', rotation=-30, loc='upper left', calc_percentage=True )


# ### Analyzing Risk Factors wrt Severity

eda_matplotlib(final_data, 'reason_of_fall', 'severity_of_fall', 'Graph 5: Risk Factors Vs Severity', 'Risk Factors', 'Severity', rotation=-30, loc='upper left', calc_percentage=True )





# ## NULL Result Section
# 
# *What Didn’t work?*
# - BERTopic with hyperparameter tuning: Unable to get insights since most of the data getting clustered as noise.
# - KMeans Clustering: Hard to find the K for KMeans and clusters were not homogeneous.
# - FLAN-T5 for generating Severity, reason, and action before fall: Very low accuracy for zero shot classification.
# - XGBoost Model for training on Gold Data and Predicting: Much lower accuracy after manual Validation

# ### Applying BERTopic for Topic Modelling

# !pip install bertopic


import random

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


from bertopic import BERTopic

import spacy
import nltk
nltk.download('stopwords')

import pyarrow.parquet as pq

from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired, TextGeneration, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from transformers import pipeline


def fit_topic_model_v2(filtered_df, stop_words, on_column = "cleaned_text", embedding_column = "embedding",
                   language="english"):

    dialogue_list = filtered_df[on_column].tolist()
    openai_embeddings = np.array(filtered_df[embedding_column].tolist())
    print("Embeddings dimensions : ", openai_embeddings.shape)
    
    #Tuned Parameters (saved as bert_topic_model_20230904_v7.pickle)
    vectorizer_model = CountVectorizer(stop_words=stop_words)
    ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=True)
    umap_model = UMAP(n_neighbors=15, n_components=100, min_dist=0.0, metric='cosine', random_state=0,  low_memory=True)
    hdbscan_model = HDBSCAN(min_samples=1, cluster_selection_method='leaf', prediction_data=True)
    representation_model = MaximalMarginalRelevance(diversity=0.4)
    topic_model = BERTopic(representation_model=representation_model, language=language, umap_model=umap_model, hdbscan_model = hdbscan_model, vectorizer_model=vectorizer_model,
                           ctfidf_model= ctfidf_model, calculate_probabilities=True, nr_topics=10000, min_topic_size=500)
    topics, probs = topic_model.fit_transform(dialogue_list, embeddings= openai_embeddings)
    topic_model.save('model/bert_topic_model_20230908_v1.pickle')
    # new_topics = topic_model.reduce_outliers(dialogue_list, topics)
    print("Total Topics: ", len(dict(topic_model.get_topic_info()["Name"])))
    return topic_model, topic_model.get_topic_info()


start = time.time()
topic_model, topic_info = fit_topic_model_v2(cleaned_narrative_with_embeddings, stop_words = english_stopwords, on_column = "cleaned_narrative",
      embedding_column = 'embedding',  language="english")
end = time.time()
elapsed = end - start
print('Overall Time Elapsed:',elapsed)
print(str(timedelta(seconds=elapsed)))
topic_info.head()


topic_model = BERTopic.load('model/bert_topic_model_20230908_v1.pickle')


topic_model.get_topic_info()


dialogue_list = cleaned_narrative_with_embeddings["cleaned_narrative"].tolist()
df_topic = topic_model.get_document_info(dialogue_list, df=cleaned_narrative_with_embeddings, metadata=None)
print(df_topic.shape)


topic_model.visualize_topics()


# ![Intertopic_distance](Intertopic_distance.png)

topic_model.visualize_barchart()


# ![visualize_barchart](visualize_barchart.png)

docs = cleaned_narrative_with_embeddings['cleaned_narrative'].tolist()
openai_embeddings = np.array(cleaned_narrative_with_embeddings['embedding'].tolist())
# Reduce dimensionality of embeddings, this step is optional
# reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
topic_model.visualize_documents(docs, embeddings= openai_embeddings, topics=[1,2,3,4, 5, 6, 7 , 8 , 9 ,10], sample = 0.1)


# ![Document and Topics](document_and_topic.png)

topic_model.visualize_heatmap()


# ![Visualize Heatmap](similarity_matrix.png)

# #### #########END OF NOTEBOOK##############



