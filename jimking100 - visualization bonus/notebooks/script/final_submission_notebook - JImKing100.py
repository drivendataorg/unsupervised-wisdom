#!/usr/bin/env python
# coding: utf-8

# # Exploring Medical Narratives on Older Adult Falls

# I use a custom Named Entity Recognition (NER) model to identify cause and precipitating event keywords in the narrative section of the primary data. I use the spaCy package to build the custom model. The custom keywords are organized into four major categories:
# 
# 1. Method of Fall – Fall, Trip, Slip
# 2. Care Facility – Whether a care facility was involved.
# 3. Medical Event – Whether a medical event was involved.
# 4. Activity – Whether a particular activity was involved.
# 
# There are several reasons for selecting this approach:
# 
# 1. A review of the narrative data indicates that most of the narrative is detailed in the other features (e.g. the body part, disposition, fire involvement, alcohol involvement, drug involvement and products involved). So, very few words remain and often those remaining words refer to the cause or precipitating event. The key is to identify the “cause” keywords and the NER model is an excellent method.
# 
# 2. A standard or generic NER provided by a package such as spaCy does not contain the specific medical terms and other terms typically used in the narrative. In addition, the medical terms used must be very specific, so a NER pre-trained on medical terms would be problematic. For example, many of the medical terms in the narrative refer to the result of the fall (e.g. a broken arm) when we are concerned with medical events that precede the fall (e.g. fainting). Thus, a custom model will provide the best results for this specific use case.
# 
# 3. Once the NER model is trained and run on the narrative, it can provide several benefits. The results can be used to categorize and detail the causes and precipitating events. The model can also be used to highlight the cause factors in the text providing more real-time feedback to those entering and using the data. Ultimately, the insights provided by the model can help make decisions on how to prevent falls.

# ## Import Packages and Set Data Paths 

import pandas as pd
import numpy as np
import json
from pathlib import Path

import spacy
from spacy import displacy
from spacy.tokens import DocBin

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge

DATA_DIRECTORY = Path("/Users/JKMacBook/Documents/Lambda/Falls/data")
data_file = DATA_DIRECTORY / "primary_data.csv"
mapping_file = DATA_DIRECTORY / "variable_mapping.json"
train_file = DATA_DIRECTORY / "train.spacy"
val_file = DATA_DIRECTORY / "val.spacy"
model_file = DATA_DIRECTORY / "output/model-best"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ## Load Primary Data

df = pd.read_csv(
    data_file,
    # set columns that can be null to nullable ints
    dtype={"body_part_2": "Int64", "diagnosis_2": "Int64"},
)
df.head(50)


# ## Recode the Data to Readable Form

with Path(mapping_file).open("r") as f:
    mapping = json.load(f, parse_int=True)
    
# convert the encoded values in the mapping to integers since they get read in as strings
for c in mapping.keys():
    mapping[c] = {int(k): v for k, v in mapping[c].items()}

for col in mapping.keys():
    df[col] = df[col].map(mapping[col])

df.head(50)


# ## Create the Custom SpaCy NER Model Training and Validation Data

# This is the heart of the NER Model where the word patterns in the narrative are given one of five labels (TRIP, SLIP, FACILITY, EVENT AND ACTIVITY).  TRIP and SLIP are two of the three methods of falls with FALL being the third method.  The word patterns were selected by simply "eyeballing" several hundered narratives and using the human brain to decide cause specific words.
# 
# The first 20,000 narratives are used as training data and the next 10,000 narratives are used as validation data.

#Build upon the spaCy blank model
nlp = spacy.blank("en")

#Create the EntityRuler
ruler = nlp.add_pipe("entity_ruler")

#List of Entities and Patterns
patterns = [
                {"label": "TRIP", "pattern": "TRIP"},
                {"label": "TRIP", "pattern": "TRIPPED"},
                {"label": "TRIP", "pattern": "TRIPPING"},
                {"label": "TRIP", "pattern": "T'D"},
                {"label": "TRIP", "pattern": "T'D&F"},
                {"label": "TRIP", "pattern": "T/P"},
                {"label": "TRIP", "pattern": "STUMBLE"},
                {"label": "TRIP", "pattern": "STUMBLED"},
                {"label": "TRIP", "pattern": "TANGLE"},
                {"label": "TRIP", "pattern": "TANGLED"},
                {"label": "TRIP", "pattern": "STUB"},
                {"label": "TRIP", "pattern": "STUBBED"},
    
                {"label": "SLIP", "pattern": "SLIP"},
                {"label": "SLIP", "pattern": "SLIPPED"},
                {"label": "SLIP", "pattern": "SLIPPING"},
                {"label": "SLIP", "pattern": "S'D"},
                {"label": "SLIP", "pattern": "S'D&F"},
                {"label": "SLIP", "pattern": "S/P"},
                {"label": "SLIP", "pattern": "SLID"},
                {"label": "SLIP", "pattern": "WET"},
    
                {"label": "FACILITY", "pattern": "NURSING HOME"},
                {"label": "FACILITY", "pattern": "NH"},
                {"label": "FACILITY", "pattern": "REHAB"},
                {"label": "FACILITY", "pattern": "ECF"},
                {"label": "FACILITY", "pattern": "CARE"},
                {"label": "FACILITY", "pattern": "FACILITY"},
                {"label": "FACILITY", "pattern": "GROUP HOME"},
    
                {"label": "EVENT", "pattern": "SYNCOPAL"},
                {"label": "EVENT", "pattern": "SYNCOPE"},
                {"label": "EVENT", "pattern": "DIZZY"},
                {"label": "EVENT", "pattern": "PASSED OUT"},
                {"label": "EVENT", "pattern": "ALZHEIMER"},
                {"label": "EVENT", "pattern": "ALZHEIMERS"},
                {"label": "EVENT", "pattern": "PARKINSON"},
                {"label": "EVENT", "pattern": "PARKINSONS"},
                {"label": "EVENT", "pattern": "DEMENTIA"},
                {"label": "EVENT", "pattern": "DEMENTED"},     
                {"label": "EVENT", "pattern": "LIGHTHEADED"},
                {"label": "EVENT", "pattern": "UNSTEADY"},
                {"label": "EVENT", "pattern": "SEIZURE"},
                {"label": "EVENT", "pattern": "PARKINSON"},
                {"label": "EVENT", "pattern": "BALANCE"},
    
                {"label": "ACTIVITY", "pattern": "WALKING"},
                {"label": "ACTIVITY", "pattern": "PLAYING"},
                {"label": "ACTIVITY", "pattern": "BENDING"},
                {"label": "ACTIVITY", "pattern": "WORKING"},
                {"label": "ACTIVITY", "pattern": "HIKING"},
                {"label": "ACTIVITY", "pattern": "CARRYING"},
                {"label": "ACTIVITY", "pattern": "MOWING"},
                {"label": "ACTIVITY", "pattern": "DANCING"},
                {"label": "ACTIVITY", "pattern": "ROLLERSKATING"},
                {"label": "ACTIVITY", "pattern": "SWIMMING"},
                {"label": "ACTIVITY", "pattern": "TRANSFERRING"},
                {"label": "ACTIVITY", "pattern": "LIFTING"},
                {"label": "ACTIVITY", "pattern": "FIXING"},
                {"label": "ACTIVITY", "pattern": "REACHING"},
                {"label": "ACTIVITY", "pattern": "STANDING ON"},
                {"label": "ACTIVITY", "pattern": "PICKING UP"},
                {"label": "ACTIVITY", "pattern": "PICK UP"},
                {"label": "ACTIVITY", "pattern": "GETTING UP"},
                {"label": "ACTIVITY", "pattern": "GET UP"},
                {"label": "ACTIVITY", "pattern": "GOT UP"},
                {"label": "ACTIVITY", "pattern": "GOING UP"},
                {"label": "ACTIVITY", "pattern": "GOING DOWN"},
                {"label": "ACTIVITY", "pattern": "GETTING OUT"},
                {"label": "ACTIVITY", "pattern": "GET OUT"},
                {"label": "ACTIVITY", "pattern": "GOT OUT"}
    
            ]

ruler.add_patterns(patterns)


TRAIN_DATA = []
VAL_DATA = []

#iterate over the corpus
for ind in df.index:
    text = df['narrative'][ind]
    doc = nlp(text)
    
    #remember, entities needs to be a dictionary in index 1 of the list, so it needs to be an empty list
    entities = []
    
    #extract entities
    for ent in doc.ents:

        #appending to entities in the correct format
        entities.append([ent.start_char, ent.end_char, ent.label_])
    
    # Use first 20,000 entries for training and next 10,000 entries for validation
    if entities:
        if ind <= 20000:
            TRAIN_DATA.append([text, {"entities": entities}])
        elif (ind > 20000) & (ind <= 30000):
            VAL_DATA.append([text, {"entities": entities}])
        else:
            break


# ## Display the First 100 Entries with Highlighted Patterns

# One of the strong features of a NER model is the ability to apply it directly to the narrative text to highlight the named enitities and their associated label.  This could be used to highlight the cause factors in the text providing more real-time feedback to those entering and using the data.

get_ipython().run_cell_magic('capture', '--no-display', 'for ind in df.index:\n    text = df[\'narrative\'][ind]\n    doc = nlp(text)\n    displacy.render(doc, style="ent")\n    if ind > 500:\n        break  \n')


# ## Convert the Data to SpaCy Format

# The data needs to be convereted to SpaCy format and saved to disk.

def convert(lang: str, TRAIN_DATA, output_path: Path):
    nlp = spacy.blank(lang)
    db = DocBin()
    for text, annot in TRAIN_DATA:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_path)

convert("en", TRAIN_DATA, train_file)
convert("en", VAL_DATA, val_file)


# ## The Following Steps Need to Occur in the Terminal (1st Time Only)

# Create config file and train model in terminal using the stored training and validation data.
# These steps are difficult to do in a notebook as specific configurations are machine and environment specific.

# See https://spacy.io/usage/training

# python -m spacy init fill-config base_config.cfg config.cfg
# python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./val.spacy


# ## Load Custom Trained NER Model

trained_nlp = spacy.load(model_file)


# ## Test on a Single Entry

text = "89YOF WAS SITTING ON THE EDGE OF HER BED WHEN SHE SLIPPED AND FELL ONTO HER BACK. DX: COMPRESSION FX OF VERTEBRA'"
doc = trained_nlp(text)

for ent in doc.ents:
    print (ent.text, ent.label_)


# ## Iterate Over the Primary Data and Categorize into Four Categories

# Iterate over all the primary data using the trained model categorizing each entry into four categories. Remember, TRIP and SLIP are in the method category.  Scroll right on the dataframe to see the four new features added to the dataframe.

#iterate over the corpus
df['method'] = ''
df['facility'] = ''
df['event'] = ''
df['activity'] = ''
for ind in df.index:
    text = df['narrative'][ind]
    doc = trained_nlp(text)
    
    #extract entities
    for ent in doc.ents:
        if ind < 100:
            print (ind, ent.text, ent.label_)
        if ent.label_ == 'TRIP':
            df.at[ind, 'method'] = ent.label_
        if ent.label_ == 'SLIP':
            df.at[ind, 'method'] = ent.label_
        if ent.label_ == 'FACILITY':
            df.at[ind, 'facility'] = ent.text
        if ent.label_ == 'EVENT':
            df.at[ind, 'event'] = ent.text
        if ent.label_ == 'ACTIVITY':
            df.at[ind, 'activity'] = ent.text


df.head(50)


# ## Fill the Method Feature with Fall for Blank Entries (Not a Trip or Slip)

# For the method featured, if the field is neither a SLIP or TRIP it is a FALL.

df['method'] = df['method'].replace('', 'FALL')
df.head(50)


# ## Code for Ishikawa Diagram

# Ishikawa diagrams are used to identify problems in a system by showing how causes and effects are linked.  The code for the Ishikawa diagram was derived from matplotlib:
# https://matplotlib.org/devdocs/gallery/specialty_plots/ishikawa_diagram.html

def problems(data: str,
             problem_x: float, problem_y: float,
             prob_angle_x: float, prob_angle_y: float):
    """
    Draw each problem section of the Ishikawa plot.

    Parameters
    ----------
    data : str
        The category name.
    problem_x, problem_y : float, optional
        The `X` and `Y` positions of the problem arrows (`Y` defaults to zero).
    prob_angle_x, prob_angle_y : float, optional
        The angle of the problem annotations. They are angled towards
        the tail of the plot.

    Returns
    -------
    None.

    """
    ax.annotate(str.upper(data), xy=(problem_x, problem_y),
                xytext=(prob_angle_x, prob_angle_y),
                fontsize='10',
                color='white',
                weight='bold',
                xycoords='data',
                verticalalignment='center',
                horizontalalignment='center',
                textcoords='offset fontsize',
                arrowprops=dict(arrowstyle="->", facecolor='black'),
                bbox=dict(boxstyle='square',
                          facecolor='tab:blue',
                          pad=0.8))


def causes(data: list, cause_x: float, cause_y: float,
           cause_xytext=(-9, -0.3), top: bool = True):
    """
    Place each cause to a position relative to the problems
    annotations.

    Parameters
    ----------
    data : indexable object
        The input data. IndexError is
        raised if more than six arguments are passed.
    cause_x, cause_y : float
        The `X` and `Y` position of the cause annotations.
    cause_xytext : tuple, optional
        Adjust to set the distance of the cause text from the problem
        arrow in fontsize units.
    top : bool

    Returns
    -------
    None.

    """
    for index, cause in enumerate(data):
        # First cause annotation is placed in the middle of the problems arrow
        # and each subsequent cause is plotted above or below it in succession.

        # [<x pos>, [<y pos top>, <y pos bottom>]]
        coords = [[0, [0, 0]],
                  [0.23, [0.5, -0.5]],
                  [-0.46, [-1, 1]],
                  [0.69, [1.5, -1.5]],
                  [-0.92, [-2, 2]],
                  [1.15, [2.5, -2.5]]]
        if top:
            cause_y += coords[index][1][0]
        else:
            cause_y += coords[index][1][1]
        cause_x -= coords[index][0]

        ax.annotate(cause, xy=(cause_x, cause_y),
                    horizontalalignment='center',
                    xytext=cause_xytext,
                    fontsize='9',
                    xycoords='data',
                    textcoords='offset fontsize',
                    arrowprops=dict(arrowstyle="->",
                                    facecolor='black'))


def draw_body(data: dict, fish_head: str):
    """
    Place each section in its correct place by changing
    the coordinates on each loop.

    Parameters
    ----------
    data : dict
        The input data (can be list or tuple). ValueError is
        raised if more than six arguments are passed.
    fish_head : str
        The heading string.

    Returns
    -------
    None.

    """
    second_sections = []
    third_sections = []
    # Resize diagram to automatically scale in response to the number
    # of problems in the input data.
    if len(data) == 1 or len(data) == 2:
        spine_length = (-2.1, 2)
        head_pos = (2, 0)
        tail_pos = ((-2.8, 0.8), (-2.8, -0.8), (-2.0, -0.01))
        first_section = [1.6, 0.8]
    elif len(data) == 3 or len(data) == 4:
        spine_length = (-3.1, 3)
        head_pos = (3, 0)
        tail_pos = ((-3.8, 0.8), (-3.8, -0.8), (-3.0, -0.01))
        first_section = [2.6, 1.8]
        second_sections = [-0.4, -1.2]
    else:  # len(data) == 5 or 6
        spine_length = (-4.1, 4)
        head_pos = (4, 0)
        tail_pos = ((-4.8, 0.8), (-4.8, -0.8), (-4.0, -0.01))
        first_section = [3.5, 2.7]
        second_sections = [1, 0.2]
        third_sections = [-1.5, -2.3]

    # Change the coordinates of the annotations on each loop.
    for index, problem in enumerate(data.values()):
        top_row = True
        cause_arrow_y = 1.7
        if index % 2 != 0:  # Plot problems below the spine.
            top_row = False
            y_prob_angle = -16
            cause_arrow_y = -1.7
        else:  # Plot problems above the spine.
            y_prob_angle = 16
        # Plot the 3 sections in pairs along the main spine.
        if index in (0, 1):
            prob_arrow_x = first_section[0]
            cause_arrow_x = first_section[1]
        elif index in (2, 3):
            prob_arrow_x = second_sections[0]
            cause_arrow_x = second_sections[1]
        else:
            prob_arrow_x = third_sections[0]
            cause_arrow_x = third_sections[1]
        if index > 5:
            raise ValueError(f'Maximum number of problems is 6, you have entered '
                             f'{len(data)}')

        # draw main spine
        ax.plot(spine_length, [0, 0], color='tab:blue', linewidth=2)
        # draw fish head
        ax.text(head_pos[0] + 0.1, head_pos[1] - 0.05, fish_head, fontsize=10,
                weight='bold', color='white')
        semicircle = Wedge(head_pos, 1, 270, 90, fc='tab:blue')
        ax.add_patch(semicircle)
        # draw fishtail
        triangle = Polygon(tail_pos, fc='tab:blue')
        ax.add_patch(triangle)
        # Pass each category name to the problems function as a string on each loop.
        problems(list(data.keys())[index], prob_arrow_x, 0, -12, y_prob_angle)
        # Start the cause function with the first annotation being plotted at
        # the cause_arrow_x, cause_arrow_y coordinates.
        causes(problem, cause_arrow_x, cause_arrow_y, top=top_row)


# ## Create Diagram Text

# The create_str procedure creates the text used in spines of the Ishikawa diagram.

def create_str(df, col, keywords, title):
    '''
    Input:
        df - a dataframe
        col - a column of the dataframe
        keywords - the keywords to group together
        title - the title to use on the diagram
    Output:
        f_str - a string
    '''
    n = 0
    
    val_len = len(df[col].value_counts().index)
    if val_len < 20:
        iterations = val_len
    else:
        iterations = 20
    
    for i in range(iterations):
        if df[col].value_counts().index[i] in keywords:
            n = n + df[col].value_counts()[i]
    
    n_pct = int(n / len(df.index) * 100)
    f_str = title + ' - ' + str(n_pct) + '%'
    return f_str


# ## Code to Categorize the Falls

# The create_categories procedure creates a dictionary consisting of category headings and lists of spine strings.  The categories dictionary is then used to render the Ishikawa diagram.

def create_categories(df):
    '''
    Input:
        df - a dataframe
    Output:
        categories - a dictionary of category headings and their corresponding spine strings
    '''
    
    # Alcohol/Drug Category
    a_pct = int((len(df.index) - df['alcohol'].value_counts()[0]) / len(df.index) * 100)
    d_pct = int((len(df.index) - df['drug'].value_counts()[0]) / len(df.index) * 100)
    ad_pct = a_pct + d_pct
    ad_str = 'Alcohol/Drug - ' + str(ad_pct) + '%'
    
    alcohol_str = create_str(df, 'alcohol', ['Yes'], 'Alcohol')
    drug_str = create_str(df, 'drug', ['Yes'], 'Drug')
    
    alcohol_drug = [alcohol_str, drug_str]

    # Fire Category
    fire_str = create_str(df, 'fire_involvement', ['FD'], 'Fire')
    fire_involvement = [fire_str]
    
    # Activity Category
    act_pct = int((len(df.index) - df['activity'].value_counts()[0]) / len(df.index) * 100)
    act_str = 'Activity - ' + str(act_pct) + '%'

    walking_str = create_str(df, 'activity', ['WALKING', 'HIKING', 'GOING UP', 'GOING DOWN'], 'Walking/Hiking')
    getup_str = create_str(df, 'activity', ['GETTING OUT', 'GETTING UP', 'GET UP', 'GOT UP', 'GET OUT', 'GOT OUT'], \
                           'Getting Up/Out')
    carrying_str = create_str(df, 'activity', ['CARRYING', 'BENDING', 'PICKING UP', 'LIFTING', 'REACHING'], 'Carrying/Bending')
    playing_str = create_str(df, 'activity', ['PLAYING', 'WORKING', 'SWIMMING', 'DANCING', 'MOWING', 'FIXING'], \
                             'Playing/Working')

    activities = [carrying_str, getup_str, playing_str, walking_str]
    
    # Facility Category
    fac_pct = int((len(df.index) - df['facility'].value_counts()[0]) / len(df.index) * 100)
    fac_str = 'Facility - ' + str(fac_pct) + '%'

    other_str = create_str(df, 'facility', ['FACILITY'], 'Other Facility')
    extended_str = create_str(df, 'facility', ['ECF', 'CARE'], 'Ext. Care Facility')
    nursing_str = create_str(df, 'facility', ['NURSING HOME', 'NH'], \
                             'Nursing Home')

    facilities = [other_str, nursing_str, extended_str]
    
    # Event Category
    event_pct = int((len(df.index) - df['event'].value_counts()[0]) / len(df.index) * 100)
    event_str = 'Medical Event - ' + str(event_pct) + '%'

    bal_str = create_str(df, 'event', ['BALANCE', 'UNSTEADY'], 'Lost Balance')
    dement_str = create_str(df, 'event', ['DEMENTIA', 'DEMENTED'], 'Dementia')
    faint_str = create_str(df, 'event', ['SYNCOPE', 'DIZZY', 'PASSED OUT', 'SYNCOPAL', 'LIGHTHEADED'], 'Fainting')

    events = [bal_str, dement_str, faint_str]
    
    # Product Category
    product_series = pd.concat([df['product_1'], df['product_2'], df['product_3']])
    frame = {'product': product_series}
    all_products = pd.DataFrame(frame)

    total_prods = len(all_products.index)
    prods_less_floors = total_prods - all_products['product'].value_counts()[1]
    product_pct = int((prods_less_floors - all_products['product'].value_counts()[0]) / prods_less_floors * 100)
    # product_pct = int(((len(df.index) * 3) - all_products['product'].value_counts()[0]) / (len(df.index) * 3) * 100)
    product_str = 'Products - ' + str(product_pct) + '%'

    floors_str = create_str(all_products, 'product', ['676 - RUGS OR CARPETS, NOT SPECIFIED'], 'Rugs/Carpets')
    beds_str = create_str(all_products, 'product', ['4076 - BEDS OR BEDFRAMES, OTHER OR NOT SPECIFIED'], 'Beds')
    stairs_str = create_str(all_products, 'product', ['1842 - STAIRS OR STEPS'], 'Stairs')
    chairs_str = create_str(all_products, 'product', ['4074 - CHAIRS, OTHER OR NOT SPECIFIED'], 'Chairs')
    baths_str = create_str(all_products, 'product', ['611 - BATHTUBS OR SHOWERS', '649 - TOILETS'], 'Baths/Showers')

    products = [stairs_str, baths_str, beds_str, chairs_str, floors_str]
    
    # Create the fishbone diagram
    header = 'FALLS'

    categories = {
        fac_str: facilities,
        event_str: events,
        act_str: activities,
        product_str: products,
        ad_str: alcohol_drug,
        fire_str: fire_involvement
    }

    draw_body(categories, header)
    plt.show()
    return categories


# ## Code to Convert Result from Dictionary to Dataframe

# The convert procedure converts the category dictionary to an ordered dataframe.  It is used to create the top ten cause rankings.

def convert(cat_dict):
    '''
    Input:
        cat_dict - a dictionary of category headings and their corresponding spine strings
    Output:
        final - a dataframe ordered by descending percentages
    '''
    cause_list = []
    pct_list = []
    cause_dict = {}

    for k, v in cat_dict.items():
        end = k.rfind('-') - 1
        header = k[0:end]
        for i in range(len(v)):
            detail = v[i]
            end = detail.rfind('-')
            name = detail[0:end]
            start = end + 1
            pct = int(detail[start:-1])
            cause = header + ' - ' + name
            cause_list.append(cause)
            pct_list.append(pct)

    cause_dict['cause'] = cause_list
    cause_dict['pct'] = pct_list
    result = pd.DataFrame.from_dict(cause_dict)
    result = result.sort_values(by=['pct'], ascending=False)
    final = result.reset_index(drop=True)
    return final


# ## Results

# The final results are summarized at the end of the analysis, but let's take a look at the individual results.

# ###  Breakdown of Falls by Fall, Trip and Slip

# The type of falls can be broken down into slips (15%), trips (16%) and other falls (67%).

df['method'].value_counts(normalize=True)


# ### Breakdown of Falls by Facility (.1% or More)

# While the location feature shows that about 60% of all falls occur at home, 38% occur in the non-descriptive PUBLIC or UNK locations.  The facility feature extracts the type of care facility involved in the fall from the narrative.  Home falls are excluded from causes as the environment is under the control of the patient.  Care facilities, however, can contribute to falls as they involve factors out of the control of the patient such as maintenance and staffing. 

df['location'].value_counts(normalize=True)


df['facility'].value_counts(normalize=True).loc[lambda x : x>.001]


# ### Breakdown of Falls by Event (.1% or More)

# About 13% of all falls involve a medical event.

df['event'].value_counts(normalize=True).loc[lambda x : x>.001]


# ### Breakdown of Falls by Activity (.1 or More)

# About 22% of all falls involve an activity.

df['activity'].value_counts(normalize=True).loc[lambda x : x>.001]


# ### Breakdown of Falls by Product (.1% or More)

# While about 13% of all falls involve floors, they have been excluded from causes in this analysis as the narratives clearly show that in the vast majority of cases the floors are what the patient hits upon falling.  Falls that involve floors are accounted for in this analysis as slips or trips.  Also rugs and carpets are still included as products. 

product_series = pd.concat([df['product_1'], df['product_2'], df['product_3']])
frame = {'product': product_series}
all_products = pd.DataFrame(frame)
all_products['product'].value_counts(normalize=True).loc[lambda x : x>.001]


# ### Diagram Showing Causes for All Falls

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

all_cats = create_categories(df)


# ### All Falls - Top Ten Causes

all_causes = convert(all_cats)
all_causes.head(10)


# ### Diagram Showing Causes of Falls Due to Slipping

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

slip_df = df[df['method'] == 'SLIP']
slip_cats = create_categories(slip_df)


# ### Slips - Top Ten Causes

slip_causes = convert(slip_cats)
slip_causes.head(10)


# ### Diagram Showing Causes of Falls Due to Tripping

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

trip_df = df[df['method'] == 'TRIP']
trip_cats = create_categories(trip_df)


# ### Trips - Top Ten Causes

trip_causes = convert(trip_cats)
trip_causes.head(10)


# ### Diagram Showing Causes of Falls Due to Falling Excluding Slipping and Tripping

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

fall_df = df[df['method'] == 'FALL']
fall_cats = create_categories(fall_df)


# ### Falls Only - Top Ten Causes

fall_causes = convert(fall_cats)
fall_causes.head(10)


# ### Diagram Showing Causes of Serious Injuries for All Falls

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

severe_df = df[(df['disposition'] == '4 - TREATED AND ADMITTED/HOSPITALIZED') | \
               (df['disposition'] == '2 - TREATED AND TRANSFERRED') ]
serious_cats = create_categories(severe_df)


# ### Serious Injuries - Top Ten Causes

serious_causes = convert(serious_cats)
serious_causes.head(10)


# ### Diagram Showing Causes of All Falls for Males

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

male_df = df[df['sex'] == 'MALE']
male_cats = create_categories(male_df)


# ### Male Falls - Top Ten Causes

male_causes = convert(male_cats)
male_causes.head(10)


# ### Diagram Showing Causes of All Falls for Females

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

female_df = df[df['sex'] == 'FEMALE']
female_cats = create_categories(female_df)


# ### Female Falls - Top Ten Causes

female_causes = convert(female_cats)
female_causes.head(10)


# ### Diagram Showing Causes of All Falls for Ages 60-69

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

df_60_69 = df[df['age'] < 70]
cats_60_69 = create_categories(df_60_69)


# ### Age 60-69 Falls - Top Ten Causes

causes_60_69 = convert(cats_60_69)
causes_60_69.head(10)


# ### Diagram Showing Causes of All Falls for Ages 70-79

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

df_70_79 = df[(df['age'] >= 70) & (df['age'] <= 79)]
cats_70_79 = create_categories(df_70_79)


# ### Age 70-79 Falls - Top Ten Causes

causes_70_79 = convert(cats_70_79)
causes_70_79.head(10)


# ### Diagram Showing Causes of All Falls for Ages 80-89

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

df_80_89 = df[(df['age'] >= 80) & (df['age'] <= 89)]
cats_80_89 = create_categories(df_80_89)


# ### Age 80-89 Falls - Top Ten Causes

causes_80_89 = convert(cats_80_89)
causes_80_89.head(10)


# ### Diagram Showing Causes of All Falls for Ages 90+

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

df_90 = df[df['age'] > 89]
cats_90 = create_categories(df_90)


# ### Age 90+ Falls - Top Ten Causes

causes_90 = convert(cats_90)
causes_90.head(10)


# ### Diagram Showing Causes of All Falls for Whites

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

white_df = df[df['race'] == 'WHITE']
white_cats = create_categories(white_df)


# ### Race White Falls - Top Ten Causes

white_causes = convert(white_cats)
white_causes.head(10)


# ### Diagram Showing Causes of All Falls for Blacks

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

black_df = df[df['race'] == 'BLACK/AFRICAN AMERICAN']
black_cats = create_categories(black_df)


# ### Race Black Falls - Top Ten Causes

black_causes = convert(black_cats)
black_causes.head(10)


# ### Diagram Showing Causes of All Falls for Hispanics

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

hispanic_df = df[df['hispanic'] == 'Yes']
hispanic_cats = create_categories(hispanic_df)


# ### Race Hispanic Falls - Top Ten Causes

hispanic_causes = convert(hispanic_cats)
hispanic_causes.head(10)


# ### Diagram Showing Causes of All Falls for Asians

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

asian_df = df[df['race'] == 'ASIAN']
asian_cats = create_categories(asian_df)


# ### Race Asian Falls - Top Ten Causes

asian_causes = convert(asian_cats)
asian_causes.head(10)


# ### Diagram Showing Causes of All Falls for 2019

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

df['year'] = df['treatment_date'].str[:4]
df_2019 = df[df['year'] == '2019']
cats_2019 = create_categories(df_2019)


# ### Year 2019 Falls - Top Ten Causes

causes_2019 = convert(cats_2019)
causes_2019.head(10)


# ### Diagram Showing Causes of All Falls for 2020

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

df_2020 = df[df['year'] == '2020']
cats_2020 = create_categories(df_2020)


# ### Year 2020 Falls - Top Ten Causes

causes_2020 = convert(cats_2020)
causes_2020.head(10)


# ### Diagram Showing Causes of All Falls for 2021

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

df_2021 = df[df['year'] == '2021']
cats_2021 = create_categories(df_2021)


# ### Year 2021 Falls - Top Ten Causes

causes_2021 = convert(cats_2021)
causes_2021.head(10)


# ### Diagram Showing Causes of All Falls for 2022

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

df_2022 = df[df['year'] == '2022']
cats_2022 = create_categories(df_2022)


# ### Year 2022 Falls - Top Ten Causes

causes_2022 = convert(cats_2022)
causes_2022.head(10)


# ## Summaries

# The following summaries by type of fall (all falls, slips, trips, other falls and serious falls), falls by sex (male, female), falls by age (60-69, 70-79, 80-89, 90+), falls by race (white, black, hispanic, asian), and by falls by year (2019, 2020, 2021, 2022) provide the key findings for this analysis.

# ### Falls by Type - Top Ten Causes

# As detailed above, the type of falls can be broken down to slips (15%), trips (16%) and other falls (67%). For slips, baths/showers play a prominent role when compared to the other types of falls. For trips, waking/hiking is the top cause with rugs/carpets also playing a prominent role. For all other falls, nursing homes are the top cause factor.  For serious falls (hospitalization), fainting is the top cause.

def highlight_cells(val):
    color = ''
    if type(val) == str:
        if 'Activity - Walking/Hiking' in val:
            color = 'lightgray'
        if 'Products - Baths/Showers' in val:
            color = 'lightskyblue'
        if 'Products - Rugs/Carpets' in val:
            color = 'gold'
        if 'Facility - Nursing Home' in val:
            color = 'lawngreen'
        if 'Medical Event - Fainting' in val:
            color = 'lightcoral'
    return 'background-color: {}'.format(color)

all_causes = all_causes.rename(columns={'cause': 'ALL CAUSES', 'pct': 'A PCT'})
all_causes['SLIP CAUSES'] = slip_causes['cause']
all_causes['S PCT'] = slip_causes['pct']
all_causes['TRIP CAUSES'] = trip_causes['cause']
all_causes['T PCT'] = trip_causes['pct']
all_causes['FALL CAUSES'] = fall_causes['cause']
all_causes['F PCT'] = fall_causes['pct']
all_causes['SERIOUS CAUSES'] = serious_causes['cause']
all_causes['R PCT'] = serious_causes['pct']

all_causes.head(10).style.applymap(highlight_cells)


# ### Falls by Sex - Top Ten Causes

# Males experience more falls due to fainting than females.  Also, drug and alcohol are involved in more male falls.

def highlight_cells(val):
    color = ''
    if type(val) == str:
        if 'Medical Event - Fainting' in val:
            color = 'lightskyblue'
        if 'Alcohol/Drug' in val:
            color = 'gold'  
    return 'background-color: {}'.format(color)

male_causes = male_causes.rename(columns={'cause': 'MALE CAUSES', 'pct': 'M PCT'})
male_causes['FEMALE CAUSES'] = female_causes['cause']
male_causes['F PCT'] = female_causes['pct']

male_causes.head(10).style.applymap(highlight_cells)


# ### Falls by Age - Top Ten Causes

# The involvement of nursing homes in falls increases significantly once people reach their 80's.  Interestingly, alcohol plays a prominent role in falls in the 60's but drops significantly once people reach their 70's.

def highlight_cells(val):
    color = ''
    if type(val) == str:
        if 'Facility - Nursing Home' in val:
            color = 'lightskyblue'
        if 'Alcohol/Drug - Alcohol' in val:
            color = 'gold'  
    return 'background-color: {}'.format(color)

causes_60_69 = causes_60_69.rename(columns={'cause': 'AGE 60-69 CAUSES', 'pct': '60s PCT'})
causes_60_69['AGE 70-79 CAUSES'] = causes_70_79['cause']
causes_60_69['70s PCT'] = causes_70_79['pct']
causes_60_69['AGE 80-89 CAUSES'] = causes_80_89['cause']
causes_60_69['80s PCT'] = causes_80_89['pct']
causes_60_69['AGE 90+ CAUSES'] = causes_90['cause']
causes_60_69['90s PCT'] = causes_90['pct']

causes_60_69.head(10).style.applymap(highlight_cells)


# ### Falls by Race - Top Ten Causes

# Fainting plays a more prominent role in non-white races while nursing homes play a more prominent role in whites.

def highlight_cells(val):
    color = ''
    if type(val) == str:
        if 'Medical Event - Fainting' in val:
            color = 'lightskyblue'
        if 'Facility - Nursing Home' in val:
            color = 'gold'  
    return 'background-color: {}'.format(color)

white_causes = white_causes.rename(columns={'cause': 'WHITE CAUSES', 'pct': 'W PCT'})
white_causes['BLACK CAUSES'] = black_causes['cause']
white_causes['B PCT'] = black_causes['pct']
white_causes['HISPANIC CAUSES'] = hispanic_causes['cause']
white_causes['H PCT'] = hispanic_causes['pct']
white_causes['ASIAN CAUSES'] = asian_causes['cause']
white_causes['A PCT'] = asian_causes['pct']

white_causes.head(10).style.applymap(highlight_cells)


# ### Falls by Year - Top Ten Causes

# The causes of falls are fairly consistent over time with a bump in drug involvement in 2020 and 2021.

def highlight_cells(val):
    color = ''
    if type(val) == str:
        if 'Alcohol/Drug - Drug' in val:
            color = 'lightskyblue'
    return 'background-color: {}'.format(color)

causes_2019 = causes_2019.rename(columns={'cause': '2019 CAUSES', 'pct': '2019 PCT'})
causes_2019['2020 CAUSES'] = causes_2020['cause']
causes_2019['2020 PCT'] = causes_2020['pct']
causes_2019['2021 CAUSES'] = causes_2021['cause']
causes_2019['2021 PCT'] = causes_2021['pct']
causes_2019['2022 CAUSES'] = causes_2022['cause']
causes_2019['2022 PCT'] = causes_2022['pct']

causes_2019.head(10).style.applymap(highlight_cells)




