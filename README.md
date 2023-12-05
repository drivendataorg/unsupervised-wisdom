[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

<img src='https://s3.amazonaws.com/drivendata-public-assets/cdc-medical-record.jpg' width='600' alt="Blue folder of medical records with orange stethoscope laid on top">

# <Unsupervised Wisdom: Explore Medical Narratives on Older Adult Falls>

## Goal of the Competition

Falls among adults 65 and older are the leading cause of injury-related deaths. Medical record narratives are a rich yet under-explored source of potential insights about how, when, and why people fall. However, narrative data sources can be difficult to work with, often requiring carefully designed, time-intensive manual coding procedures.

**The goal in this challenge was to identify effective methods of using unsupervised machine learning to extract insights about older adult falls from emergency department narratives.** Insights extracted from medical record narratives can potentially inform interventions for reducing falls.

## What's in this Repository

This repository contains code from winning competitors in the [Unsupervised Wisdom: Explore Medical Narratives on Older Adult Falls](https://www.drivendata.org/competitions/217/cdc-fall-narratives/) DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Summary of Approach
--- | --- | ---
1   | [artvolgin](https://www.drivendata.org/users/artvolgin/) | Using a downsampling method with ChatGPT and ML techniques, we obtained a full NEISS dataset across all accidents and age groups from 2013-2022 with six new variables: fall/not fall, prior activity, cause, body position, home location, and facility. We showed how these new indicators vary by age and sex for the fall-related cases and how they impact the likelihood of falls and post-fall hospitalization of older people. We also revealed seasonal and yearly trends in falls and provided a comparative age-based perspective, underscoring the value of the full ED narratives dataset in studying falls of older adults.
2   | [zysymu](https://www.drivendata.org/users/zysymu/) | An information extraction pipeline was developed to aid in obtaining more specific details about what happened in the fall events. Inspired by TabLLM I created a custom prompt in a human-readable format for each data point, combining information contained on tabular variables and event narratives (which are "translated" so that technical terms are easier to understand). Six questions, each related to a different characteristic of the fall (cause, diagnosis, etc.), are appended to the prompts and fed to a Text2Text Generation LLM that extracts information by answering the questions. Finally, I train one LDA model for each kind of question to model the topics of the answers, creating a granular way of analyzing and recovering falls given their different characteristics.
3   | [saket](https://www.drivendata.org/users/saket/) | We leveraged the ‘OpenAI’s text-embedding-ada-002’ embeddings provided by DrivenData to create a vector store in FAISS (Facebook Semantic Search). This allowed us to query the narrative using range search that returned relevant results. A key part of our solution involved meaningful context-aware and narrative (limited character) imposed query construction and understanding/categorizing actions that cause a fall. We also dug into the coding manual to understand how the narrative was structured and how best to extract precipitating events. We compared our solution to samples from OpenAI Chat GPT 3.5 and concluded ours was just as effective.
4   | [xvii](https://www.drivendata.org/users/xvii/) | My approach made use of embeddings, dimensionality reduction, clustering algorithms, network graphs and text summarization techniques to effectively identify and understand themes from medical narratives on older adults falls. The clustering analysis employed DBSCAN algorithm in conjunction with UMAP processing on the dimension reduced (PCA) embeddings data to uncover key themes cluster. Network graphs were used to explore keyword pair occurrences within narratives for the different clusters, with keyword ranking via the PageRank algorithm highlighting significant terms. Text ranking and summarization were applied to generate summaries for narrative clusters to provide insights into key themes.

## Bonus Prizes

Prize | Team or User
--- | ---
Most novel approach | [SeaHawk](https://github.com/drivendataorg/unsupervised-wisdom/blob/main/Bonus%3A%20Most%20novel%20approach/notebooks/Final%20Submission%20-%20Seahawk.ipynb)
Most compelling insight | [artvolgin](https://github.com/drivendataorg/unsupervised-wisdom/blob/main/1st%20Place/reports/Executive-Summary.pdf)
Best visualization | [jimking100](https://github.com/drivendataorg/unsupervised-wisdom/blob/main/Bonus%3A%20Best%20visualization/notebooks/final_submission_notebook%20-%20JImKing100.ipynb)
Best null result | [Research4Good](https://github.com/drivendataorg/unsupervised-wisdom/blob/main/Bonus%3A%20Best%20null%20result/reports/Executive_Summary.pdf)
Most helpful shared code | [zysymu](https://www.drivendata.org/competitions/217/cdc-fall-narratives/community-code/13/)
Most promising mid-point submissions | [RyoyaKatafuchi](https://github.com/drivendataorg/unsupervised-wisdom/blob/main/Bonus%3A%20Most%20promising%20midpoint%20submission/reports/Midpoint-Executive-Summary.pdf), [artvolgin](https://github.com/drivendataorg/unsupervised-wisdom/blob/main/1st%20Place/reports/Midpoint_Executive_Summary.pdf), and [saket](https://github.com/drivendataorg/unsupervised-wisdom/blob/main/3rd%20Place/reports/Midpoint_Executive_Summary.pdf)

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Winners Blog Post: [Meet the Winners of the Unsupervised Wisdom Challenge](https://drivendata.co/blog/unsupervised-wisdom-winners)**
