# App for analyzing tweets through topic modeling and sentiment analysis.

This repository contains the code of my Master’s Degree Final Project. This project aims to provide a pipeline (from extracting tweets to designing an interactive app) to explore a group of tweets with visual widgets and machine learning techniques such as topic modeling and sentiment analysis.

This pipeline consists in:
* Extracting tweets from Twitter with [snscrape](https://github.com/JustAnotherArchivist/snscrape)
* Preprocessing tweets and their metadata with well-known libraries such as [pandas](https://pandas.pydata.org/) or [spacy](https://spacy.io/)
* Compute tweets embeddings with [sentence transformers](https://www.sbert.net/). These contextual embeddings will improve topic modeling compared to classical techniques like LDA and also allow us to build a simple logistic regression for sentiment classification
* Train a sentiment analysis model with labeled datasets  
* Clustering tweets and assigning them topics with [contextualized topic modeling](https://github.com/MilaNLProc/contextualized-topic-models)
* Build an interactive app with [streamlit](https://streamlit.io/) and [plotly](https://plotly.com/python/)

In this project I use two different group of tweets: tweets from @IbaiLlanos and spanish tweets with the keyword 'netflix'. <br>
**App in spanish deployed on** https://tweets-visualizer.streamlit.app/

<img src="https://github.com/mortfer/TFM/blob/master/Ibai_example.png" width="850"/>

## Repository structure
* **app** contains scripts to deploy streamlit's app in Heroku
* **data** contains all the data needed along the process, from raw data extracted with snscrape to embeddings and datasets with sentiment labels  
* **dev** contains scripts for local development: preprocessing, creating embeddings, training sentiment model and topic modeling
 
## Using the code

Notebooks for sentiment classification in dev/sentiment_model are run once to train a model and use it for every group of tweets.
GPU is highly recommended when computing embeddings and creating topics <br>
Whenever we want to analyze a new group of tweets: 
* First, from data/raw_data folder extract tweets with snscrape's commands. Example used: <br>
`snscrape --jsonl --progress twitter-search "from:IbaiLlanos -filter:replies AND -filter:quote" > IbaiLlanos.json`
* Run main.py in dev/ for preprocessing, embeddings, infering sentiment and saving results <br>
`python main.py --data_name IbaiLlanos`
* Run main_opics.py in dev/ for topics creation. Results saved in data/topic_data <br>
`python main_topics.py --data_name IbaiLlanos`
* Finally, choose which data and script to use in app/app.py and run it <br>
`streamlit run app.py`


