# App for analyzing tweets through topic modeling and sentiment analysis.


This repository contains the code of my final master's thesis. This project aims to provide a pipeline (from extracting tweets to designing an interactive app) to explore a group of tweets 
with graphical widgets and machine learning techniques such as topic modeling and sentiment analysis.

This pipeline consists in:
* Extracting tweets from Twitter with [snscrape](https://github.com/JustAnotherArchivist/snscrape)
* Preprocessing tweets and their metadata with well-known libraries such as [pandas](https://pandas.pydata.org/) or [spacy](https://spacy.io/)
* Compute tweets embeddings with [sentence transformers](https://www.sbert.net/)
* Train a sentiment analysis model with labeled datasets  
* Clustering tweets and assigning them topics with [contextualized topic modeling](https://github.com/MilaNLProc/contextualized-topic-models)
* Build an interactive app with [streamlit](https://streamlit.io/) and [plotly](https://plotly.com/python/)

In this project I use two different group of tweets: tweets from @IbaiLlanos and spanish tweets with the keyword 'netflix'. <br>
**App in spanish deployed on** https://tfm-marc.herokuapp.com/

## Repository structure
* **app** contains scripts to deploy streamlit's app in Heroku
* **data** contains all the data needed along the process, from raw data extracted with snscrape to embeddings and datasets with sentiment labels  
* **dev** contains scripts for local development: preprocessing, creating embeddings, training sentiment model and topic modeling
 
## Using the code
`pruebas`
