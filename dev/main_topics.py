import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import json
import argparse
import os 
import pandas as pd 
import time

def topic_modeling(preprocessed_data_path,embeddings_path): 
    time0=time.time()  
    #Imports inside function to avoid GPU errors    
    from contextualized_topic_models.models.ctm import CombinedTM
    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
    from contextualized_topic_models.evaluation.measures import CoherenceNPMI

    #Load dataframe and embeddings array
    preprocessed_data=pd.read_parquet(preprocessed_data_path,columns=['content_tokens'])     
    with open(embeddings_path, 'rb') as f:
            embeddings=np.load(f)

    #scikit object to convert a collection of text documents to a matrix of token counts.
    vectorizer = CountVectorizer(max_features=2000,ngram_range=(1,1), token_pattern=r'@?#?\$?\b\S\S+\b',
                    lowercase=False)
    train_bow_embeddings=vectorizer.fit_transform(list(preprocessed_data['content_tokens']))
    #Every token (word) considered in vectorizer
    vocab = vectorizer.get_feature_names()
    #Dict mapping int to token
    id2token = {k: v for k, v in zip(range(0, len(vocab)), vocab)}

    #Prepare contextualized_topic_models objects
    qt = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")        
    training_dataset = qt.load(embeddings, train_bow_embeddings, id2token)  
    #We consider various models with different number of topics in order to choose the model with highest npmi,
    #a topic coherence measure 
    models_list=[]
    coherence_list=[]
    #Entre 9 topics y 15 topics
    num_topics=[9,11,13,15]
    topwords=6 #Número de palabras que representan cada topic
    texts=[x.lower().split() for x in list(preprocessed_data['content_tokens'])]
    for comp in num_topics:
        print(str(round(time.time()-time0,2))+'s')
        print('{} topics...'.format(comp))
        #CTM call
        ctm = CombinedTM(bow_size=len(vocab), contextual_size=embeddings.shape[1], n_components=comp,num_epochs=5)
        ctm.fit(training_dataset,verbose=False)
        models_list.append(ctm)
        npmi = CoherenceNPMI(texts=texts, topics=ctm.get_topic_lists(topwords))
        coherence_list.append(npmi.score(topk=topwords))
    idxmax=coherence_list.index(max(coherence_list))
    final_model=models_list[idxmax]
    #Once we have the best model, we train it more epochs.
    print(str(round(time.time()-time0,2))+'s')
    print('We choose {} topics and we make more training'.format(num_topics[idxmax]))
    final_model.fit(training_dataset)
    final_model.fit(training_dataset)

    #CTM has a stochastic component so we infer the topic of each document 10 times and we average it 
    lda_vis_data = final_model.get_ldavis_data_format(vocab, training_dataset, n_samples=10)
    #Data from lda_vis_data:
    #topic_term_dists: distances between topics and each term
    #doc_topic_dists: distances between documents and each topics
    #doc_lengths: length of documents
    #vocab: number of tokens
    #term_frequency: frequency of each term in the corpus
    topic_data=dict()
    #Convert arrays to lists so we can save it in json format
    for key in lda_vis_data.keys():
        try:
            topic_data[key]=lda_vis_data[key].tolist()
        except:
            topic_data[key]=lda_vis_data[key]
    print('Done! {}s'.format(round(time.time()-time0,2)))
    return topic_data 

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Topic modeling with embeddings for further analysis in Streamlit')

    parser.add_argument('--data_name',required=True,help='Argument to specify name of documents. If we have\
        IbaiLlanos_preprocessed.pq and IbaiLlanos_embeddings files, \'--data_name IbaiLlanos\' is expected')

    args = parser.parse_args()

    data_path='{}/../data'.format(os.path.dirname(os.path.abspath(__file__)))
    preprocessed_data_path='{}/preprocessed_data/{}_preprocessed.pq'.format(data_path,args.data_name)
    embeddings_data_path='{}/embeddings_data/{}_embeddings.npy'.format(data_path,args.data_name)
    topic_data_path='{}/topic_data/{}_topics.json'.format(data_path,args.data_name)

    #Run topic models and dump it to json
    topic_data=topic_modeling(preprocessed_data_path,embeddings_data_path)
    with open(topic_data_path, "w") as f:
        json.dump(topic_data,f)