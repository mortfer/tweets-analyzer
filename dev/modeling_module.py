import pandas as pd
import numpy as np
import time
import copy
import pickle


def embeddings(preprocessed_data_path):
        #Imports inside the function to avoid gpu errors      
        from sentence_transformers import SentenceTransformer

        preprocessed_data=pd.read_parquet(preprocessed_data_path,columns=['content_embeddings'])
        
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2") 
        #GPU highly recommended
        train_contextualized_embeddings=np.array(model.encode(list(preprocessed_data['content_embeddings']), 
                                            show_progress_bar=True,batch_size=64))
        return train_contextualized_embeddings 

def sentiment_inference(sentiment_model_path,embeddings_path):
        #Inference of sentiments as 1 (postive), 0 (neutral) or -1 (negative)
        with open(sentiment_model_path, 'rb') as f:
            sentiment_model = pickle.load(f)
        with open(embeddings_path, 'rb') as f:
            embeddings=np.load(f)
            
        predictions=np.array(sentiment_model.predict(embeddings))
        return predictions

