import os
import argparse
import preprocessing_module as pr
import modeling_module as md
import numpy as np
import json
import time
def main():
    time0=time.time()
    #Principal script for preprocessing tweets, creating their embeddings and inferring sentiment. Due to some gpu errors,
    #topic module has to be run in a separate script.
    parser = argparse.ArgumentParser(description=\
        'Process tweets with preprocessing techniques and creating embeddings. Next step is topic modeling')

    required = parser.add_argument_group('required arguments')

    required.add_argument('--data_name',required=True,help='Argument to specify name of documents. For instance, if we have\
        IbaiLlanos.json file in ../data/raw_data/ \'--data_name IbaiLlanos\' is expected')

    
    parser.add_argument('--preprocessing',default= True, help='Boolean. Wether you want to make preprocessing step or not')
    parser.add_argument('--create_embeddings',default= True, help='Boolean. Wether you want to make embeddings step or not')
    parser.add_argument('--infer_sentiment',default= True, help='Boolean. Wether you want to make sentiment inference step or not')

    args = parser.parse_args()

    data_path='{}/../data'.format(os.path.dirname(os.path.abspath(__file__)))

    raw_data_path='{}/raw_data/{}.json'.format(data_path,args.data_name)
    preprocessed_data_path='{}/preprocessed_data/{}_preprocessed.pq'.format(data_path,args.data_name)
    embeddings_data_path='{}/embeddings_data/{}_embeddings.npy'.format(data_path,args.data_name)
    sentiment_data_path='{}/sentiment_data/{}_sentiment.npy'.format(data_path,args.data_name)

    sentiment_model_path='./sentiment_model/sentiment_model.sav'

    if args.preprocessing is True:
        print(str(round(time.time()-time0,2))+'s')
        print('Preprocessing step...')        
        preprocessed_data=pr.json_to_dataframe(raw_data_path)
        preprocessed_data.to_parquet(preprocessed_data_path)

    if args.create_embeddings is True:
        print(str(round(time.time()-time0,2))+'s')
        print('Embeddings step...') 
        embeddings=md.embeddings(preprocessed_data_path)
        with open(embeddings_data_path, 'wb') as f:
            np.save(f,embeddings)

    if args.infer_sentiment is True:
        print(str(round(time.time()-time0,2))+'s')
        print('Sentiment inference step...')
        predictions=md.sentiment_inference(sentiment_model_path,embeddings_data_path)
        with open(sentiment_data_path, 'wb') as f:
            np.save(f,predictions)
    print('Done! {}s'.format(round(time.time()-time0,2)))
    
if __name__ == '__main__':
    main()