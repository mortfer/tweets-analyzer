import json
import re
import numpy as np
import pandas as pd
import unidecode
from nltk.corpus import stopwords as stop_words
import string
import spacy
import time
from spacy.tokenizer import _get_regex_pattern


def clean_mentions(mentions):
    """
    Parameters: List of dicts with 'username' as key 
    Returns: List of values of username's keys
    """
    lista=[]
    for mention in mentions:
        lista.append(mention['username'])
    return lista


def clean_source(source):
    """
    Parameters: String
    Returns: String
    """
    if 'iPhone' in source:
        return 'iPhone'
    elif 'Android' in source:
        return 'Android'
    elif 'Web' in source:
        return 'Web'
    else:
        return 'Otros'


def inreplytoUser(columna):
    """
    Parameters: pandas column with dict in each row
    Returns: pandas column with value of username's key in each row
    """
    columna=columna[columna.notnull()].apply(lambda x:x['username'])
    return columna

def check_media(lista,tipo='Photo'):
    """
    Parameters: 
    -lista: list of dicts with '_type' as key
    -tipo: string to compare it with '_type' keys
    Returns: Boolean
    """
    if tipo=='Photo':
        tipo='snscrape.modules.twitter.Photo'
    elif tipo=='Video':
        tipo='snscrape.modules.twitter.Video'
    for i in lista:
        if i['_type']==tipo:
            return True
    return False

def webs_de_outlinks(links_list):
    """
    Function that gets the web page of a full link
    Parameters: 
    -links_list: list of strings
    Returns: list of strings
    """
    new_list=[]   
    for link in links_list:
        link=link.lower()        
        web=re.search('https?:\/\/(www\.)?([^\s\/]+)(\/)?(.*)$', link).group(2)
        #We only care about external webs
        if 'twitter' not in web:
            new_list.append(web)
    return new_list

def preprocessing_for_emebeddings(texto):
    """
    Function that process a text
    Parameters: string
    Returns: processed string
    """
    #Remove line breaks
    texto=re.sub('\n',' ',texto)
    #Remove links   
    texto=re.sub('(https?:\/\/\S+\.\S+)','',texto)
    #HTML 
    texto=re.sub(r'&gt;','>',texto)
    texto=re.sub(r'&lt;','<',texto)
    texto=re.sub(r'&[a-z]+;',' ',texto) 
    texto=texto.strip() 
    return texto

def preprocessing_for_tokens(texto,punctuation_list):
    """
    Function that processes a text
    Parameters: 
    -texto: string
    -punctuation_list: string of characters to remove from texto
    Returns: processed string
    """

    #lowercase
    texto=texto.lower()
    #Remove punctuations   
    texto=texto.translate(str.maketrans(punctuation_list,' '*len(punctuation_list)))
    #Remove number
    texto=re.sub(r'\b[0-9]+\b',' ',texto)   

    texto=texto.strip()
    return texto

def lemmatize_pipe(doc):
    """
    Parameters:
    -doc: list of spacy objects
    Returns: processed string
    """
    #Hashtag, mention or cashtag are not lemmatized
    lemma_list = [str(tok.lemma_) if tok.text[0] not in ['#','$','@'] else str(tok.text) for tok in doc] 
    return lemma_list

def preprocess_pipe(texts,nlp):
    """
    Parameters:
    -texts: iterable of texts
    -nlp: spacy model
    Returns: list of strings
    """
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=32):
        preproc_pipe.append(' '.join(lemmatize_pipe(doc)))
     
    return preproc_pipe

def json_to_dataframe(datos):
    """
    Parameters:
    -datos: pandas dataframe
    Returns: preprocessed pandas dataframe
    """
    #Load dataframe and remove useless fields
    datos = pd.read_json(datos, lines=True, encoding='utf-8').drop(
            ['_type','source','sourceUrl','tcooutlinks','place','coordinates','user','conversationId'],axis=1)
    
    datos=datos.drop_duplicates(subset=['id'])
    datos=datos[datos.retweetedTweet.isnull()]
    datos=datos.drop(['retweetedTweet','id'],axis=1)

    datos['sourceLabel']=datos['sourceLabel'].apply(lambda x: clean_source(x))
    
    #Fill empty fields with empty lists
    for i in ['cashtags','hashtags','mentionedUsers','media','outlinks']:
        datos[i]=datos[i].fillna("").apply(list)
    #Boolean columns    
    datos['Photo']=datos['media'].apply(lambda x: check_media(x,'Photo'))
    datos['Video']=datos['media'].apply(lambda x: check_media(x,'Video'))
    
    datos['mentionedUsers']=datos['mentionedUsers'].apply(lambda x:clean_mentions(x))
    datos['inReplyToUser']=inreplytoUser(datos['inReplyToUser'])
    #Boolean column 
    datos['quotedTweet']=np.where(datos['quotedTweet'].notnull(),True,False)
    datos['outlinks']=datos['outlinks'].apply(webs_de_outlinks)
    
    #New column for embeddings
    datos['content_embeddings']=datos['content'].apply(preprocessing_for_emebeddings)
    #New column for tokens
    datos['content_tokens']=datos['content_embeddings']
    #New token 'fotos_y_videos' added to tweets with photos or videos
    datos['content_embeddings']=np.where(datos['Photo'],datos['content_embeddings']+' fotos_y_videos',datos['content_embeddings'])
    datos['content_embeddings']=np.where(datos['Video'],datos['content_embeddings']+' fotos_y_videos',datos['content_embeddings'])
      
    datos=datos.drop(['content'],axis=1)
    
   
    #Preprocessing
    punctuation_list=string.punctuation.replace('_','').replace('@','').replace('#','').replace('$','')+'¡¿'
    datos['content_tokens']=datos['content_tokens'].apply(lambda x: preprocessing_for_tokens(x,punctuation_list))

    #Stopwords
    #Languages considered: spanish and english
    languages={'spanish':'es','english':'en'}
    for lang in languages.keys():
        stopwords=set(stop_words.words(lang))
        if lang=='spanish':
            #añado las palabras sin acentos como 'habeis' ya que en twitter es común escribir sin acentos.
            stopwords=stopwords.union(set(map(lambda x:unidecode.unidecode(x),stopwords)))
        datos.loc[datos.lang==languages[lang],'content_tokens']=datos.loc[datos.lang==languages[lang],'content_tokens'].apply(lambda x: 
                                                              ' '.join([w for w in x.split() if w not in stopwords]))
    #Spacy models for lemmatizing
    for lang_model in ['es_core_news_lg','en_core_web_lg']:
        nlp = spacy.load(lang_model, disable=['parser','ner'])#Remove some components we will not use
        nlp.tokenizer.token_match = re.compile(r'@?#?\$?\b\S\S+\b').match #Token pattern 
        lang=lang_model[:2]     
        #We use spacy batches for efficiency
        datos.loc[datos.lang==lang,'content_tokens']=preprocess_pipe(datos.loc[datos.lang==lang,'content_tokens'],nlp)
               
    #New token 'fotos_y_videos' added to tweets with photos or videos
    datos['content_tokens']=np.where(datos['Photo'],datos['content_tokens']+' fotos_y_videos',datos['content_tokens'])
    datos['content_tokens']=np.where(datos['Video'],datos['content_tokens']+' fotos_y_videos',datos['content_tokens'])
    datos['content_tokens']=datos['content_tokens'].apply(lambda x: x.strip())
    
    datos=datos.drop(['media'],axis=1)
    
    return datos
