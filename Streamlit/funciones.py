import streamlit as st
import pandas as pd
pd.set_option('max_colwidth',250)
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import json
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_plotly_events import plotly_events
from collections import OrderedDict
import datetime
import copy
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
#Aunque siempre se utilizará get_data() para el df_orig, allow_output_mutation=True para que la operación sea mucho más rápida
#ya que no tiene que comprobar que el objeto cacheado es igual al calculado. Pero ahora hay que ser más cuidadoso de no mutar el objeto.
#Del orden de ms con allow_output_mutation
@st.cache(allow_output_mutation=True)  #static data
def get_data():
    df=pd.read_parquet('../preprocessed_data/IbaiLlanos_limpio.pq')
    df.drop(['inReplyToTweetId','inReplyToUser','conversationId','id','quotedTweet'],axis=1,inplace=True)
    #df['date_str']=df['date'].dt.strftime('%Y-%m-%d %H:%M')
    

    with open('../sentiment_data/IbaiLlanos_sentiment.npy','rb') as f:
        sentiments=np.load(f)
    num_to_sentiment={-1:'Negativo',0:'Neutral',1:'Positivo'}
    sentiments=[num_to_sentiment[sentiment] for sentiment in sentiments]
    df['Sentimiento']=sentiments  
    with open('../topic_data/IbaiLlanos_topics.json','rb') as f:
        topics=json.load(f) 
    vocab=topics['vocab']    
    df['Cluster']=np.argmax(topics['doc_topic_dists'],axis=1)
    df['Cluster_prob']=np.max(topics['doc_topic_dists'],axis=1)
    N=30 #número de palabras representando a cada topic
    indices_words=np.flip(np.argsort(topics['topic_term_dists'], axis=1),axis=1)[:,:N]
    topic_terms=[]
    for i in indices_words:
        tmp=[]
        for j in i:
            tmp.append(vocab[j])
        topic_terms.append(tmp)
    
    term_frequency=topics['term_frequency']  

    return df,topic_terms,vocab,term_frequency

#Parece que allow_output_mutation no hace que vayan más rápido estas funciones. Si es así, va más rápido sin cachear
#@st.cache(max_entries=15,ttl=24*60*60,allow_output_mutation=True)  
def get_fechas(df):
    fechas = pd.unique(df['date'])
    #set() para quitar duplicados, list para ordenar con numpy y luego vuelvo a list
    #[:-15] para quitar minutos,horas y segundos y quedarse con dia,mes y año
    #Otra opción .strftime("%Y-%m-%d")
    fechas = list(np.sort(list(set([str(x)[:-15] for x in fechas]))))
    return fechas 

#@st.cache(max_entries=15,ttl=24*60*60,allow_output_mutation=True)
def df_filtered(df,start_date,end_date,text_in_tweet,clusters,langs,outlinks,source):
    
    df_filtrado=df[(df['date']> "{}".format(start_date)) & 
            (df['date']< "{}".format(end_date))]
    if text_in_tweet.strip()!='':
        df_filtrado=df_filtrado[(df_filtrado['content_embeddings'].apply(lambda x: text_in_tweet.lower().strip() in str(x).lower()))
                                |(df_filtrado['content_tokens'].apply(lambda x: text_in_tweet.lower().strip() in str(x).lower()))]
    if len(clusters)!=0:                            
        df_filtrado=df_filtrado[df_filtrado['Cluster'].isin(clusters)] 
    if len(langs)!=0:
        df_filtrado=df_filtrado[df_filtrado['lang'].isin(langs)]
    if len(outlinks)!=0:
        df_filtrado=df_filtrado[df_filtrado['outlinks'].apply(lambda x: True if set(x).intersection(set(outlinks)) else False)]
    if len(source)!=0:                            
        df_filtrado=df_filtrado[df_filtrado['sourceLabel'].isin(source)]    
        
    return df_filtrado


#@st.cache(max_entries=15,ttl=24*60*60,allow_output_mutation=True)
def cluster_figure(df_orig,df,topic_terms):  
    cluster_counts_orig=df_orig['Cluster'].value_counts(ascending=True)
    clusters_indices=np.array(cluster_counts_orig.index).astype(str)

    pos_to_cluster=OrderedDict()

    for pos,cluster in enumerate(cluster_counts_orig.index):
        pos_to_cluster[pos]=cluster

    cluster_counts_tmp=df['Cluster'].value_counts()
    cluster_counts_tmp=dict(zip(cluster_counts_tmp.keys().tolist(),cluster_counts_tmp.tolist()))
    for i in list(pos_to_cluster.keys()):
        if i not in cluster_counts_tmp.keys():
            cluster_counts_tmp[i]=0

    cluster_counts_nuevos=[]
    for i in range(len(pos_to_cluster)):
        cluster_counts_nuevos.append(cluster_counts_tmp[pos_to_cluster[i]])

    trace0=go.Bar(y=clusters_indices,
            x=list(cluster_counts_orig),
            orientation='h',
            marker_color=px.colors.qualitative.Dark24,name='original').update(opacity=0.25)

    trace1=go.Bar(y=clusters_indices,
                x=list(cluster_counts_nuevos),
                orientation='h',
                marker_color=px.colors.qualitative.Dark24,name='filtrado')
    data=[trace0,trace1]

    annotations = [dict(
                x=0.1,
                y=y,
                text=",".join(topic_terms[cluster][:6]),
                xanchor='left',
                yanchor='middle',
                showarrow=False,
                font=dict(color="#ffffff")
            ) for y,cluster in pos_to_cluster.items()]

    layout=go.Layout(title='Clusters detectados automáticamente',title_x=0.5,title_y=0.93,xaxis_title="Tweets",
                        yaxis_title="Clusters",barmode='overlay',
                        legend=dict(itemclick="toggleothers",itemdoubleclick="toggle",xanchor='auto',yanchor='auto',x=1,y=.05),#Adjust click behavior
                        annotations=annotations,
                        margin=go.layout.Margin(l=0,r=50, b=0,t=60),
                        plot_bgcolor='#15202b',
                        paper_bgcolor='#15202b',
                        font = dict(color = '#ffffff'),
                        height=len(cluster_counts_orig)*20+200,
                        )



    cluster_counts_fig= go.Figure(data=data,
                                  layout=layout)
    return cluster_counts_fig

def to_1D(series):
    return pd.Series([x for _list in series for x in _list])


def value_counts_figure(df,columna,top,title):
    if columna in ['hashtags','mentionedUsers','cashtags','outlinks']:
        tmp=to_1D(df[columna])
    else:
        tmp=df[columna]
    column_counts=tmp.value_counts(sort=True,ascending=True)[-top:]   
    
    layout=go.Layout(title=title,title_x=0.5,title_y=0.85,margin=go.layout.Margin(l=15,r=0, b=0,t=50),
                    plot_bgcolor='#15202b',
                    paper_bgcolor='#15202b',
                    font = dict(color = '#ffffff'), 
                    #width=500,
                    height=len(column_counts)*15+105 #Un mínimo para cuando hay una sola barra que no se corte la figura
                    )
    column_counts_fig= go.Figure(go.Bar(y=column_counts.index,x=list(column_counts),
                                        orientation='h',marker_color='#1d9bf0'
                                        ),
                                layout=layout
                                )
    return column_counts_fig

def hourday_figure(df_orig,df):
    hourday_counts_orig=df_orig.groupby(df_orig['date'].dt.hour).size() #Hora del día
    hourday_counts_orig_dict=dict(zip(hourday_counts_orig.index.astype('int').tolist(),hourday_counts_orig.values.tolist()))#Lo paso a diccionario
    hourday_counts=df.groupby(df['date'].dt.hour).size() #Hora del día
    hourday_counts_dict=dict(zip(hourday_counts.index.astype('int').tolist(),hourday_counts.values.tolist()))#Lo paso a diccionario
    #Relleno con 0's las horas que no estén
    for i in range(24):
        if i not in hourday_counts_orig.keys():
            hourday_counts_orig_dict[i]=0
        if i not in hourday_counts.keys():
            hourday_counts_dict[i]=0
    #Figura
    trace0=go.Bar(y=list(hourday_counts_orig_dict.values()),
            x=list(hourday_counts_orig_dict.keys()),
            marker_color='#1d9bf0',name='original'
            ).update(opacity=0.25)

    trace1=go.Bar(y=list(hourday_counts_dict.values()),
                x=list(hourday_counts_dict.keys()),
                marker_color='#1d9bf0',name='filtrado'
            )

    layout=go.Layout(title='Tweets a lo largo del día',title_x=0.5,barmode='overlay',margin=go.layout.Margin(l=0,r=0, b=0,t=50),
                     xaxis_title="h",height=300,legend=dict(itemclick="toggleothers",itemdoubleclick="toggle",xanchor='auto',yanchor='middle',bgcolor='rgba(0,0,0,0)'),
           )
    data=[trace0,trace1]
    hourday_fig= go.Figure(data=data,layout=layout)

    return hourday_fig


def representative_tweets(df_tmp):
    df=df_tmp.copy(deep=True)
    df=df[['renderedContent','Cluster','Cluster_prob']]
    df.sort_values(by=['Cluster_prob'],ascending=False,inplace=True)

    longitud=df.Cluster.unique().size*15
    if longitud>40:
        longitud=40

    df=df.round({'Cluster_prob':2}).head(longitud)
    fig = go.Figure(data=[go.Table(
    columnwidth = [0.76,0.1,0.14],
    header=dict(values=['Tweets','Cluster','Confianza'],
                fill_color='#192734',
                align='left'),
    cells=dict(values=[df.renderedContent,df.Cluster,df.Cluster_prob],
               fill_color='#15202b',line_color='#15202b',
               align='left',font=dict(color='#FFFFFF')))
    ])
    fig.update_layout(title='Tweets más representativos del cluster...',margin=go.layout.Margin(l=0,r=10, b=0,t=40))
    return fig
  


def relevant_terms_table(df):
    longitud=df.Cluster.unique().size*15
    if longitud>50:
        longitud=50
    df_fig=pd.DataFrame(df.content_tokens.str.split(expand=True).stack().value_counts(),columns=['frecuencia']).head(longitud)

    fig = go.Figure(data=[go.Table(
    columnwidth = [0.65,0.35],
    header=dict(values=['término','frecuencia'],
                fill_color='#192734',
                align='left'),
    cells=dict(values=[df_fig.index,df_fig.frecuencia],
               fill_color='#15202b',line_color='#15202b',
               align='left',font=dict(color='#FFFFFF')))
    ])
    fig.update_layout(title='Términos más usados...',margin=go.layout.Margin(l=10,r=20, b=0,t=40))
    return fig

def temporal_fig(df_filtered,cluster_to_pos,clusters_filtro,colores):
    NumClusters=copy.deepcopy(clusters_filtro)
    if len(clusters_filtro)==0:
        NumClusters.append(-1)    
    df=df_filtered.copy(deep=True)
    df=df.set_index('date')
    cols=['replyCount','retweetCount','likeCount','quoteCount','Photo','Video','Cluster']
    df=df[cols]
    df.columns=['Respuestas','Retweets','Likes','Citas','Fotos','Videos','Cluster']
    if (df.index.max()-df.index.min())>datetime.timedelta(days=365):
        escalas=['Y','M','W']
    else:
        escalas=['M','W','D']

    fig=go.Figure()
    dfs=[]
    for cluster in NumClusters:
        if cluster==-1:
            color='#1d9bf0' 
        else:
            color=colores[cluster_to_pos[cluster]]
        for escala in escalas:
            if cluster==-1:
                df_tmp=df.resample(escala).mean()
                df_tmp.insert(0,'Tweets',df.resample(escala).size())
            else:
                df_tmp=df[df.Cluster==cluster].resample(escala).mean()
                df_tmp.insert(0,'Tweets',df[df.Cluster==cluster].resample(escala).size())
            df_tmp.drop(['Cluster'],axis='columns',inplace=True)
            df_tmp['Fotos']=100*df_tmp['Fotos']
            df_tmp['Videos']=100*df_tmp['Videos']
            df_tmp=round(df_tmp)
            dfs.append(df_tmp)
            visibility=(escala==escalas[0])
            fig.add_trace(go.Scatter(x=df_tmp.index,
                             y=df_tmp[df_tmp.columns[0]],
                             visible=visibility,
                             line=dict(shape='spline'),
                             #name=str(cluster)+str(escala),
                             marker=go.scatter.Marker(color=color),
                        hoverinfo='x+y+text'
                        )
            )
    buttonsCol=[]
    col_to_label=dict(Tweets='Número de tweets',Respuestas='Media de Respuestas',Retweets='Media de Retweets',
                    Likes='Media de Likes',Citas='Media de Citas',Fotos='Tweets con Fotos (%)',Videos='Tweets con Videos (%)'
                    )
    for col in dfs[0].columns:
        dfs_col=[df[col] for df in dfs]
        buttonsCol.append(dict(method='update',
                            label=col,
                            visible=True,
                            args=[{'y':dfs_col}, 
                                  {#"title": col,
                                   #"xaxis": {"title": "Date"},
                                    "yaxis": {"title": col_to_label[col]},
                                  },
                                  list(range(len(dfs)))]
                            )
                      )
    buttonsVisible = []
    escala_to_label=dict(Y='Año',M='Mes',W='Semana',D='Dia')
    for i,escala in enumerate(escalas):
        visibility=[False]*len(escalas)
        visibility[i]=True
        visibility=visibility*len(NumClusters)
        buttonsVisible.append(dict(label=escala_to_label[escala],method = 'restyle',args = ['visible',visibility]))

    updatemenu = []

    menuCol = dict(buttons=buttonsCol,direction='down',showactive=True,x=1,yanchor='bottom',xanchor='right',
                   bgcolor='rgba(29, 155, 240, .8)',font=dict(color='black')
                   #bgcolor es azul clarito de twitter pero reduciendo opacidad
            )
    updatemenu.append(menuCol)

    menuVis = dict(buttons=buttonsVisible,direction='right',showactive=True,xanchor='left',yanchor='bottom',
                    x=0,bgcolor='rgba(29, 155, 240, .8)',font=dict(color='black'),type='buttons'
            )

    updatemenu.append(menuVis)
    fig.update_layout(showlegend=False,updatemenus=updatemenu,margin=go.layout.Margin(l=0,r=30, b=0,t=60)
                        ,)#plot_bgcolor='#192734'
    fig.update_xaxes(title_text='Fechas')
    return fig

def df_tabla(df_tmp):
    df=df_tmp.copy(deep=True)
    df=df[['url','renderedContent','date','retweetCount','replyCount','quoteCount','Photo','Video','Cluster','Sentimiento']] 
    df.columns=['Url','Tweet','Fecha','Retweets','Respuestas','Citas','Foto','Video','Cluster','Sentimiento']
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    gb.configure_default_column(copyHeadersToClipboard=True,filterable=False,editable=False,resizable=False, groupable=False,)

    gridOptions = gb.build()
    AgGrid(df,gridOptions=gridOptions,enable_enterprise_modules=True)
    return None