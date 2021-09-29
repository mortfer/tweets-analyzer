import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import json
import copy
from streamlit_plotly_events import plotly_events
from collections import OrderedDict
import datetime
import copy
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

#allow_output_mutation=True para que la operación de cachear sea mucho más rápida
#ya que no tiene que comprobar que el objeto cacheado es igual al calculado. Hay que ser más cuidadoso de no mutar el objeto.
@st.cache(allow_output_mutation=True) #static data
def get_data(data_name,data_path):

    #Directorios
    preprocessed_data_path='{}/preprocessed_data/{}_preprocessed.pq'.format(data_path,data_name)
    sentiment_data_path='{}/sentiment_data/{}_sentiment.npy'.format(data_path,data_name)
    topic_data_path='{}/topic_data/{}_topics.json'.format(data_path,data_name)

    df=pd.read_parquet(preprocessed_data_path)
    df.drop(['inReplyToTweetId','inReplyToUser','quotedTweet'],axis=1,inplace=True)

    #Cargo la predicción de cada fila, cambio el array numérico a palabras y añado al dataframe
    with open(sentiment_data_path,'rb') as f:
        sentiments=np.load(f)
    num_to_sentiment={-1:'Negativa',0:'Neutral',1:'Positiva'}
    sentiments=[num_to_sentiment[sentiment] for sentiment in sentiments]
    df['Opinión']=sentiments 

    #Cargo el json de topics
    with open(topic_data_path,'rb') as f:
        topics=json.load(f) 
    #Las palabras que pueden definir a los topics
    vocab=topics['vocab'] 
    #Se asigna el cluster que tenga mayor probabilidad   
    df['Cluster']=np.argmax(topics['doc_topic_dists'],axis=1)
    #Se guarda también esa probabilidad
    df['Cluster_prob']=np.max(topics['doc_topic_dists'],axis=1)
    N=30 #número de palabras representando a cada topic.
    #topics['topic_term_dists'] es array de tamaño numero_topics,len(vocab) y lo que hago es asignar a cada cluster
    #las palabras con mayor probabilidad
    indices_words=np.flip(np.argsort(topics['topic_term_dists'], axis=1),axis=1)[:,:N]
    #Hago lista de listas de palabras. La primera lista corresponde a las palabras del cluster 0
    topic_terms=[]
    for i in indices_words:
        tmp=[]
        for j in i:
            tmp.append(vocab[j])
        topic_terms.append(tmp)
    #Frecuencia de los términos
    term_frequency=topics['term_frequency']  

    return df,topic_terms,vocab,term_frequency


def df_filtered(df,start_date,end_date,text_in_tweet,clusters,langs,outlinks,source,opinion_filtro):
    """
    Parameters: df,start_date,end_date,text_in_tweet,clusters,langs,outlinks,source,opinion_filtro
    -df: dataframe
    -star_data,end_data: strings of dates
    -text_in_tweet: string
    -rest of parameters are lists in order to filter
    Returns: pandas dataframe
    """
    #Filtro de fecha
    df_filtrado=df[(df['date']> "{}".format(start_date)) & 
            (df['date']< "{}".format(end_date))]
    #Filtro de texto
    if text_in_tweet.strip()!='':
        df_filtrado=df_filtrado[(df_filtrado['content_embeddings'].apply(lambda x: text_in_tweet.lower().strip() in str(x).lower()))
                                |(df_filtrado['content_tokens'].apply(lambda x: text_in_tweet.lower().strip() in str(x).lower()))]
    #Filtros para las columnas correspondientes. Si están vacias, no lo compruebo.
    if len(clusters)!=0:                            
        df_filtrado=df_filtrado[df_filtrado['Cluster'].isin(clusters)] 
    if len(langs)!=0:
        df_filtrado=df_filtrado[df_filtrado['lang'].isin(langs)]
    if len(outlinks)!=0:
        df_filtrado=df_filtrado[df_filtrado['outlinks'].apply(lambda x: True if set(x).intersection(set(outlinks)) else False)]
    if len(source)!=0:                            
        df_filtrado=df_filtrado[df_filtrado['sourceLabel'].isin(source)]    
    if len(opinion_filtro)!=0:                            
        df_filtrado=df_filtrado[df_filtrado['Opinión'].isin(opinion_filtro)] 

    return df_filtrado


def cluster_figure(df_orig,df,topic_terms,colores,pos_to_cluster):
    """
    Figure to draw histogram of clusters. It draws from two dataframes, df_orig and df.
    Parameters: df_orig,df,topic_terms,colores,pos_to_cluster
    -topic_terms: list of lists with strings. Each nested list represents each cluster
    -colores: list of color codes
    -pos_to_cluster: dict mapping cluster number to an index in colores
    Returns: plotly figure
    """  
    #Value counts de df original
    cluster_counts_orig=df_orig['Cluster'].value_counts(ascending=True)
    #Value counts de df, se pasa a dict y se añaden los clusters que no estén con el valor 0
    cluster_counts_tmp=df['Cluster'].value_counts()
    cluster_counts_tmp=dict(zip(cluster_counts_tmp.keys().tolist(),cluster_counts_tmp.tolist()))
    for i in list(pos_to_cluster.keys()):
        if i not in cluster_counts_tmp.keys():
            cluster_counts_tmp[i]=0
    #Se reordenan los valores de los clusters según las posiciones de pos_to_cluster que deben ser 
    #las mismas que en cluster_counts_orig
    cluster_counts_nuevos=[]
    for i in range(len(pos_to_cluster)):
        cluster_counts_nuevos.append(cluster_counts_tmp[pos_to_cluster[i]])

    #Añado los value_counts del df original con una opacidad reducida
    trace0=go.Bar(y=[str(i) for i in pos_to_cluster.values()],
            x=list(cluster_counts_orig),
            orientation='h',
            marker_color=colores,name='original').update(opacity=0.25)

    #Añado los value_counts del df 
    trace1=go.Bar(y=[str(i) for i in pos_to_cluster.values()],
                x=list(cluster_counts_nuevos),
                orientation='h',
                marker_color=colores,name='filtrado')
    data=[trace0,trace1]

    #Añado las palabras de cada cluster
    annotations = [dict(
                x=0.1,
                y=y,
                text=",".join(topic_terms[cluster][:6]),
                xanchor='left',
                yanchor='middle',
                showarrow=False,
                font=dict(color="#ffffff")
            ) for y,cluster in pos_to_cluster.items()]
    #Arreglo layout para margenes, título,colores...
    layout=go.Layout(title='Clusters detectados',title_x=0.5,title_y=0.93,xaxis_title="Tweets",
                        yaxis_title="Clusters",barmode='overlay',
                        legend=dict(itemclick="toggleothers",itemdoubleclick="toggle",xanchor='auto',yanchor='auto',x=1,y=.05),#Adjust click behavior
                        annotations=annotations,
                        margin=go.layout.Margin(l=0,r=50, b=0,t=60),
                        plot_bgcolor='#15202b',
                        paper_bgcolor='#15202b',
                        font = dict(color = '#ffffff'),
                        height=len(cluster_counts_orig)*20+200
                        )

    cluster_counts_fig= go.Figure(data=data,
                                  layout=layout)
    cluster_counts_fig.update_yaxes(ticksuffix = " ")

    return cluster_counts_fig

def hourday_figure(df_orig,df):
    """
    Distribution plot of dataframe over the hours of the day
    Parameters: df_orig,df
    -df_orig: dataframe to compare with df
    Returns: plotly figure
    """
    #Para df_orig
    hourday_counts_orig=df_orig.groupby(df_orig['date'].dt.hour).size() #Distribucion de las filas a lo largo de las horas del dia
    hourday_counts_orig_dict=dict(zip(hourday_counts_orig.index.astype('int').tolist(),hourday_counts_orig.values.tolist()))#Lo paso a diccionario
    #Para df
    hourday_counts=df.groupby(df['date'].dt.hour).size() #Hora del día
    hourday_counts_dict=dict(zip(hourday_counts.index.astype('int').tolist(),hourday_counts.values.tolist()))#Lo paso a diccionario
    #Relleno con 0's las horas que no estén
    for i in range(24):
        if i not in hourday_counts_orig.keys():
            hourday_counts_orig_dict[i]=0
        if i not in hourday_counts.keys():
            hourday_counts_dict[i]=0
    #Figura
    #Añado el original
    trace0=go.Bar(y=list(hourday_counts_orig_dict.values()),
            x=list(hourday_counts_orig_dict.keys()),
            marker_color='#1d9bf0',name='original'
            ).update(opacity=0.25)
    #Añado el del df
    trace1=go.Bar(y=list(hourday_counts_dict.values()),
                x=list(hourday_counts_dict.keys()),
                marker_color='#1d9bf0',name='filtrado'
            )
    #Cambio layout
    layout=go.Layout(title='Tweets a lo largo del día',title_x=0.5,barmode='overlay',margin=go.layout.Margin(l=0,r=0, b=0,t=50),
                     xaxis_title="h",height=275,legend=dict(itemclick="toggleothers",itemdoubleclick="toggle",xanchor='auto',yanchor='middle',bgcolor='rgba(0,0,0,0)'),
           )
    data=[trace0,trace1]
    hourday_fig= go.Figure(data=data,layout=layout)

    return hourday_fig

def dayofweek_figure(df_orig,df):
    """
    Distribution plot of dataframe over the days of the week
    Parameters: df_orig,df
    -df_orig: dataframe to compare with df
    Returns: plotly figure
    """
    #dict para cambiar número devueltos por pandas por las letras de los días
    num_to_day={0:'L',1:'M',2:'X',3:'J',4:'V',5:'S',6:'D'}
    #Invierto orden para la figura
    dayofweek_counts_orig=df_orig.groupby(df_orig['date'].dt.dayofweek).size()[::-1]#Distribucion
    #Lo paso a diccionario
    dayofweek_counts_orig_dict=dict(zip(dayofweek_counts_orig.index.tolist(),dayofweek_counts_orig.values.tolist()))
    
    dayofweek_counts=df.groupby(df['date'].dt.dayofweek).size()[::-1] 
    dayofweek_counts_dict=dict(zip(dayofweek_counts.index.tolist(),dayofweek_counts.values.tolist()))#Lo paso a diccionario
    
    #Relleno con 0's las días que no estén
    for i in range(7):
        if i not in dayofweek_counts_orig.keys():
            dayofweek_counts_orig_dict[i]=0
        if i not in dayofweek_counts.keys():
            dayofweek_counts_dict[i]=0

    #Figuras
    y_orig=[num_to_day[i] for i in list(dayofweek_counts_orig_dict.keys())] #Cambio de números a letras
    trace0=go.Bar(y=y_orig,
            x=list(dayofweek_counts_orig_dict.values()),
            marker_color='#1d9bf0',name='original',orientation='h'
            ).update(opacity=0.25)

    y=[num_to_day[i] for i in list(dayofweek_counts_dict.keys())] #Cambio de números a letras
    trace1=go.Bar(y=y,
                x=list(dayofweek_counts_dict.values()),
                marker_color='#1d9bf0',name='filtrado',orientation='h'
            )

    layout=go.Layout(title='Tweets a lo largo de la semana',title_x=0.5,barmode='overlay',margin=go.layout.Margin(l=0,r=0, b=0,t=50)
                    ,height=190,legend=dict(itemclick="toggleothers",itemdoubleclick="toggle",xanchor='center',yanchor='middle'
                    ,bgcolor='rgba(0,0,0,0)')
           )
    data=[trace0,trace1]

    dayofweek_figure = go.Figure(data=data,layout=layout)
    dayofweek_figure.update_yaxes(ticksuffix = " ")

    return dayofweek_figure

#Función necesaria para expandir las de columnas con listas y hacer hacer value counts
def to_1D(series):
    return pd.Series([x for _list in series for x in _list])

def value_counts_figure(df,columna,top,title):
    """
    Histogram of categorical columns in dataframe
    Parameters: df,columna,top,title
    -top: number of categorias to show in histogram
    Returns: plotly figure
    """
    #Si la columna es una de estas, es una columna de listas y hay que expandirla
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
                    height=len(column_counts)*12+104 #Un mínimo para cuando hay una sola barra que no se corte la figura
                    )
    column_counts_fig= go.Figure(go.Bar(y=column_counts.index,x=list(column_counts),
                                        orientation='h',marker_color='#1d9bf0'
                                        ),
                                layout=layout
                                )
    return column_counts_fig

def temporal_fig(df_filtered,cluster_to_pos,clusters_filtro,colores):
    """
    Time series of specific dataframe's columns and filtering by Cluster's column
    Parameters: df_filtered,cluster_to_pos,clusters_filtro,colores
    -clusters_filtro: list with clusters to consider
    -colores: list of color codes
    -cluster_to_pos: dict with cluster number to index in colores list
    Returns: plotly figure
    """
    #Añado el cluster -1 para repsentar a todo el dataframe si la lista está vacía
    NumClusters=copy.deepcopy(clusters_filtro)
    if len(clusters_filtro)==0:
        NumClusters.append(-1)  

    df=df_filtered.copy(deep=True)
    df=df.set_index('date')
    #Columnnas para enseñar
    cols=['replyCount','retweetCount','likeCount','quoteCount','Photo','Video','Cluster']
    df=df[cols]
    df.columns=['Respuestas','Retweets','Likes','Citas','Fotos','Videos','Cluster']
    #Dependiendo los limites temporales del df, agrupo por años o días. Por semanas y por meses siempre lo hago
    if (df.index.max()-df.index.min())>datetime.timedelta(days=365):
        escalas=['Y','M','W']
    else:
        escalas=['M','W','D']

    fig=go.Figure()
    dfs=[]
    #Bucle para cada cluster y escala de tiempo. Tengo que dibujar todas. Habrá un butón para ocultar las escalas temporales
    #que no queramos. Habrá otro botón para elegir la columna y cambiar las curvas que hay.
    for cluster in NumClusters:
        if cluster==-1:
            color='#1d9bf0' 
        else:
            color=colores[cluster_to_pos[cluster]]
        for escala in escalas:
            if cluster==-1:
                df_tmp=df.resample(escala).mean()#Agrupo por la escala temporal y hago la media de cada filas
                df_tmp.insert(0,'Tweets',df.resample(escala).size()) #Añado el número de filas para cada escala
            else:
                df_tmp=df[df.Cluster==cluster].resample(escala).mean()
                df_tmp.insert(0,'Tweets',df[df.Cluster==cluster].resample(escala).size())
            df_tmp.drop(['Cluster'],axis='columns',inplace=True)
            #Paso fotos y videos a porcentaje
            df_tmp['Fotos']=100*df_tmp['Fotos']
            df_tmp['Videos']=100*df_tmp['Videos']
            df_tmp=round(df_tmp)
            dfs.append(df_tmp)
            #Las primeras curvas que enseño son las de la primera escala en la lista
            visibility=(escala==escalas[0])
            if cluster==-1:
                hoverinfo='x+y'  
            else:
                hoverinfo='x+y+name'  
            fig.add_trace(go.Scatter(x=df_tmp.index,
                             y=df_tmp[df_tmp.columns[0]],
                             visible=visibility,
                             line=dict(shape='spline'),
                             name='Cluster '+str(cluster),
                             marker=go.scatter.Marker(color=color),
                        hoverinfo=hoverinfo
                        )
            )
    #Botón para elegir que columna mostrar. Hay que añadir diccionarios por cada columna a la lista
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
    #Botón para invisibilizar las escalas de tiempo que no queremos 
    #Las trazas se ordenan de tal forma que si tenemos 2 clusters y 3 escalas temporales, la lista es
    #[1Y,1M,1D,2Y,2M,2D] con los números representando el cluster y las letras las escalas. Si queremos mostrar los meses,
    #la lista de visibility será [False,True,False,False,True,False]
    buttonsVisible = []
    escala_to_label=dict(Y='Año',M='Mes',W='Semana',D='Dia')
    for i,escala in enumerate(escalas):
        visibility=[False]*len(escalas)
        visibility[i]=True
        visibility=visibility*len(NumClusters)
        buttonsVisible.append(dict(label=escala_to_label[escala],method = 'restyle',args = ['visible',visibility]))

    #Añado los botones al updatemenu con diccionarios para especificar varios parámetros
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

def violin_distribution(df_filtered,column,clusters_filtro,cluster_to_pos,colores):
    """
    Violin distribution of columns. Designed to use it on likes and retweets
    Parameters: df_filtered,column,clusters_filtro,cluster_to_pos,colores
    -clusters_filtro: list with clusters to consider
    -colores: list of color codes
    -cluster_to_pos: dict with cluster number to index in colores list
    Returns: plotly figure
    """
    df=df_filtered.copy(deep=True)
    violin_fig=go.Figure()
    #Si no hay varios clusters a considerar, dibujo la distribución para todo el df
    if len(clusters_filtro)==0:
        color='#1d9bf0'
        violin_fig.add_trace(go.Violin(y=df[column]
                            ,name='Total',line_color=color,hoverinfo='x+y'
                            )
        )
    #Si no, para cada cluster
    else:
        for cluster in clusters_filtro:
            color=colores[cluster_to_pos[cluster]]
            violin_fig.add_trace(go.Violin(y=df[df.Cluster==cluster][column]
                                ,name='Cluster '+str(cluster),line_color=color,hoverinfo='x+y'
                                )
            )

    violin_fig.update_traces(orientation='v',side='both', width=1, points=False)
    #Cambio rango de 'y' para mejorar la figura con distribuciones con grandes outliers como pueden ser los retweets 
    max_y=df[column].quantile(0.995)
    min_y=df[column].min()-10

    column_to_name={'retweetCount':'retweets','likeCount':'likes'}
    title=dict(text='Distribución {}'.format(column_to_name[column]),x=0.5,y=1,yanchor='top')

    violin_fig.update_layout(title=title,height=250,margin=go.layout.Margin(l=0,r=10,b=0,t=20),
                    xaxis_showgrid=True, xaxis_zeroline=False,legend=dict(yanchor="top",y=1,xanchor="right",x=1)
                    ,yaxis_range=[min_y,max_y],showlegend=False
    )

    return violin_fig

def representative_tweets(df_tmp):
    """
    Dataframe sorted by cluster_prob and showed in a plotly figure 
    Returns: plotly figure
    """
    df=df_tmp.copy(deep=True)
    #Las tres columnas a considerar
    df=df[['renderedContent','Cluster','Cluster_prob']]
    df.sort_values(by=['Cluster_prob'],ascending=False,inplace=True)
    #Limito el número de filas a mostrar  
    longitud=df.Cluster.unique().size*20
    if longitud>50:
        longitud=50

    df=df.round({'Cluster_prob':2}).head(longitud)
    #Figura de la tabla
    fig = go.Figure(data=[go.Table(
    columnwidth = [0.76,0.1,0.14],
    header=dict(values=['Tweets','Cluster','Confianza'],
                fill_color='#192734',
                align='left'),
    cells=dict(values=[df.renderedContent,df.Cluster,df.Cluster_prob],
               fill_color='#15202b',line_color='#15202b',
               align='left',font=dict(color='#FFFFFF')))
    ])
    fig.update_layout(title='Tweets más representativos de los clusters',margin=go.layout.Margin(l=0,r=10, b=0,t=40)
                    ,height=375)
    return fig
  

def relevant_terms_table(df):
    """
    Value counts of words for a column of texts    
    Returns: plotly figure
    """
    #Limito el número de filas
    longitud=df.Cluster.unique().size*15
    if longitud>50:
        longitud=50
    #Uno la columna, separo por espacios y hago value_counts
    df_fig=pd.Series(' '.join(df.content_tokens).split()).value_counts().head(longitud)
    tokens_fuera=[x for x in ['fotos_y_videos'] if x in df_fig.index]
    df_fig.drop(tokens_fuera,axis=0,inplace=True)

    fig = go.Figure(data=[go.Table(
    columnwidth = [0.65,0.35],
    header=dict(values=['término','frecuencia'],
                fill_color='#192734',
                align='left'),
    cells=dict(values=[df_fig.index,df_fig.values],
               fill_color='#15202b',line_color='#15202b',
               align='left',font=dict(color='#FFFFFF')))
    ])
    fig.update_layout(title='Términos más usados',margin=go.layout.Margin(l=10,r=20, b=0,t=40)
                ,height=375
    )
    return fig


def df_tabla(df_tmp):
    """
    AgGrid figure of the dataframe    
    Returns: plotly figure
    """
    df=df_tmp.copy(deep=True)
    df=df.round({'Cluster_prob':2})
    #Filtro columnas y cambio nombres
    df=df[['url','renderedContent','date','retweetCount','replyCount','quoteCount','Cluster','Cluster_prob','Opinión']] 
    df.columns=['Url','Tweet','Fecha','Retweets','Respuestas','Citas','Cluster','Confianza','Opinión']
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    #Parametros que me gustaria personalizar y no funcionan
    gb.configure_default_column(copyHeadersToClipboard=True,filterable=False,editable=False,resizable=False, groupable=False,)

    gridOptions = gb.build()
    AgGrid(df,gridOptions=gridOptions,enable_enterprise_modules=True)
    return None