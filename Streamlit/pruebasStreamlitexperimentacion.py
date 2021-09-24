import streamlit as st
import pandas as pd
pd.set_option('max_colwidth',250)
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import json
import copy
from streamlit_plotly_events import plotly_events
import time
import funciones as F
from collections import OrderedDict
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
plt.style.use('seaborn')
# primary color theme: #7295E8
st.set_page_config(page_title="TFM_Marc",layout="wide",initial_sidebar_state="expanded")

#Carpetas a las que tengo que entrar: preprocessed_data,topic_data y embeddings

#Functions
#colores=sns.color_palette("hls", 14)
#st.pyplot(sns.palplot(colores))

inicio=time.time()    
df_orig,topic_terms,vocab,term_frequency=F.get_data()
#st.write(time.time()-inicio,'get_data')
df=df_orig.copy(deep=True)

with st.sidebar:
    num_tweets=st.empty()
with st.sidebar:
    st.subheader('Filtro de fechas')
    
    minfechas=df['date'].min().strftime("%Y-%m-%d")
    maxfechas=df['date'].max().strftime("%Y-%m-%d")
    
    start_date=st.text_input('Desde (mínimo {})'.format(minfechas), minfechas,autocomplete=minfechas)
    end_date=st.text_input('Hasta (máximo {})'.format(maxfechas), maxfechas,autocomplete=maxfechas)
    if start_date.strip()=='':
        start_date=minfechas
    if end_date.strip()=='':
        end_date=maxfechas
    
with st.sidebar:
    st.subheader('Filtro de clusters')
    Clusters = np.sort(df_orig['Cluster'].unique())
    clusters_filtro = st.multiselect('Elige clusters', Clusters)
    

with st.sidebar:   
    st.subheader('Filtro de texto')
    text_filtro = st.text_input("Tweets con el texto...", '').strip()    

with st.sidebar:   
    st.subheader('Filtro de enlaces externos')
    outlinks_filtro = st.text_input("Si se quiere elegir varios enlaces, escribir entre comas (p. ej.: youtube.com, youtu.be)", '').replace(' ','').split(',')
    outlinks_filtro= [i for i in outlinks_filtro if i] #Para quitar el string vacío si no se pone nada en la caja de texto   

with st.sidebar:   
    st.subheader('Filtro de idioma')
    lang_filtro = st.text_input("Si se quiere elegir varios idiomas, escribir entre comas (p. ej.: es,en)", '').replace(' ','').split(',')
    lang_filtro= [i for i in lang_filtro if i] #Para quitar el string vacío si no se pone nada en la caja de texto

with st.sidebar:   
    st.subheader('Filtro de dispositivo')
    sources=pd.unique(df_orig['sourceLabel'])
    source_filtro = st.multiselect('Elige dispositivos', sources)

st.title('Tweets de IbaiLlanos')
#Filtro df 
inicio2=time.time()
df=F.df_filtered(df,start_date,end_date,text_filtro,clusters_filtro,lang_filtro,outlinks_filtro,source_filtro)
#st.write(time.time()-inicio2,'filter_data')

num_tweets.write('Tweets: {} ({:.1f}%)'.format(len(df),100*len(df)/len(df_orig)))
#Colores elegidos en https://plotly.com/python/discrete-color/ 
colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
        '#1F77B4','#2CA02C','#8C564B','#BCBD22']

c11, c12 = st.columns((3,1))


inicio24=time.time()
cluster_counts_fig_func=F.cluster_figure(df_orig,df,topic_terms)
#st.write(time.time()-inicio24,'Función figura cluster')

#'staticPlot': True,
configc11 = {'modeBarButtonsToRemove': ['zoom','lasso','zoomIn','zoomOut','AutoScale']}
c11.plotly_chart(cluster_counts_fig_func,use_container_width=True, config=configc11)

configc12 = {'displayModeBar': False}



colores=px.colors.qualitative.Dark24
cluster_counts_orig=df_orig['Cluster'].value_counts(ascending=True)
clusters_indices=np.array(cluster_counts_orig.index).astype(str)

cluster_to_pos=OrderedDict()

for pos,cluster in enumerate(cluster_counts_orig.index):
    cluster_to_pos[cluster]=pos

time_series_figure=F.temporal_fig(df,cluster_to_pos,clusters_filtro,colores)
c11.plotly_chart(time_series_figure,use_container_width=True, config=configc11)

hourday_fig=F.hourday_figure(df_orig,df)
c12.plotly_chart(hourday_fig,use_container_width=True, **{'config': configc12})

hashtags_counts_fig=F.value_counts_figure(df,'hashtags',top=5,title='Hashtags')
c12.plotly_chart(hashtags_counts_fig,use_container_width=True, **{'config': configc12})

mentionedUsers_counts_fig=F.value_counts_figure(df,'mentionedUsers',top=5,title='Menciones')
c12.plotly_chart(mentionedUsers_counts_fig,use_container_width=True, config=configc12)

Sentimiento_counts_fig=F.value_counts_figure(df,'Sentimiento',top=5,title='Sentimiento')
c12.plotly_chart(Sentimiento_counts_fig,use_container_width=True, config=configc12)

c21,c22,c23=st.columns((2,1,1))

#Meter tabla
representative_tweets=F.representative_tweets(df)
inicio22=time.time()
c21.plotly_chart(representative_tweets,use_container_width=True, **{'config': configc12})
#st.write(time.time()-inicio22,'representative_tweets')
inicio21=time.time()
c22.plotly_chart(F.relevant_terms_table(df),use_container_width=True,config={'displayModeBar': False})
#st.write(time.time()-inicio21,'term mas relev df plotly')

outlinks_counts_fig=F.value_counts_figure(df,'outlinks',top=5,title='Enlaces externos')
c23.plotly_chart(outlinks_counts_fig,use_container_width=True, **{'config': configc12})

lang_counts_fig=F.value_counts_figure(df,'lang',top=5,title='Idioma')
c23.plotly_chart(lang_counts_fig,use_container_width=True, config=configc12)


source_counts_fig=F.value_counts_figure(df,'sourceLabel',top=5,title='Dispositivo')
c23.plotly_chart(source_counts_fig,use_container_width=True, **{'config': configc12})

#inicio32=time.time()
#st.dataframe(F.df_table(df))
#st.write(time.time()-inicio32,'tabla df')
#st.write()


#hacer filtro hora del dia
#Mostrar los tweets mas representativos para el cluster legidos
#Mostrar los terminos mas relevantes
#--> Mostrar distr. de horas y año/mes/dia en la columna derecha

#Histograms de retweets,favorites,replies...

inicio47=time.time()
F.df_tabla(df)
#st.write(time.time()-inicio47,'tabla df')






