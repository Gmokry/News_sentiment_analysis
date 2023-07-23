'''CÓDIGO ANÁLISIS SENTIMIENTO PRENSA ESCRITA'''
'''LIBRERIAS MATRICES'''
import pandas as pd
import numpy as np
import nltk

'''CARGARMOS LOS 3 CSV DE DATOS PROVENIENTE DE KAGGLE Y LOS UNIFICAMOS EN UN ÚNICO DATAFRAME PARA PODER TRABAJAR CON ELLOS'''
datos1 = pd.read_csv(r"C:\Users\sarer\Desktop\Bootcamp\00_mis ejercicios\Entegables\EDA\Dataset_noticias\UNIFICADOS DE DISTINTOS MEDIOS\articles1.csv")
datos2 = pd.read_csv(r"C:\Users\sarer\Desktop\Bootcamp\00_mis ejercicios\Entegables\EDA\Dataset_noticias\UNIFICADOS DE DISTINTOS MEDIOS\articles2.csv")
datos3 = pd.read_csv(r"C:\Users\sarer\Desktop\Bootcamp\00_mis ejercicios\Entegables\EDA\Dataset_noticias\UNIFICADOS DE DISTINTOS MEDIOS\articles3.csv")

'''CREO UN DATAFRAME CON LOS DATOS DE LOS 3 CSV UNIFICADOS. LO GUARDO PARA TENER DIFERENTES VERSIONES E IR MODIFICANDO DATOS'''
df_allnews = pd.concat([datos1,datos2,datos3], ignore_index=True)

'''LIMPIO COLUMNAS QUE NO NECESITO PARA ESTE ANÁLISIS'''
df_allnews_cleaned = df_allnews.drop(df_allnews.columns[[0,1,4,5,8]], axis=1)

'''LIMPIEZA DE LOS DATOS. Existen filas que no tienen datos o que estos no son strings, por lo que no deben analizarse. Limpiamos estos datos.'''

df_allnews_cleaned = df_allnews_cleaned.dropna() #con dropna, eliminamos todas las filas que contengan NaN o non-string values

'''ANALIZADOR DE SENTIMIENTOS DE VADER. ANALIZAMOS TITULARES'''
from nltk.sentiment import SentimentIntensityAnalyzer

vader_sentiment_analyzer = SentimentIntensityAnalyzer()

def vader_sentiment (texto):
    diccionario_vader = vader_sentiment_analyzer.polarity_scores(texto)
    compound_value = diccionario_vader["compound"]
    return compound_value  

df_allnews_cleaned["vader_titulares"] = df_allnews_cleaned.title.apply(vader_sentiment)
df_allnews_cleaned_2 = df_allnews_cleaned.copy()


'''ANALIZADOR SENTIMIENTOS TEXTBLOB'''
from textblob import TextBlob

#es necesario convertir el texto en objetos TextBlob antes de poder usarlo. 
df_allnews_cleaned_2['textblob_titulares'] = df_allnews_cleaned_2.title.apply(lambda x: TextBlob(x).sentiment.polarity)
df_allnews_cleaned_3 = df_allnews_cleaned_2.copy()


'''ANALISIS UTILIZANDO AFINN. EN EL CASO DE AFINN, ES NECESARIO PREPROCESAR EL TEXTO ANTES'''

from afinn import Afinn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
import string #para eliminar punctuation

def remove_stopwords_and_punctuation(texto):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(texto)
    filtered_tokens = []
    for word in tokens:
        if word.lower() not in stop_words and word not in string.punctuation:
            filtered_tokens.append(word) 
    return filtered_tokens

'''Hago un apply para aplicarlo a todas las celdas de la columna de content'''

df_allnews_cleaned_3["Title_tokenization"] = df_allnews_cleaned_3.title.apply(remove_stopwords_and_punctuation) 

'''Una vez tokenizado el texto, se realiza el análisis Afinn sobre la lista generada. El análisis Afinn NO SE PUEDE REALIZAR SOBRE UNA LISTA, por lo que hay que 
volver a hacer un string con las palabras'''

afinn_analyzer = Afinn()

def Afinn_analysis (tokens):
    texto_procesado = " ".join(tokens)
    sentiment_score = afinn_analyzer.score(texto_procesado)
    return sentiment_score

df_allnews_cleaned_3["Afinn_titulares"] = df_allnews_cleaned_3.Title_tokenization.apply(Afinn_analysis)
df_allnews_cleaned_4 = df_allnews_cleaned_3.copy()



'''ANALISIS UTILIZANDO TRANSFORMERS PIPELINE Y EL MODELO DISTILBERT ("model="distilbert-base-uncased-finetuned-sst-2-english")'''

from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
def analisis_pipeline (texto):
    valor_sentimiento = sentiment_pipeline(texto)
    return valor_sentimiento

df_allnews_cleaned_4["Pipeline_titulares"] = df_allnews_cleaned_4.title.apply(analisis_pipeline)

df_allnews_cleaned_5 = df_allnews_cleaned_4.copy()

df_allnews_cleaned_5["Pipeline_titulares_label"] = df_allnews_cleaned_4.Pipeline_titulares.apply(lambda x: x[0]["label"])
df_allnews_cleaned_5.drop("Title_tokenization", axis=1, inplace=True)
df_allnews_cleaned_6 = df_allnews_cleaned_5.copy()


'''Habría que ajustar la escala de Afinn para que pudiese ser comparable a la escala de Vader o Textblob. Por tanto, realizamos escalamiento lineal:
'''
maximo_inicial = df_allnews_cleaned_5.Afinn_titulares.max()
minimo_inicial = df_allnews_cleaned_5.Afinn_titulares.min()
rango = maximo_inicial - minimo_inicial 
amplitud = 2

df_allnews_cleaned_6['Afinn_titulares_escala']= df_allnews_cleaned_6.Afinn_titulares.apply(lambda x: (((x-minimo_inicial)/rango)*amplitud)-1)
df_allnews_cleaned_7 = df_allnews_cleaned_6.copy()


'''Eliminamos todas las columnas que ya no nos hacen falta para el análisis'''

df_allnews_cleaned_7.drop("content",axis=1, inplace=True)
df_allnews_cleaned_7.drop("Afinn_titulares",axis=1, inplace=True)
df_allnews_cleaned_7.drop("Pipeline_titulares",axis=1, inplace=True)

df_allnews_cleaned_8 = df_allnews_cleaned_7
df_allnews_cleaned_7.to_csv(r"C:\Users\sarer\Desktop\Bootcamp\00_mis ejercicios\Entegables\EDA\Dataset_noticias\0_csv_procesados\nlp_scores_7")

'''DATA VISUALIZATION'''

import pandas as pd
import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

datos = pd.read_csv((r"C:\Users\sarer\Desktop\Bootcamp\00_mis ejercicios\Entegables\EDA\Dataset_noticias\0_csv_procesados\nlp_scores_7"))

'''KDE SUPERPUESTOS'''
Analisis_labels = ["Vader", "TextBlob","Afinn"]
columns_analyzed = datos.loc[:,["vader_titulares","textblob_titulares","Afinn_titulares_escala"]]
fig, ax = plt.subplots()

for i in range(3):
    plot = sns.kdeplot(data= datos, x=columns_analyzed[columns_analyzed.columns[i]], 
                       fill= True, color=sns.color_palette("Set1")[i], ax=ax, label=Analisis_labels[i])

plot.set_xlabel("Sentiment score")
plot.set_title("Analysis types vs sentiment score distribution") 
plt.legend(loc="center left")
plt.savefig("KDE_plots.jpg", format='jpg')
plt.show()

'''ANÁLISIS POR PUBLICACIONES AGRUPADAS'''

analisis_por_publicacion = datos.groupby("publication")[["vader_titulares","textblob_titulares","Afinn_titulares_escala"]].mean()
analisis_por_publicacion.head()

'''SUBPLOTS - PLOTEAR EN BARPLOTS, POR PUBLICACIÓN, LA MEDIA DE POSITIVISMO POR TIPO DE ANÁLISIS'''

sns.set_theme(style="whitegrid")
sns.set_palette("mako")
Analisis_labels = ["Vader analysis", "TextBlob analysis", "Afinn analysis"]
fig, axes = plt.subplots(3,1, sharex= False, sharey=True, figsize =(20,20))
axes = axes.flatten()
fig.suptitle ("Análisis por publicación y tipo de análisis", fontsize= 20)
for i in range (3):
    ax = sns.barplot(x= analisis_por_publicacion.index, y= analisis_por_publicacion.iloc[:,i], ax=axes[i])
    ax.set(ylabel = Analisis_labels[i])
    
plt.tight_layout()   
plt.savefig('Analisis por publicacion y tipo de analisis.jpg', format='jpg')
plt.show()

'''LA FIGURA SE VE MUY BIEN PERO DIFICIL PARA POWERPOINT'''

'''VADER POR PUBLICACION'''
sns.set_theme(style="whitegrid")
sns.set_palette("mako")

plt.figure(figsize=(10,5))
barplot_vader = sns.barplot (x=analisis_por_publicacion.index, y=analisis_por_publicacion.vader_titulares)
barplot_vader.set_ylabel("Score")
barplot_vader.set_title("Vader Analysis")
barplot_vader.set_xlabel(None)
barplot_vader.tick_params(axis="x", labelsize=15, rotation=90)
barplot_vader.set_ylim(-0.15,0.15)
plt.tight_layout()

plt.savefig('Vader_por_publicacion.jpg', format='jpg')

'''TEXTBLOB POR PUBLICACION'''
sns.set_theme(style="whitegrid")
sns.set_palette("mako")

plt.figure(figsize=(10,5))
barplot_vader = sns.barplot (x=analisis_por_publicacion.index, y=analisis_por_publicacion.textblob_titulares)
barplot_vader.set_ylabel("Score")
barplot_vader.set_title("TextBlob Analysis")
barplot_vader.set_xlabel(None)
barplot_vader.tick_params(axis="x", labelsize=15, rotation=90)
barplot_vader.set_ylim(-0.15,0.15)
plt.tight_layout()

plt.savefig('TextBlob_por_publicacion.jpg', format='jpg')

'''AFINN POR PUBLICACION'''
sns.set_theme(style="whitegrid")
sns.set_palette("mako")

plt.figure(figsize=(10,5))
barplot_vader = sns.barplot (x=analisis_por_publicacion.index, y=analisis_por_publicacion.Afinn_titulares_escala)
barplot_vader.set_title("Afinn Analysis")
barplot_vader.set_xlabel(None)
barplot_vader.set_ylabel("Score")
barplot_vader.tick_params(axis="x", labelsize=15, rotation=90)
barplot_vader.set_ylim(-0.05,0.05)
plt.tight_layout()

plt.savefig('Afinn_por_publicacion.jpg', format='jpg')


'''BOXPLOT VADER POR SCORE Y PUBLICACION'''
sns.set_theme(style="whitegrid")
sns.set_palette("mako")

plt.figure(figsize=(10,5))
barplot_vader = sns.boxplot (x=datos.publication, y=datos.vader_titulares)
barplot_vader.set_ylabel("Score")
barplot_vader.set_title("Vader Analysis")
barplot_vader.set_xlabel (None)
barplot_vader.tick_params(axis="x", labelsize=15, rotation=90)
plt.tight_layout()

plt.savefig('Vader_boxplot_por_publicacion.jpg', format='jpg')

'''BOXPLOT TEXTBLOB POR SCORE Y PUBLICACION'''
sns.set_theme(style="whitegrid")
sns.set_palette("mako")

plt.figure(figsize=(10,5))
barplot_vader = sns.boxplot (x=datos.publication, y=datos.textblob_titulares)
barplot_vader.set_title("TextBlob Analysis")
barplot_vader.set_xlabel(None)
barplot_vader.set_ylabel("Score")
barplot_vader.tick_params(axis="x", labelsize=15, rotation=90)
barplot_vader.set_ylim(-1,1)
plt.tight_layout()

plt.savefig('TextBlob_boxplot_por_publicacion.jpg', format='jpg')


'''BOXPLOT AFINN POR SCORE Y PUBLICACION'''
sns.set_theme(style="whitegrid")
sns.set_palette("mako")

plt.figure(figsize=(10,5))
barplot_vader = sns.boxplot (x=datos.publication, y=datos.Afinn_titulares_escala)
barplot_vader.set_title("Afinn Analysis")
barplot_vader.set_xlabel(None)
barplot_vader.set_ylabel("Score")
barplot_vader.tick_params(axis="x", labelsize=15, rotation=90)
plt.tight_layout()

plt.savefig('Afinn_boxplot_por_publicacion.jpg', format='jpg')

'''CONVERTIR DATOS NUMÉRICOS A DATOS CUALITATIVOS PARA COMPARAR CON TRASNFORMERS'''
def evaluar_label (valor):
    if valor >0:
        return "POSITIVE"
    elif valor <0:
        return "NEGATIVE"
    else:
        return "NEUTRAL"
    
datos["Vader_cualitative"] = datos.vader_titulares.apply(evaluar_label)
datos["Textblob_cualitative"] = datos.textblob_titulares.apply(evaluar_label)
datos["Afinn_cualitative"] = datos.Afinn_titulares_escala.apply(evaluar_label)

'''CREAMOS DATAFRAME AGRUPADO POR PUBLICACIONES Y NUMERO DE POSITIVE,NEGATIVE Y NEUTRAL'''
Pipeline_cualitativo_por_publication = datos.groupby("publication")["Pipeline_titulares_label"].value_counts(normalize=True)
Vader_cualitativo_por_publication = datos.groupby("publication")["Vader_cualitative"].value_counts(normalize=True)
TextBlob_cualitativo_por_publication = datos.groupby("publication")["Textblob_cualitative"].value_counts(normalize=True)
Afinn_cualitativo_por_publication = datos.groupby("publication")["Afinn_cualitative"].value_counts(normalize=True)

dataframe_cualitativo = pd.DataFrame ({"Pipeline_cual":Pipeline_cualitativo_por_publication,
                                      "Vader_cual":Vader_cualitativo_por_publication,
                                     "TextBlob_cual": TextBlob_cualitativo_por_publication,
                                     "Afinn_cual": Afinn_cualitativo_por_publication})

dataframe_cualitativo.fillna(0,inplace=True)
df= dataframe_cualitativo.unstack()

'''GROUPED BARPLOT CUALITATIVO POR PUBLICACIÓN'''

sns.set_theme(style="whitegrid")
sns.set_palette("mako")

grouped_barplot = df.Pipeline_cual.plot(kind="bar", stacked=True,figsize = (10,6))
grouped_barplot.set_title("Pipeline Analysis")
grouped_barplot.set_ylabel("Relative frequency")
grouped_barplot.legend(loc="lower right")
plt.tight_layout()

plt.savefig('Pipeline_cualitativo_publicacion.jpg', format='jpg')


sns.set_theme(style="whitegrid")
sns.set_palette("mako")

grouped_barplot = df.Vader_cual.plot(kind="bar", stacked=True,figsize = (10,6))
grouped_barplot.set_title("Vader Analysis")
grouped_barplot.set_ylabel("Relative frequency")
grouped_barplot.legend(loc="lower right", fontsize = (10))
plt.tight_layout()

plt.savefig('Vader_cualitativo_publicacion.jpg', format='jpg')


sns.set_theme(style="whitegrid")
sns.set_palette("mako")


grouped_barplot = df.TextBlob_cual.plot(kind="bar", stacked=True, figsize = (10,6))
grouped_barplot.set_title("TextBlob Analysis")
grouped_barplot.set_ylabel("Relative frequency")
grouped_barplot.legend(loc="center right", fontsize = (9))
plt.tight_layout()

plt.savefig('TextBlob_cualitativo_publicacion.jpg', format='jpg')

sns.set_theme(style="whitegrid")
sns.set_palette("mako")


grouped_barplot = df.Afinn_cual.plot(kind="bar", stacked=True, figsize = (10,6))
grouped_barplot.set_title("Afinn Analysis")
grouped_barplot.set_ylabel("Relative frequency")
grouped_barplot.legend(loc="upper right", fontsize = (10))
plt.tight_layout()

plt.savefig('Afinn_cualitativo_publicacion.jpg', format='jpg')

'''AGRUPADO POR TIPOS DE ANÁLISIS Y LA MEDIA DE POSITIVO,NEGATIVO Y NEUTRO DE TODOS LOS TEST'''

media_TipoAnalisis_porPublicacion = dataframe_cualitativo.T.describe().iloc[1]
media_TipoAnalisis_porPublicacion= media_TipoAnalisis_porPublicacion.unstack()


sns.set_theme(style="whitegrid")
sns.set_palette("mako")


grouped_barplot = media_TipoAnalisis_porPublicacion.plot(kind="bar", stacked=True, figsize = (10,6))
grouped_barplot.set_title("All Analysis")
grouped_barplot.set_ylabel("Relative frequency")
grouped_barplot.legend(loc="upper right", fontsize = (10))
plt.tight_layout()

plt.savefig('Mean_allAnalysis.jpg', format='jpg')


sns.set_theme(style="whitegrid")
sns.set_palette("muted")


grouped_barplot = media_TipoAnalisis_porPublicacion.plot(kind="bar", figsize = (10,5))
grouped_barplot.set_title("All Analysis")
grouped_barplot.set_ylabel("Relative frequency")
grouped_barplot.set_xlabel(None)
grouped_barplot.legend(loc="upper right", fontsize = (10))
plt.tight_layout()

plt.savefig('Mean_allAnalysis.jpg', format='jpg')

'''GRÁFICA FINAL TIPO LOLLIPOP PARA TOTAL DE POSITIVOS, NEGATIVO Y NEUTROS (TODOS ALGORITMOS Y PUBLICACIONES UNIFICADAS)'''

columns_to_analyze = ["Pipeline_titulares_label","Vader_cualitative","Textblob_cualitative","Afinn_cualitative"]
total_cualitative_analysis = pd.DataFrame(datos[columns_to_analyze].unstack().value_counts(normalize=True).sort_values()*100)


x= total_cualitative_analysis.index
y = total_cualitative_analysis["proportion"]
plt.figure(figsize=(3,3))
plt.stem(x,y,use_line_collection = True)
plt.ylabel(r"% of total scores")
plt.title("Total scores. All methods and publications.")
plt.show()


'''CORRELACIÓN ENTRE TIPOS DE ANÁLISIS'''
from scipy.stats import pearsonr


sns.set_theme(style="white")
g = sns.JointGrid(data=datos, x="vader_titulares", y="textblob_titulares", space=0)
g.plot_joint(sns.kdeplot, fill=True, cmap="rocket")
g.plot_marginals(sns.histplot, kde=True)
g.set_axis_labels("Vader Analysis", "TextBlob Analysis")


# Calcular el coeficiente de correlación de Pearson. Te devuelve una tupla con el coeficiente de correlación y el p-valor
correlation, p_value = pearsonr(datos["vader_titulares"], datos["textblob_titulares"])
plt.show()
print(correlation)


sns.set_theme(style="white")
g = sns.JointGrid(data=datos, x="vader_titulares", y="Afinn_titulares_escala", space=0)
g.plot_joint(sns.kdeplot, fill=True, cmap="rocket")
g.plot_marginals(sns.histplot, kde=True)
g.set_axis_labels("Vader Analysis", "Afinn Analysis")


correlation, p_value = pearsonr(datos["vader_titulares"], datos["Afinn_titulares_escala"])
plt.show()
print(correlation)

sns.set_theme(style="white")
g = sns.JointGrid(data=datos, x="textblob_titulares", y="Afinn_titulares_escala", space=0)
g.plot_joint(sns.kdeplot, fill=True, cmap="rocket")
g.plot_marginals(sns.histplot, kde=True)
g.set_axis_labels("TextBlob Analysis", "Afinn Analysis")


correlation, p_value = pearsonr(datos["textblob_titulares"], datos["Afinn_titulares_escala"])
plt.show()
print(correlation)