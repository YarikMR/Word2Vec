
# Crear un modelo de Word2Vec

Este ejemplo esta disponible con extencion 
- __.py__ para ser cargado desde cualquier IDE (Pychar, Spyder, etc) 
- __.ipynb__ para notebook

Requerimientos:
- NLTK: Installar ejecutar el comando __pip install nltk__
- Gensim: Instalar con __pip install gensim__


```python
import nltk
import gensim
```


```python
#Descomenta las dos siguientes lineas si no tienes instalado nltk y/o gensim
#! pip install nltk 
#! pip install gensim
```

# Leer Corpus


```python
with open ('data/reglamento_transito.txt',
           'r',encoding='utf-8') as file:
        document = file.read()
```


```python
document[:1000]
```




    'REGLAMENTO DE TRÁNSITO DEL MUNICIPIO SOLIDARIDAD, QUINTANA ROO \n\n(Reforma publicada en el Periódico Oficial del Gobierno del Estado el 30 de agosto de 2006) \n\n\n(Periódico Oficial del Estado del 16 de enerode 2012: Se Reforman los artículos 112; 185, Fracciones III y VIII, \n\ny se le Adiciona un segundo párrafo; Se adiciona un Inciso e) al Artículo 202) \n\n(Reforma publicada en el Periódico Oficial del Estado el 31 de marzode 2015) \n\nReforma: Se adiciona una fracción LXIX al artículo 2 y se modifican las fracciones I \na VIII del artículo 185; Publicado en el Periódico Oficial del Estado de Quintana Roo el 17 \nde febrero del año 201 7. \n\nÚltim a reforma: al párrafo primero y se adiciona un párrafo tercero del artículo 53 ; \n\nPublicado en el Periódico Oficial del Estado de Quintana Roo el 7 de abril del año 201 7. \n\n\nCAPÍTULO I \n\nDISPOSICIONES GENERALES \n\nArtículo 1.- El presente Reglamento es de orden público y de interés social y tiene por objeto \nestablecer las normas del transporte de p'




```python
# tokenizar el documento en oraciones
sentences = nltk.sent_tokenize(document)
sentences[1]
```




    'Últim a reforma: al párrafo primero y se adiciona un párrafo tercero del artículo 53 ; \n\nPublicado en el Periódico Oficial del Estado de Quintana Roo el 7 de abril del año 201 7.'




```python
#tokenizar cada oracion en palbras
word_tokens = [nltk.tokenize.word_tokenize(sentence.lower()) for sentence in sentences]
```


```python
#word_tokens[1]
```

# Modelo de W2V


```python
modelo_w2v = gensim.models.Word2Vec(sentences = word_tokens,
                                    size = 50,
                                    iter=100,
                                    min_count=1)
```


```python
modelo_w2v.most_similar('auto')
```

    /anaconda3/envs/curso36/lib/python3.6/site-packages/ipykernel_launcher.py:1: DeprecationWarning: Call to deprecated `most_similar` (Method will be removed in 4.0.0, use self.wv.most_similar() instead).
      """Entry point for launching an IPython kernel.





    [('automóvil', 0.9729188084602356),
     ('coche.-', 0.9714019298553467),
     ('ómnibus', 0.8502559065818787),
     ('motocicleta.-', 0.8487198948860168),
     ('tracto-camión.-', 0.8475431203842163),
     ('aliento', 0.8473482131958008),
     ('l.-', 0.8402494788169861),
     ('camioneta.-', 0.8338930010795593),
     ('xxxv.-', 0.8240362405776978),
     ('fabricado', 0.8169386386871338)]




```python
modelo_w2v['auto']
```

    /anaconda3/envs/curso36/lib/python3.6/site-packages/ipykernel_launcher.py:1: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      """Entry point for launching an IPython kernel.





    50




```python

```
