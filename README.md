
# Crear un modelo de Word2Vec

Este ejemplo esta disponible con extencion 
- __.py__ para ser cargado desde cualquier IDE (Pychar, Spyder, etc) 
- __.ipynb__ para notebook

La carpeta Data tiene un texto en español para que el codigo pueda ser probado.

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

```python
modelo_w2v['auto']
```
