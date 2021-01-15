import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer #libreria para obtener las derivadas de las palabrasmas frecuentes
stemmer = PorterStemmer()

def tokenize(oraciones):
    """
    dividir la oración en una matriz de palabras / fichas
    un token puede ser una palabra o un carácter de puntuación, o un número
    """
    return nltk.word_tokenize(oraciones)


def stem(palabra): #derivar
    """
    derivar = encontrar la forma raíz de la palabra
    ejemplos:
    palabras = ["organizar", "organiza", "organizar"]
    palabras = [raíz (w) para w en palabras]
    -> ["órgano", "órgano", "órgano"]
    """
    return stemmer.stem(palabra.lower())


def bag_of_words(tokenized_sentence, palabras):
    """
    bolsa de devolución de matriz de palabras:
     1 por cada palabra conocida que existe en la oración, 0 en caso contrario
     ejemplo:
     frase = ["hola", "cómo", "estas", "tú"]
     palabras =                   ["hola", "hola", "yo", "tú", "adiós", "gracias", "genial"]
     palabras de entrenamiento =  [  0   ,    1  ,  0  ,  1  ,    0   ,    0     ,    0    ]
    """
    # derivar cada palabra
    sentence_words = [stem(word) for word in tokenized_sentence]
    # inicializar bolsa con 0 para cada palabra
    bag = np.zeros(len(palabras), dtype=np.float32)
    for idx, w in enumerate(palabras):
        if w in sentence_words: 
            bag[idx] = 1

    return bag