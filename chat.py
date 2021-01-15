import random
import json

import torch #libreria para el aprendizaje automatico, utiliza los tensorflow, ejecuta el codigo de forma nativa usando la GPU

from modelo import NeuralNet
from nltk_utils import bag_of_words, tokenize

#declaramos el objeto torch
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#cargamos nuestra bd
with open('basededatos.json', 'r') as json_data:
    INTENCIONES = json.load(json_data)

#cargamos losdatos de entrenamiento
FILE = "datos.pth"
datos = torch.load(FILE)

#obtenemos los paraetros
_tamanio_entrada = datos["tamanio_de_entrada"]
_tamanio_oculto = datos["tamanio_oculto"]
_tamanio_salida = datos["tamanio_de_salida"]
_todas_las_palabras = datos['todas_las_palabras']
_etiqueta = datos['etiquetas']
_estado_modelo = datos["estado_del_modelo"]

#crea el modelo de reconocimiento
MODELO = NeuralNet(_tamanio_entrada, _tamanio_oculto, _tamanio_salida).to(dispositivo)
MODELO.load_state_dict(_estado_modelo)
MODELO.eval() #evalua las respuestas


bot_name = "PsicoRobo"
print(f"{bot_name}: Mi nombre es PsicoRobo. Responderé a tus consultas, si desea salir, escriba adios")
while True:
    sentencias = input("Tu: ")
    if sentencias == "adios":
        print(f"{bot_name}:Que tengas un buen día")
        break

    sentencias = tokenize(sentencias) #tokenizamos las palabras
    X = bag_of_words(sentencias, _todas_las_palabras) #coincidencia de palabras con la bd json
    X = X.reshape(1, X.shape[0]) 
    X = torch.from_numpy(X).to(dispositivo) #Objeto de entrenamiento y posibles respuestas

    salida = MODELO(X) 
    _, prediccion = torch.max(salida, dim=1)

    ETIQUETA = _etiqueta[prediccion.item()] #etiquetas y predicciones

    problemas = torch.softmax(salida, dim=1) #entrada de n dimensiones y rescalado de salidas de n dimensiones
    probabilidad = problemas[0][prediccion.item()] #prediccion de la respuesta mas acertada
    if probabilidad.item() > 0.75: #probabilidades mayores a 75%
        for intenc in INTENCIONES['intenciones']: #posibles intenciones
            if ETIQUETA == intenc["etiqueta"]:
                print(f"{bot_name}: {random.choice(intenc['respuestas'])}")
    else:
        print(f"{bot_name}: No entiendo...")