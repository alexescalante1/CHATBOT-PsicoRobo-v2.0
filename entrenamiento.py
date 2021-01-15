import nltk #herramientas de lenguaje natural
import numpy as np #Numerical Python (Estructuras de datos y matrices)
import random 
import json 

import torch #libreria para el aprendizaje automatico, utiliza los tensorflow, ejecuta el codigo de forma nativa usando la GPU
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from modelo import NeuralNet

nltk.download('punkt')
nltk.download('wordnet')

#cargamos nuestra bd
with open('basededatos.json', 'r') as f:
    intenciones = json.load(f)

todas_las_palabras = []
etiquetas = []
xy = []
# recorrer cada oración en nuestros patrones de intenciones
for inten in intenciones['intenciones']:
    etiqueta = inten['etiqueta']
    # agregar a la lista de etiquetas
    etiquetas.append(etiqueta)
    for patron in inten['patrones']:
        # tokenizar cada palabra de la oración
        w = tokenize(patron)
        # agregar a nuestra lista de palabras
        todas_las_palabras.extend(w)
        # agregar al par xy
        xy.append((w, etiqueta))

# ignorar palabra
ignorar_palabras = ['?', '.', '!']
todas_las_palabras = [stem(w) for w in todas_las_palabras if w not in ignorar_palabras]
# eliminar duplicados y ordenar
todas_las_palabras = sorted(set(todas_las_palabras))
etiquetas = sorted(set(etiquetas))

print(len(xy), "patrones")
print(len(etiquetas), "etiquetas:", etiquetas)
print(len(todas_las_palabras), "palabras derivadas unicas:", todas_las_palabras)

# crear datos de entrenamiento
X_cola = []
Y_cola = []
for (patron_de_oracion, etiqueta) in xy:
    # X: bolsa de palabras para cada patron de sentencia
    bolsa = bag_of_words(patron_de_oracion, todas_las_palabras)
    X_cola.append(bolsa)
    # y: PyTorch CrossEntropyLoss solo necesita etiquetas de clase, no one-hot
    ETIQUETA_ = etiquetas.index(etiqueta)
    Y_cola.append(ETIQUETA_)

X_cola = np.array(X_cola)
Y_cola = np.array(Y_cola)

# Hiperparámetros
interaciones = 1000 #NUMERO DE INTERACIONES
batch_size = 8      #tamanio de lote
tasa_de_aprendizaje = 0.001
tamanio_de_entrada = len(X_cola[0])
tamanio_oculto = 8
tamanio_de_salida = len(etiquetas)
print(tamanio_de_entrada, tamanio_de_salida)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_cola)
        self.x_data = X_cola
        self.y_data = Y_cola

    # Admite la indexación de modo que el conjunto de datos [i] se pueda utilizar para obtener la i-ésima muestra
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # podemos llamar a len (conjunto de datos) para devolver el tamaño
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
cargador_de_cola = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modelo = NeuralNet(tamanio_de_entrada, tamanio_oculto, tamanio_de_salida).to(dispositivo)

# Pérdida y optimizacion
criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_de_aprendizaje)

# Entrena el modelo
for nivel in range(interaciones):
    for (PALABRAS, ETIQUETAS) in cargador_de_cola:
        PALABRAS = PALABRAS.to(dispositivo)
        ETIQUETAS = ETIQUETAS.to(dtype=torch.long).to(dispositivo)
        
        # Pase adelantado
        SALIDAS = modelo(PALABRAS)
        # si y sería one-hot, debemos aplicar
        # etiquetas = antorcha.max (etiquetas, 1) [1]
        PERDIDAS = criterio(SALIDAS, ETIQUETAS)
        
        # Retroceder y optimizar
        optimizador.zero_grad()
        PERDIDAS.backward()
        optimizador.step()
        
    if (nivel+1) % 100 == 0:
        print (f'interaciones [{nivel+1}/{interaciones}], perdidos: {PERDIDAS.item():.4f}')


print(f'perdida final: {PERDIDAS.item():.4f}')

datos = {
"estado_del_modelo": modelo.state_dict(),
"tamanio_de_entrada": tamanio_de_entrada,
"tamanio_oculto": tamanio_oculto,
"tamanio_de_salida": tamanio_de_salida,
"todas_las_palabras": todas_las_palabras,
"etiquetas": etiquetas
}

FILE = "datos.pth"
torch.save(datos, FILE)

print(f'Entrenamiento completado, archivo guardado en: {FILE}')