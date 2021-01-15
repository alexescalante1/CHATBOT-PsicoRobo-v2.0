import torch #libreria para el aprendizaje automatico
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, tamanio_de_entrada, tamanio_oculto, numero_de_clases):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(tamanio_de_entrada, tamanio_oculto) 
        self.l2 = nn.Linear(tamanio_oculto, tamanio_oculto) 
        self.l3 = nn.Linear(tamanio_oculto, numero_de_clases)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # sin activación y sin softmax al final
        return out