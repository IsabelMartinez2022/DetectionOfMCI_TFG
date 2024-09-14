import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected
from model import HeteroGNN, pretrain_model

# Verificación de GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mapeo de valores categóricos
sex_map = {'M': 0, 'F': 1}
diagnosis_map = {'CN': 0, 'MCI': 1}

# Cargar CSVs
has_region = pd.read_csv('/mnt/data/has_region.csv')
is_connected_to_CN = pd.read_csv('/mnt/data/is_connected_to_CN.csv')
is_connected_to_MCI = pd.read_csv('/mnt/data/is_connected_to_MCI.csv')
is_functionally_connected = pd.read_csv('/mnt/data/is_functionally_connected.csv')
regions = pd.read_csv('/mnt/data/regions.csv')
subjects = pd.read_csv('/mnt/data/subjects.csv')

# Preprocesar los datos de los sujetos
subjects['sex'] = subjects['sex'].map(sex_map)
subjects['diagnosis'] = subjects['diagnosis'].map(diagnosis_map)

# Crear el grafo heterogéneo
data = HeteroData()

# Agregar las características de nodos para 'region'
data['region'].x = torch.eye(len(regions))

# Agregar características de nodos para 'subject' (sexo, edad, diagnóstico)
data['subject'].x = torch.tensor(subjects[['sex', 'age', 'diagnosis']].values, dtype=torch.float)

# Agregar las relaciones entre sujetos y regiones
data['subject', 'has_region', 'region'].edge_index = torch.tensor(
    [has_region['subject_id'].astype('category').cat.codes.values, has_region['region_id'] - 1], dtype=torch.long)

# Agregar las conexiones entre regiones (is_functionally_connected)
edges_region = np.array([is_functionally_connected['Region1'] - 1, is_functionally_connected['Region2'] - 1])
data['region', 'is_functionally_connected', 'region'].edge_index = torch.tensor(edges_region, dtype=torch.long)

# Definir el modelo, optimizador y parámetros de entrenamiento
hidden_channels = 64
out_channels = 2
num_layers = 2
model = HeteroGNN(hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entrenar el modelo y guardar los pesos
pretrain_model(data, model, optimizer, epochs=50)

# Visualización del entrenamiento: ya se muestra la pérdida en cada época
# Se puede observar que la pérdida disminuye con el tiempo
