import sys
import os
import matplotlib.pyplot as plt  # Importar Matplotlib para graficar
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

# Rutas a los archivos CSV en la carpeta 'data'
data_dir = '/workspaces/DetectionOfMCI_TFG/data'  # Cambia a la ruta relativa de la carpeta 'data'
has_region_path = os.path.join(data_dir, 'has_region.csv')
is_connected_to_CN_path = os.path.join(data_dir, '../data/is_connected_to_CN.csv')
is_connected_to_MCI_path = os.path.join(data_dir, '../data/is_connected_to_MCI.csv')
is_functionally_connected_path = os.path.join(data_dir, '../data/is_functionally_connected.csv')
regions_path = os.path.join(data_dir, '../data/regions.csv')
subjects_path = os.path.join(data_dir, '../data/subjects.csv')

# Cargar CSVs
has_region = pd.read_csv(has_region_path)
is_connected_to_CN = pd.read_csv(is_connected_to_CN_path)
is_connected_to_MCI = pd.read_csv(is_connected_to_MCI_path)
is_functionally_connected = pd.read_csv(is_functionally_connected_path)
regions = pd.read_csv(regions_path)
subjects = pd.read_csv(subjects_path)

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

# Lista para guardar las pérdidas (losses)
losses = []

# Entrenar el modelo y guardar los pesos
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    target = torch.randint(0, 2, (len(data['subject'].x),))  # Target ficticio para demostración
    loss = F.cross_entropy(out['subject'], target)
    loss.backward()
    optimizer.step()
    
    # Guardar la pérdida actual
    losses.append(loss.item())
    
    # Mostrar progreso
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Guardar los pesos del modelo
model_file = 'trained_heteroGNN_model.pth'
torch.save(model.state_dict(), model_file)
print(f"Modelo guardado en {model_file}")

# Guardar la gráfica de la pérdida
plt.figure()
plt.plot(range(1, epochs + 1), losses, label="Training Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')  # Guardar la visualización en un archivo
print("Gráfica de pérdida guardada como 'training_loss.png'")
