import torch
import torch.nn.functional as F 
import torch_geometric.nn as gnn
from torch.cuda.amp import GradScaler, autocast

# Definición del modelo HeteroGNN
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        # Configuración de capas convolucionales para grafo heterogéneo
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = gnn.HeteroConv({
                # Paso de mensajes de sujetos a regiones
                ('region', 'has_region', 'subject'): gnn.SAGEConv((-1, -1), hidden_channels),
                # Paso de mensajes entre regiones basado en conexiones funcionales
                ('region', 'is_functionally_connected', 'region'): gnn.GCNConv(-1, hidden_channels)
            }, aggr='sum')
            self.convs.append(conv)
        
        # Capa lineal final para producir la salida
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Aplicar las capas convolucionales y procesar los datos
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return {key: self.lin(x) for key, x in x_dict.items()}

# Función para entrenar el modelo y guardar los pesos
def pretrain_model(data, model, optimizer, epochs=50, lr=0.01):
    scaler = GradScaler()  # Para entrenamiento con precisión mixta

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        with autocast():  # Habilitar precisión mixta
            out = model(data.x_dict, data.edge_index_dict)
            target = torch.randint(0, 2, (len(data['subject'].x),))  # Target ficticio para demostración
            loss = F.cross_entropy(out['subject'], target)
        
        # Backpropagación y optimización
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Mostrar progreso
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Guardar los pesos del modelo
    model_file = 'trained_heteroGNN_model.pth'
    torch.save(model.state_dict(), model_file)
    print(f"Modelo guardado en {model_file}")

    return model_file

# Función para cargar un modelo guardado
def load_model(model_file, hidden_channels, out_channels, num_layers):
    model = HeteroGNN(hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers)
    model.load_state_dict(torch.load(model_file))
    return model
