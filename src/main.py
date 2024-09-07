import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neo4j import GraphDatabase
from model import training_model, evaluation_model, HeteroGNN
import pandas as pd
import torch
import torch.nn.functional 
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import to_undirected

# For categorical to numeric,ial conversion
sex_map = {'M': 0, 'F': 1}
diagnosis_map = {'CN': 0, 'MCI': 1}

uri = "neo4j+s://1bb5bfcc.databases.neo4j.io"
auth = ("neo4j", "6lW_NKJovTAXx2YNuB-t5L2PKtEX2uLfPZaQeKS1ods")

class GraphManager:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        try:
            self.driver.verify_connectivity()
            print("--")
            print("Connected to Neo4j successfully.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            exit(1)

    def close(self):
        self.driver.close()

    # Retrieving graph as a list of dictionaries
    def get_graph_data(self):
        with self.driver.session() as session:
            try:
                # Retrieving nodes
                subject_nodes = session.run("MATCH (s:Subject) RETURN s.subject_id AS id, s.diagnosis AS diagnosis, s.age AS age, s.sex AS sex")
                region_nodes = session.run("MATCH (r:Region) RETURN r.roi_id AS id, r.name AS name")
                # Retrieving edges
                has_region_edges = session.run("MATCH (r:Region)-[h:HAS_REGION]-(s:Subject) RETURN s.subject_id AS source, r.roi_id AS target,"
                                               "h.gm_volume AS gm_volume, h.regional_ct AS regional_ct")
                functionally_connected_edges= session.run("MATCH (r1:Region)-[f:IS_FUNCTIONALLY_CONNECTED]-(r2:Region) RETURN r1.roi_id "
                                                             "AS source, r2.roi_id AS target, f.corr_mci AS corr_mci, f.corr_cn AS corr_cn")
                
                # record.data() for dictionary conversion
                subject_nodes = [record.data() for record in subject_nodes]
                region_nodes = [record.data() for record in region_nodes]
                has_region_edges = [record.data() for record in has_region_edges]
                functionally_connected_edges = [record.data() for record in functionally_connected_edges]
                return subject_nodes, region_nodes, has_region_edges, functionally_connected_edges
            except Exception as e:
                print(f"Error retrieving graph data: {e}")
                return [], [], [], []


# Conversion to suitable data types
def transform_data(subject_nodes, region_nodes, has_region_edges, functionally_connected_edges):

    # Converting to pandas DataFrame format for the creation of tensors
    subject_df = pd.DataFrame(subject_nodes)
    region_df = pd.DataFrame(region_nodes)
    has_region_df = pd.DataFrame(has_region_edges)
    functionally_connected_df = pd.DataFrame(functionally_connected_edges)

    # Maps categorical values to numerical ones
    subject_df['sex'] = subject_df['sex'].map(sex_map)
    subject_df['diagnosis'] = subject_df['diagnosis'].map(diagnosis_map)
    # Subject_id is a string so it is converted to an integer taking the last four digits
    has_region_df['source'] = (has_region_df['source'].str[-4:]).astype(int)
    has_region_df['target'] = (has_region_df['target'].str[-4:]).astype(int)
    functionally_connected_df['source'] = (functionally_connected_df['source'].str[-4:]).astype(int)
    functionally_connected_df['target'] = (functionally_connected_df['target'].str[-4:]).astype(int)

    # Conversion to PyTorch tensors (multidimensional arrays)
    # PyG object describing a heterogeneous graph (multiple node and/or edge types) in disjunct storage objects
    data = HeteroData()
    
    # Subject node feature matrix
    # Structure: subject=torch.tensor([[age1, sex1], [age2, sex2],...])
    data['subject'].x = torch.tensor(subject_df[['age', 'sex']].values, dtype=torch.float)
    # Target to train against 
    data['subject'].y = torch.tensor(subject_df['diagnosis'].values, dtype=torch.long)

    # Region node feature matrix
    # Using index as dummy feature
    data['region'].x = torch.tensor(region_df.index.values, dtype=torch.float).unsqueeze(1) 

    # Graph connectivity with shape [2, num_edges] for has_region edges
    # Convert list to a single array to avoid efficiency warning
    has_region_edge_index = np.array([has_region_df['source'].tolist(), has_region_df['target'].tolist()])
    data[('region', 'has_region', 'subject')].edge_index = torch.tensor(
        has_region_edge_index, dtype=torch.long)
    # Has_region edge feature matrix with shape [num_edges, num_edge_features]= [num_edges, 2]
    data[('region', 'has_region', 'subject')].edge_attr = torch.tensor(
        has_region_df[['gm_volume', 'regional_ct']].values, dtype=torch.float)


    # Graph connectivity with shape [2, num_edges] for is_functionally_connected edges
    functionally_connected_edge_index = np.array([functionally_connected_df['source'].tolist(), functionally_connected_df['target'].tolist()])
    data[('region', 'is_functionally_connected', 'region')].edge_index = torch.tensor(
        functionally_connected_edge_index, dtype=torch.long)
    # Is_functionally_connected edge feature matrix with shape [num_edges, num_edge_features]= [num_edges, 2]
    data[('region', 'is_functionally_connected', 'region')].edge_attr = torch.tensor(
        functionally_connected_df[['corr_mci', 'corr_cn']].values, dtype=torch.float)

    # Convert to dictionaries
    data.edge_index_dict = {
    ('region', 'has_region', 'subject'): data[('region', 'has_region', 'subject')].edge_index,
    ('region', 'is_functionally_connected', 'region'): data[('region', 'is_functionally_connected', 'region')].edge_index
    #('region', 'rev_has_region', 'subject'): to_undirected(data[('subject', 'has_region', 'region')].edge_index)
    }

    data.edge_attr_dict = {
    ('region', 'has_region', 'subject'): data[('region', 'has_region', 'subject')].edge_attr,
    ('region', 'is_functionally_connected', 'region'): data[('region', 'is_functionally_connected', 'region')].edge_attr
    #('region', 'has_region', 'region'): data[('subject', 'has_region', 'region')].edge_attr,
    }

    print(data.is_undirected()) # Returns false
    print("After ToUndirected:")
    print(data.edge_index_dict)     
    print(data.is_undirected())

    return data

def check_hetero_data(x_dict, edge_index_dict, edge_attr_dict, expected_edge_types, model):
    # Check node feature tensors
    print("\nNode Features Check (x_dict):")
    for node_type, features in x_dict.items():
        print(f"- Node type: {node_type}")
        print(f"  Type: {type(features)}, Shape: {features.shape}")
        assert isinstance(features, torch.Tensor), f"{node_type} features are not a tensor!"
        print("--------")
        print(type(x_dict))

    # Check edge index tensors
    print("\nEdge Index Check (edge_index_dict):")
    for edge_type in edge_index_dict:
        assert edge_type in expected_edge_types, f"Edge type {edge_type} is not defined in the model's layers!"
        print("All edge types are valid.")
    


    print("\nModel Configuration Check:")
    print(f"Expected Edge Types in Model: {expected_edge_types}")

def run_debug_checks(model, data):
    # Extract expected edge types from model
    expected_edge_types = set()
    for conv_layer in model.convs:
        expected_edge_types.update(conv_layer.convs.keys())
    
    # Now you have the expected edge types for all layers
    print("Expected Edge Types from Model Layers:", expected_edge_types)

    # Run the verification function to check all tensors
    check_hetero_data(data.x_dict, data.edge_index_dict, data.edge_attr_dict, expected_edge_types, model)


def main():
    # Initialize GraphManager and retrieve data
    graph_manager = GraphManager(uri, auth)
    subject_nodes, region_nodes, has_region_edges, functionally_connected_edges = graph_manager.get_graph_data()
    
    # Preprocess data
    data = transform_data(subject_nodes, region_nodes, has_region_edges, functionally_connected_edges)
    print("Data transformation completed.")
    print(data)
    
    # Close connection
    graph_manager.close()

    # 20% of the nodes will be randomly selected for the test and validation sets
    transform = RandomNodeSplit(num_test=0.1, num_val=0.1)

    edge_indices = data.edge_index_dict[('region', 'is_functionally_connected', 'region')]
    node_indices = data.x_dict['region'].size(0)  # Number of nodes in the 'region' node type

    # Check for invalid edge indices
    invalid_indices = edge_indices[edge_indices >= node_indices]
    if invalid_indices.numel() > 0:
        print(f"Invalid edge indices found: {invalid_indices}")
        
    # Initialization of the GNN model
    num_node_features= 2 #sex, age
    num_labels= 2 #CN/MCI
    # 50% of the node features are randomly set to zero in each layer during training to avoid overfitting
    pred = HeteroGNN(hidden_channels=64, out_channels=num_labels, num_layers=2, ) 

    run_debug_checks(pred, data)

    # Using Adam optimization algorithm
    optimizer = torch.optim.Adam(pred.parameters(), lr=0.01, weight_decay=5e-4)
    print(f"Data.x_dict type: {type(data.x_dict)}")
    # Training and evaluation of the model
    training_model(pred, data, optimizer)
    
    print(f"Data.x_dict type: {type(data.x_dict)}")
    evaluation_model(pred, data)
    
    print(f"Data.x_dict type: {type(data.x_dict)}")

if __name__ == "__main__":
    main()
