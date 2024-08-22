from neo4j import GraphDatabase
from src.model import training_model, evaluation_model
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.nn import GCN

# For catgorical to numerical conversion
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
                has_region_edges = session.run("MATCH (s:Subject)-[h:HAS_REGION]-(r:Region) RETURN s.subject_id AS source, r.roi_id AS target,"
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
    region_df = pd.DataFrame( region_nodes)
    has_region_df = pd.DataFrame(has_region_edges)
    functionally_connected_df = pd.DataFrame(functionally_connected_edges)

    # Maps categorical values to numerical ones
    subject_df['sex'] = subject_df['sex'].map(sex_map)
    subject_df['diagnosis'] = subject_df['diagnosis'].map(diagnosis_map)
    # Subject_id is a string so it is converted to an integer taking the last four digits
    has_region_df['source'] = (has_region_df['source'].str[-4:]).astype(int)

    # Conversion to PyTorch tensors
    # PyG object describing a heterogeneous graph (multiple node and/or edge types) in disjunct storage objects
    data = HeteroData()
    
    # Subject node feature matrix
    data['subject'].x = torch.tensor(subject_df[['age', 'sex']].values, dtype=torch.float)
    # Target to train against 
    data['subject'].y = torch.tensor(subject_df['diagnosis'].values, dtype=torch.long)

    # Region node feature matrix
    # Using index as dummy feature
    data['region'].x = torch.tensor(region_df.index.values, dtype=torch.float)  

    # Graph connectivity with shape [2, num_edges] for has_region edges
    # Convert list to a single array to avoid efficiency warning
    has_region_edge_index = np.array([has_region_df['source'].values, has_region_df['target'].values])
    data[('subject', 'has_region', 'region')].edge_index = torch.tensor(
        has_region_edge_index, dtype=torch.long)
    # Has_region edge feature matrix with shape [num_edges, num_edge_features]= [num_edges, 2]
    data[('subject', 'has_region', 'region')].edge_attr = torch.tensor(
        has_region_df[['gm_volume', 'regional_ct']].values, dtype=torch.float)

    # Graph connectivity with shape [2, num_edges] for is_functionally_connected edges
    functionally_connected_edge_index= np.array([functionally_connected_df['source'].values, 
        functionally_connected_df['target'].values])
    data[('region', 'is_functionally_connected', 'region')].edge_index = torch.tensor(
        functionally_connected_edge_index, dtype=torch.long)
    # Is_functionally_connected edge feature matrix with shape [num_edges, num_edge_features]= [num_edges, 2]
    data[('region', 'is_functionally_connected', 'region')].edge_attr = torch.tensor(
        functionally_connected_df[['corr_mci', 'corr_cn']].values, dtype=torch.float
    )
    print(data.is_undirected()) # Returns false
    return data

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
    data = transform(data)

    # Initialization of the GCN model
    num_node_features= 2
    num_labels= 2
    # 50% of the node features are randomly set to zero in each layer during training to avoid overfitting
    pred = GCN(in_channels=num_node_features, hidden_channels=16, num_layers=2, out_channels=num_labels, dropout=0.5) 
    # Using Adam optimization algorithm
    optimizer = torch.optim.Adam(pred.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Training and evaluation of the model
    training_model(pred, data, optimizer)
    evaluation_model(pred, data)

if __name__ == "__main__":
    main()
