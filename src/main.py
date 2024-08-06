from neo4j import GraphDatabase
import pandas as pd
import torch
from torch_geometric.data import Data

# To convert categorical values to numerical ones
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

    def get_graph_data(self):
        with self.driver.session() as session:
            try:
                # Retrieving nodes 
                nodes = session.run("MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties")
                # session.run returns Record instances of object type Result
                # Retrieving edges
                edges = session.run("MATCH (n)-[r]->(m) RETURN elementId(n) AS source, elementId(m) AS target, type(r) AS type, properties(r) AS properties")
                
                # Converting to DataFrame format for the creation of tensors
                nodes_df = pd.DataFrame([record.data() for record in nodes])
                # record.data() to obtain record as dictionary
                edges_df = pd.DataFrame([record.data() for record in edges])
                return nodes_df, edges_df
            except Exception as e:
                print(f"Error retrieving graph data: {e}")
                return pd.DataFrame(), pd.DataFrame()

# Conversion to suitable data types
def transform_data(nodes_df, edges_df):
    nodes_df = nodes_df.fillna('')  # In case of NaN values, it replaces them with empty strings

    # Maps to numerical type using a dictionary of properties for each node
    def map_edge_properties(edge_type, properties):
    # Ensure properties is a dictionary
        if not isinstance(properties, dict):
            properties = {}

        if edge_type == 'has_region':
            # Extract features for 'has_region' type edges
            return [
            float(properties.get('gm_volume', 0)),  # Default to 0 if key is missing
            float(properties.get('regional_ct', 0))  # Default to 0 if key is missing
        ]
        else:  # Handle other edge types
            return [
            float(properties.get('median_correlation', 0))  # Default to 0 if key is missing
        ]

    # Apply the mapping function to the edges DataFrame
    edges_df['edge_features'] = edges_df.apply(lambda row: map_edge_properties(row['type'], row['properties']), axis=1)

    # Conversion to PyTorch tensors
    edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long).contiguous()
    edge_features = torch.tensor(edges_df['edge_features'].tolist(), dtype=torch.float)
    
    # We specify the node property "features" as the zero-layer node embeddings
    features = torch.tensor(nodes_df['features'].tolist(), dtype=torch.float)
    # We specify the node property "diagnosis" as class labels
    labels = torch.tensor(labels.tolist(), dtype=torch.long)

    # PyG Data object
    data = Data(x=features, edge_index=edge_index, edge_attr=edge_features,y=labels)
    
    return data

def main():
    # Initialize GraphManager and retrieve data
    graph_manager = GraphManager(uri, auth)
    nodes_df, edges_df = graph_manager.get_graph_data()
    
    # Preprocess data
    data = transform_data(nodes_df, edges_df)
    
    print("Data Transformation Completed")
    print(data)

    # Close connection
    graph_manager.close()

if __name__ == "__main__":
    main()
