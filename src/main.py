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
                nodes = session.run("MATCH (n) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties")
                # session.run returns Record instances of object type Result
                # Retrieving edges
                edges = session.run("MATCH (n)-[r]->(m) RETURN id(n) AS source, id(m) AS target, type(r) AS type, properties(r) AS properties")
                
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
    def map_node_properties(properties):
        numerical = []
        if 'sex' in properties:
            numerical.append(sex_map.get(properties.get('sex', ''), -1))
        if 'age' in properties:
            numerical.append(float(properties.get('age', 0)))
        return numerical
    nodes_df['features'] = nodes_df['properties'].apply(map_node_properties)
    # Extracting and converting labels separately
    labels = nodes_df['properties'].apply(lambda props: diagnosis_map.get(props.get('diagnosis', ''), -1))
    
    # Maps to numerical type using a dictionary of properties for each edge
    def map_edge_properties(row):
        properties = row['properties']
        edge_type = row['type']
        
        if edge_type == 'has_region':
            return [
                float(properties.get('gm_volume', 0)), # Set to 0 if missing
                float(properties.get('regional_ct', 0)) 
            ]
        else:  # Other edge types
            return [
                float(properties.get('median_correlation', 0))  # Default value if missing
            ]
    
    edges_df['edge_features'] = edges_df['properties'].apply(map_edge_properties)

    # Convert edge indices to integers
    edges_df['source'] = edges_df['source'].astype(int)  
    edges_df['target'] = edges_df['target'].astype(int)  
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
