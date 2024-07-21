from neo4j import GraphDatabase
import csv
import os

# Connection to Neo4j
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

def load_csv_data():
    # Loads data from regions.csv
    with open(os.path.join('data', 'regions.csv'), 'r') as file:
        reader = csv.DictReader(file)
        regions = [row for row in reader]
        print("Loaded csv data.")

    # Loads data from subjects.csv
    with open(os.path.join('data', 'subjects.csv'), 'r') as file:
        reader = csv.DictReader(file)
        subjects = [row for row in reader]
        print("Loaded csv data.")

    # Loads data from has_region.csv
    with open(os.path.join('data', 'has_region.csv'), 'r') as file:
        reader = csv.DictReader(file)
        has_region = [row for row in reader]
        print("Loaded csv data.")

    # Loads data from is_connected_to_CN.csv
    with open(os.path.join('data', 'is_connected_to_CN.csv'), 'r') as file:
        reader = csv.DictReader(file)
        connected_to_cn = [row for row in reader]
        print("Loaded csv data.")

    # Loads data from is_connected_to_MCI.csv
    with open(os.path.join('data', 'is_connected_to_MCI.csv'), 'r') as file:
        reader = csv.DictReader(file)
        connected_to_mci = [row for row in reader]
        print("Loaded csv data.")

    return regions, subjects, has_region, connected_to_cn, connected_to_mci


def create_graph(session, regions, subjects, has_region, connected_to_cn, connected_to_mci):

    if not regions or not subjects or not has_region or not connected_to_cn or not connected_to_mci:
        print("CSV data loading failed. Aborting graph creation.")
        return
    
    try:
        with session.begin_transaction() as tx:
            print("Starting to create the graph in Neo4j...")
        
            # Clears out the database
            tx.run("MATCH (n) DETACH DELETE n")
            print("Cleared the database.")

            # Creates ROI nodes
            for region in regions:
                tx.run("CREATE (:ROI {id: $id, region: $region})",
                    id=int(region['roi_id']), region=region['roi_name'])
                print("Created ROI nodes.")

            # Creates subject nodes
            for subject in subjects:
                tx.run("CREATE (:Subject {id: $id, age: $age, sex: $sex, diagnosis: $diagnosis})",
                    id=subject['subject_id'], age=int(subject['age']),
                    sex=subject['sex'], diagnosis=subject['diagnosis'])
                print("Created Subject nodes.")

            # Creates subject-ROI relationships
            for hr in has_region:
                tx.run("""
                MATCH (s:Subject {id: $subject_id}), (r:ROI {id: $region_id})
                CREATE (s)-[:HAS_REGION {cortical_thickness: $cortical_thickness, volume: $volume}]->(r)
                """, subject_id=hr['subject_id'], region_id=int(hr['region_id']),
                    cortical_thickness=float(hr['cortical_thickness']), volume=float(hr['volume']))
                print("Created HAS_REGION relationships.")

            # Creates ROI-ROI relationships using CN correlation values
            for conn in connected_to_cn:
                tx.run("""
                MATCH (r1:ROI {id: $region1}), (r2:ROI {id: $region2})
                CREATE (r1)-[:CONNECTED_TO_CN {pearson_correlation: $median_correlation}]->(r2)
                """, region1=int(conn['Region1']), region2=int(conn['Region2']),
                    median_correlation=float(conn['Median_Correlation']))
                print("Created CONNECTED_TO_CN relationships.")

            # Creates ROI-ROI relationships using MCI correlation values
            for conn in connected_to_mci:
                tx.run("""
                MATCH (r1:ROI {id: $region1}), (r2:ROI {id: $region2})
                CREATE (r1)-[:CONNECTED_TO_MCI {pearson_correlation: $median_correlation}]->(r2)
                """, region1=int(conn['Region1']), region2=int(conn['Region2']),
                    median_correlation=float(conn['Median_Correlation']))
            print("Created CONNECTED_TO_MCI relationships.")

            tx.commit()
    
    except Exception as e:
        print(f"Failed to create the neo4j graph: {e}")
        # Rolls back the transaction on error to avoid partial updates
        session.rollback()
        raise


# Consulta para obtener los nodos y aristas
def get_graph_data(session):
    try:
        nodes = session.run("MATCH (n) RETURN n")
        edges = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")
        return {
            "nodes": [record["n"].data() for record in nodes],
            "edges": [{"from": record["n"].id, "to": record["m"].id, "relationship": record["r"].type} for record in edges]
        }
    except Exception as e:
        print(f"Error retrieving graph data: {e}")
        return [], []
    
def main():
    graph_manager = GraphManager(uri, auth)
    regions, subjects, has_region, connected_to_cn, connected_to_mci = load_csv_data()
       
    with graph_manager.driver.session() as session:
        create_graph(session, regions, subjects, has_region, connected_to_cn, connected_to_mci)
    
        # Retrieve graph data
        node_list, edge_list = get_graph_data(session)
        
        # Print or process retrieved data
        print("Nodes:", node_list)
        print("Edges:", edge_list)

    graph_manager.close()

if __name__ == "__main__":
    main()
