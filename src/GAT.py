import json 
import os
import enum

import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import igraph as ig
from igraph import plot

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

# Supported datasets - only PPI in this notebook  
class DatasetType(enum.Enum):
    PPI = 0

class GraphVisualizationTool(enum.Enum):
    IGRAPH = 0

# Files locations
DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')
PPI_PATH = os.path.join(DATA_DIR_PATH, 'ppi')
PPI_URL = 'https://data.dgl.ai/dataset/ppi.zip'  # preprocessed PPI data from Deep Graph Library

# PPI specific constants
PPI_NUM_INPUT_FEATURES = 50
PPI_NUM_CLASSES = 50

def json_read(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def load_graph_data(training_config, device):
    dataset_name = training_config['dataset_name'].lower()
    should_visualize = training_config['should_visualize']

    if dataset_name == DatasetType.PPI.name.lower(): # Protein-Protein Interaction dataset 
        if not os.path.exists(PPI_PATH):
            os.makedirs(PPI_PATH)
             
             #Download dataset zip
            zip_tmp_path = os.path.join(PPI_PATH, 'ppi.zip')
            download_url_to_file(PPI_URL, zip_tmp_path)

            with zipfile.Zipfile(zip_tmp_path) as zf:
                zf.extracall(path=PPI_PATH)
            print(f'unzipping to: {PPI_PATH} finished.')

            os.remove(zip_tmp_path)
            print(f'removing tmp file {zip_tmp_path}.')

        # collect train/val/test graphs 
        edge_index_list = []
        node_features_list = []
        node_labels_list = []

        # dynamically determine how many graphs there are per split
        num_graphs_per_split_cumulative = [0]

        # optimisation trick for only need to test in the playground.py
        splits = ['test'] if training_config['ppi_load_test_only'] else ['train', 'valid', 'test']

        for split in splits:
            # PPI has 50 features per node, it's a combination of positinal gene sets, motif gene sets,
            # and immunological signatures - you can treat it as a black box
            # shape = (NS, 50) - where NS is the number of (N)odes in the training/val/test (S)plit
            # Note: node features are already preprocessed
            node_features = np.load(os.path.join(PPI_PATH, f'{split}_feats.npy'))

            # PPI has 121 labels and each node can have multiple labels associated
            # SHAPE = (NS, 121)
            node_labels = np.load(os.path.join(PPI_PATH, f'{split}_labels.npy'))

            # Graph topology stored in a special nodes-links NetworkX format
            nodes_links_dict = json_read(os.path.join(PPI_PATH, f'{split}_graph.json'))
            
            # PPI contains undirected graphs with self edges - 20 train graphs, 2 validation graphs and 2 test graphs
            # The reason to use a NetworkX's directed graph is because both directions are needed to model 
            # because of the edge index and the way GAT implementation #3 works
            collection_of_graphs = nx.DiGraph(json_graph.node_link_graph(nodes_links_dict))
            # for each node in the above collection, ids specify to which graph the node belongs to
            graph_ids = np.load(os.path.join(PPI_PATH, F'{split}_graph_id.npy'))
            num_graphs_per_split_cumulative.append(num_graphs_per_split_cumulative[-1] + len(np.unique(graph_ids)))
            # print(np.unique(num_graphs_per_split_cumulative))

            # Split the collection of graphs into separate PPI graphs
            for graph_id in range(np.min(graph_ids), np.max(graph_ids) + 1):
                mask = graph_ids == graph_id # find the nodes which belong to the current graph 
                graph_node_ids = np.asarray(mask).nonzero()[0]
                graph = collection_of_graphs.subgraph(graph_node_ids) # returns the induced subgraph over these nodes
                print(f'Loading {split} graph {graph_id} to CPU. '
                      f'It has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.')

                # shape = (2, E) where E is the number of edges in the graph
                # Note: leaving the tensors on CPU and load them to GPU in the training loop on-the-gly as VRAM
                # is a scarcer resource than CPU's RAM and the whole PPI dataset can't fit during the training.
                edge_index = torch.tensor(list(graph.edges),  dtype=torch.long).transpose(0,1).contiguous()
                edge_index = edge_index - edge_index.min() #bring the edges to [0, num_of_nodes] range
                edge_index_list.append(edge_index)


                # shape = (N, 50) where N is the number of nodes in the graph
                node_features_list.append(torch.tensor(node_features[mask], dtype=torch.float))
                
                #shape (N, 121), BCEWithLOgitsLoss doesn't require long/int64 so saving some memory using float32
                node_labels_list.append(torch.tensor(node_labels[mask], dtype=torch.float))
                
                if should_visualize:
                    plot_in_out_degree_distributions(edge_index.numpy(), graph.number_of_nodes(), dataset_name)
                    visualize_graph(edge_index.numpy(), node_labels[mask], dataset_name)

            #
            # Prepare graph data loaders
            #

            # As mentioned, if only test data loader is needed:
        if training_config['ppi_load_test_only']:
            data_loader_test = GraphDataLoader(
                    node_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                    node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                    edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                    batch_size = training_config['batch_size'],
                    shuffle=False,
                    )
            return data_loader_test

        else:
            data_loader_train = GraphDataLoader(
                    node_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                    node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                    edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                    batch_size = training_config['batch_size'],
                    shuffle=True
            ),

            data_loader_val = GraphDataLoader(
                    node_features_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
                    node_labels_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
                    edge_index_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
                    batch_size = training_config['batch_size'],
                    shuffle=False # no need to shuffle the validation and test graphs
            ),

            data_loader_test = GraphDataLoader(
                    node_features_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
                    node_labels_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
                    edge_index_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
                    batch_size = training_config['batch_size'],
                    shuffle=False # no need to shuffle the validation and test graphs
            ),

            return data_loader_train, data_loader_val, data_loader_test

    else:
        raise Exception(f'{dataset_name} not yet supported')

class GraphDataLoader(DataLoader):
    # When dealing with batches it's good idea to use PyTorch's classes (Dataset/Dataloader)
    def __init__(self, node_features_list, node_labels_list, edge_index_list, batch_size = 1, shuffle = False):
        graph_dataset = GraphDataset(node_features_list, node_labels_list, edge_index_list)
        # custom collate function to make it work
        super().__init__(graph_dataset, batch_size, shuffle, collate_fn=graph_collate_fn)

class GraphDataset(Dataset):
    # fetch a single graph from the split when graphdataloader 'asks' for it
    def __init__(self, node_features_list, node_labels_list, edge_index_list):
        self.node_features_list = node_features_list
        self.node_labels_list = node_labels_list
        self.edge_index_list = edge_index_list

    # 2 interface functions that need to be defined are len and getitem so that DataLoader can do it's magic
    def __len__(self):
        return len(self.edge_index_list)

    def __getitem__(self, idx):
        return self.node_features_list[idx], self.node_labels_list[idx], self.edge_index_list[idx]

def graph_collate_fn(batch):
    # The main idea here is to take multiple graphs from PPI as defined by the batch size
    # and merge them into a single graph with multiple connected components.
    # It's important to adjust the node ids in edge indices such that they form a consecutive range. Otherwise
    # the scatter functions in the implementation 3 will fail.
    # :param batch: contains a list of edge_index, node_features, node_labels tuples (as provided by the GraphDataset)
    
    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    num_nodes_seen = 0
    
    for features_labels_edge_index_tuple in batch:
        # Just collect into separate lists
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])
        
        edge_index = features_labels_edge_index_tuple[2] # all of the components are in the [0, N] range
        edge_index_list.append(edge_index + num_nodes_seen) # very importat as this transalte the range of this component
        num_nodes_seen += len(features_labels_edge_index_tuple[1]) # update the number of nodes we've seen so far

    # Merge the PPI graphs inot a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)

    return node_features, node_labels, edge_index


def plot_in_out_degree_distributions(edge_index, num_of_nodes, dataset_name):
    """
        Note: It would be easy to do various kinds of powerful network analysis using igraph/networkx, etc.
        I chose to explicitly calculate only the node degree statistics here, but you can go much further if needed and
        calculate the graph diameter, number of triangles and many other concepts from the network analysis field.

    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
        
    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'

    # Store each node's input and output degree (they're the same for undirected graphs such as Cora/PPI)
    in_degrees = np.zeros(num_of_nodes, dtype=int)
    out_degrees = np.zeros(num_of_nodes, dtype=int)

    # Edge index shape = (2, E), the first row contains the source nodes, the second one target/sink nodes
    # Note on terminology: source nodes point to target/sink nodes
    num_of_edges = edge_index.shape[1]
    for cnt in range(num_of_edges):
        source_node_id = edge_index[0, cnt]
        target_node_id = edge_index[1, cnt]

        out_degrees[source_node_id] += 1  # source node points towards some other node -> increment it's out degree
        in_degrees[target_node_id] += 1  # similarly here

    hist = np.zeros(np.max(out_degrees) + 1)
    for out_degree in out_degrees:
        hist[out_degree] += 1

    fig = plt.figure(figsize=(12,8), dpi=100)  # otherwise plots are really small in Jupyter Notebook
    fig.subplots_adjust(hspace=0.6)

    plt.subplot(311)
    plt.plot(in_degrees, color='red')
    plt.xlabel('node id'); plt.ylabel('in-degree count'); plt.title('Input degree for different node ids')

    plt.subplot(312)
    plt.plot(out_degrees, color='green')
    plt.xlabel('node id'); plt.ylabel('out-degree count'); plt.title('Out degree for different node ids')

    plt.subplot(313)
    plt.plot(hist, color='blue')
    plt.xlabel('node degree'); plt.ylabel('# nodes for a given out-degree'); plt.title(f'Node out-degree distribution for {dataset_name} dataset')
    plt.xticks(np.arange(0, len(hist), 20.0))

    plt.grid(True)
    plt.show()
def visualize_graph():
    pass



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
        'dataset_name': DatasetType.PPI.name,
        'should_visualize': False,
        'batch_size': 1,
        'ppi_load_test_only': False,
}

data_loader_train, data_loader_val, data_loader_test = load_graph_data(config, device)
# Fetch a single batch from the train graph data loader
batch = next(iter(data_loader_train))
node_features, node_labels, edge_index = next(iter(batch))

print('*' * 20)
print(node_features.shape, node_features.dtype)
print(node_labels.shape, node_labels.dtype)
print(edge_index.shape, edge_index.dtype)

num_of_nodes = len(node_labels)
plot_in_out_degree_distributions(edge_index, num_of_nodes, config['dataset_name'])

"""
Check out this blog for available graph visualization tools:
    https://towardsdatascience.com/large-graph-visualization-tools-and-approaches-2b8758a1cd59

Basically depending on how big your graph is there may be better drawing tools than igraph.

Note: I unfortunatelly had to flatten this function since igraph is having some problems with Jupyter Notebook,
we'll only call it here so it's fine!

"""

dataset_name = config['dataset_name']
visualization_tool=GraphVisualizationTool.IGRAPH

if isinstance(edge_index, torch.Tensor):
    edge_index_np = edge_index.cpu().numpy()

if isinstance(node_labels, torch.Tensor):
    node_labels_np = node_labels.cpu().numpy()

num_of_nodes = len(node_labels_np)
edge_index_tuples = list(zip(edge_index_np[0, :], edge_index_np[1, :]))  # igraph requires this format

# Construct the igraph graph
ig_graph = ig.Graph()
ig_graph.add_vertices(num_of_nodes)
ig_graph.add_edges(edge_index_tuples)

# Prepare the visualization settings dictionary
visual_style = {}

# Defines the size of the plot and margins
visual_style["bbox"] = (650, 650)
visual_style["margin"] = 5

# I've chosen the edge thickness such that it's proportional to the number of shortest paths (geodesics)
# that go through a certain edge in our graph (edge_betweenness function, a simple ad hoc heuristic)

# line1: I use log otherwise some edges will be too thick and others not visible at all
# edge_betweeness returns < 1 for certain edges that's why I use clip as log would be negative for those edges
# line2: Normalize so that the thickest edge is 1 otherwise edges appear too thick on the chart
# line3: The idea here is to make the strongest edge stay stronger than others, 6 just worked, don't dwell on it

edge_weights_raw = np.clip(np.log(np.asarray(ig_graph.edge_betweenness())+1e-16), a_min=0, a_max=None)
edge_weights_raw_normalized = edge_weights_raw / np.max(edge_weights_raw)
edge_weights = [w**6 for w in edge_weights_raw_normalized]
visual_style["edge_width"] = edge_weights

# A simple heuristic for vertex size.
visual_style["vertex_size"] = [deg / 10 for deg in ig_graph.degree()]

# Set the layout - the way the graph is presented on a 2D chart. Graph drawing is a subfield for itself!
# I used "Kamada Kawai" a force-directed method, this family of methods are based on physical system simulation.
visual_style["layout"] = ig_graph.layout_kamada_kawai()

print('Plotting results ... (it may take couple of seconds).')
ig.plot(ig_graph, **visual_style)
plot(ig_graph, "graph_output.png", **visual_style)
# This website has got some awesome visualizations check it out:
# http://networkrepository.com/graphvis.php?d=./data/gsm50/labeled/cora.edges
