import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for molecular property prediction using molecular graphs
    """
    
    def __init__(self, 
                 node_features: int = 74,
                 edge_features: int = 12,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 output_dim: int = 1,
                 dropout_rate: float = 0.2,
                 conv_type: str = 'gcn'):
        super(GraphNeuralNetwork, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.conv_type = conv_type
        
        # Node embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if conv_type == 'gcn':
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'gat':
                self.conv_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Graph-level prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, data):
        """Forward pass through the GNN"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Node embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # Graph convolution layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Graph-level representation
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1)
        
        # Prediction
        out = self.prediction_head(x_graph)
        return out

class MolecularGraphProcessor:
    """Convert molecules to PyTorch Geometric graph format"""
    
    def __init__(self):
        self.atom_features = [
            'atomic_num', 'formal_charge', 'chiral_tag', 'hybridization',
            'num_explicit_hs', 'is_aromatic', 'degree', 'total_valence'
        ]
        
        self.bond_features = [
            'bond_type', 'stereo', 'is_conjugated', 'is_in_ring'
        ]
    
    def get_atom_features(self, atom):
        """Extract atom features"""
        features = []
        
        # Atomic number (one-hot encoded for common elements)
        atomic_num = atom.GetAtomicNum()
        common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
        atom_encoding = [1 if atomic_num == x else 0 for x in common_atoms]
        features.extend(atom_encoding)
        
        # Other atomic properties
        features.append(atom.GetFormalCharge())
        features.append(atom.GetChiralTag())
        features.append(atom.GetHybridization())
        features.append(atom.GetTotalNumHs())
        features.append(int(atom.GetIsAromatic()))
        features.append(atom.GetDegree())
        features.append(atom.GetTotalValence())
        
        # Atom in ring
        features.append(int(atom.IsInRing()))
        
        # Additional descriptors
        features.extend([0] * 60)  # Padding to reach 74 features
        
        return features
    
    def get_bond_features(self, bond):
        """Extract bond features"""
        features = []
        
        # Bond type
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                     Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_encoding = [1 if bond.GetBondType() == bt else 0 for bt in bond_types]
        features.extend(bond_encoding)
        
        # Stereo
        features.append(bond.GetStereo())
        
        # Is conjugated
        features.append(int(bond.GetIsConjugated()))
        
        # Is in ring
        features.append(int(bond.IsInRing()))
        
        # Padding
        features.extend([0] * 5)  # Padding to reach 12 features
        
        return features
    
    def mol_to_graph(self, mol) -> Optional[Data]:
        """Convert RDKit molecule to PyTorch Geometric Data object"""
        if mol is None:
            return None
        
        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        
        if not atom_features:
            return None
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Edge indices and features
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions
            edge_indices.extend([[i, j], [j, i]])
            bond_feat = self.get_bond_features(bond)
            edge_features.extend([bond_feat, bond_feat])
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 12), dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES to graph"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            return self.mol_to_graph(mol)
        except:
            return None
    
    def batch_smiles_to_graphs(self, smiles_list: List[str]) -> Tuple[Optional[Batch], List[bool]]:
        """Convert batch of SMILES to graph batch"""
        graphs = []
        valid_mask = []
        
        for smiles in smiles_list:
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        
        if graphs:
            batch = Batch.from_data_list(graphs)
            return batch, valid_mask
        else:
            return None, valid_mask