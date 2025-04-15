from dig.threedgraph.method import SphereNet
from dig.threedgraph.evaluation import ThreeDEvaluator
import torch
import numpy as np
from torch_geometric.data import Data
from pymatgen.core.structure import Structure, Molecule
from torch_geometric.loader import DataLoader
import ocelotml
from typing import Literal
import os

FP_DIR = os.path.dirname(ocelotml.__file__)

CHECKPOINT_PATHS = {
 'hh' : os.path.join(FP_DIR,'./models/hh.pt'),
 'll' : os.path.join(FP_DIR,'./models/ll.pt'),
}

def load_models(checkpoint: Literal['hh','ll']):

    model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
                      hidden_channels=128, out_channels=1, int_emb_size=64,
                      basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                      num_spherical=3, num_radial=6, envelope_exponent=5,
                      num_before_skip=1, num_after_skip=2, num_output_layers=3)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATHS[checkpoint],map_location=torch.device('cpu'), weights_only = False
    )["model_state_dict"])
    model.eval()

    return model

def _molecules_to_data_list(molecules) -> list[Data]:
    """providing a list of pymatgen molecules, produce a datalist

    Parameters
    ----------
    molecules: Molecule
        pymatgen molecule

    Returns
    -------
    list[Data]
        List of data objects to be processed

    
    """
    data_list = []
    for molecule in molecules:
        R_i = torch.tensor(molecule.cart_coords, dtype=torch.float32)
        z_i = torch.tensor(np.array(molecule.atomic_numbers), dtype=torch.int64)
        data = Data(pos=R_i, z=z_i)
        data_list.append(data)
    return data_list

def predict_from_list(molecules: list[Molecule], model) -> list:
    """Build a Dataloader batch and predict from list of molecules.

    Parameters
    ----------
    molecules: list[Molecule]

    Returns
    -------
    list   
      list of coupling 
    
    """
    molecules_data_list = _molecules_to_data_list(molecules=molecules)
    loader = DataLoader(molecules_data_list, batch_size=len(molecules_data_list), shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            # Assuming each graph gives one prediction; adjust extraction as needed.
            batch_preds = out.detach().cpu().numpy()  # shape: (n_graphs, 1)
            predictions.extend([round(pred[0], 3) for pred in batch_preds])
    return predictions


def predict(batch, model):
    model.eval()
    results = []
    for data in batch:
        prediction = model(data)
        results.append(round(prediction.detach().numpy()[-1][0],3))
    return results

def predict_from_molecule(molecule, model):
    """
    args: 

    molecule (Molecule): pymatgen Molecule object
    checkpoint (str): path to checkpoint file 

    Description:

    Makes prediction from Molecule object
    """
    R_i = torch.tensor(molecule.cart_coords, dtype=torch.float32)
    z_i = torch.tensor(np.array(molecule.atomic_numbers), dtype=torch.int64)
    data = Data(pos=R_i, z=z_i,)
    # batch = DataLoader([data], batch_size=1)
    # Use torch.no_grad() to avoid building the computation graph
    # Manually assign a batch attribute: all nodes belong to batch 0
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    
    # Use no_grad to avoid creating the computation graph during inference
    with torch.no_grad():
        output = model(data)
        # Ensure output is on CPU and detach from computation graph
        output_np = output.detach().cpu().numpy()
        # Adjust indexing based on expected output dimensions
        prediction = round(output_np[-1][0], 3)
    return prediction

def predict_from_file(filename, model):
    """
    args: 
    
    filename (str): path to xyz file
    checkpoint (str): path to checkpoint file

    Description:

    Makes prediction from XYZ file with dimer coordinates
    """
    molecule = Molecule.from_file(filename)
    return predict_from_molecule(molecule, model)