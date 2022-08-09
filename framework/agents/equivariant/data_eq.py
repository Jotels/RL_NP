from typing import List
import sys
import warnings
import logging
import multiprocessing
import threading
import torch
import numpy as np
import scipy.spatial
import ase.db
import pandas as pd



def _cell_heights(cell_object):
    volume = cell_object.volume
    crossproducts = np.cross(cell_object[[1, 2, 0]], cell_object[[2, 0, 1]])
    crosslengths = np.sqrt(np.sum(np.square(crossproducts), axis=1))
    heights = volume / crosslengths
    return heights


class AseNeigborListWrapper:
    """
    Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist

    """

    def __init__(self, cutoff, atoms):
        self.neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert (
            cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)

        rel_positions = (
            self.atoms_positions[indices]
            + offsets @ self.atoms_cell
            - self.atoms_positions[i][None]
        )

        dist2 = np.sum(np.square(rel_positions), axis=1)

        return indices, rel_positions, dist2


class TransformRowToGraph:
    def __init__(self, cutoff=5.0, targets="U0"):
        self.cutoff = cutoff

        if isinstance(targets, str):
            self.targets = [targets]
        else:
            self.targets = targets

    def __call__(self, row):
        atoms = row.toatoms()

        if np.any(atoms.get_pbc()):
            edges, edges_features = self.get_edges_neighborlist(atoms)
        else:
            edges, edges_features = self.get_edges_simple(atoms)

        # Extract targets if they exist
        targets = []
        for target in self.targets:
            if hasattr(row, target):
                t = getattr(row, target)
            elif hasattr(row, "data") and target in row.data:
                t = row.data[target]
            else:
                t = np.nan
            targets.append(t)
        targets = np.array(targets)

        default_type = torch.get_default_dtype()

        # pylint: disable=E1102
        graph_data = {
            "nodes": torch.tensor(atoms.get_atomic_numbers()),
            "num_nodes": torch.tensor(len(atoms.get_atomic_numbers())),
            "edges": torch.tensor(edges),
            "edges_features": torch.tensor(edges_features, dtype=default_type),
            "num_edges": torch.tensor(edges.shape[0]),
            "targets": torch.tensor(targets, dtype=default_type),
        }

        return graph_data

    def get_edges_simple(self, atoms):
        # Compute distance matrix
        pos = atoms.get_positions()
        dist_mat = scipy.spatial.distance_matrix(pos, pos)

        # Build array with edges and edge features (distances)
        valid_indices_bool = dist_mat < self.cutoff
        np.fill_diagonal(valid_indices_bool, False)  # Remove self-loops
        edges = np.argwhere(valid_indices_bool)  # num_edges x 2
        edges_features = dist_mat[valid_indices_bool]  # num_edges
        edges_features = np.expand_dims(edges_features, 1)  # num_edges, 1

        return edges, edges_features

    def get_edges_neighborlist(self, atoms):
        edges = []
        edges_features = []

        # Compute neighborlist
        if (
            np.any(atoms.get_cell().lengths() <= 0.0001)
            or (
                np.any(atoms.get_pbc())
                and np.any(_cell_heights(atoms.get_cell()) < self.cutoff)
            )
            or ("asap3" not in sys.modules)
        ):
            neighborlist = AseNeigborListWrapper(self.cutoff, atoms)
        else:
            neighborlist = asap3.FullNeighborList(self.cutoff, atoms)

        for i in range(len(atoms)):
            neigh_idx, _, neigh_dist2 = neighborlist.get_neighbors(i, self.cutoff)
            neigh_dist = np.sqrt(neigh_dist2)

            self_index = np.ones_like(neigh_idx) * i
            this_edges = np.stack((neigh_idx, self_index), axis=1)

            edges.append(this_edges)
            edges_features.append(neigh_dist)

        return np.concatenate(edges), np.expand_dims(np.concatenate(edges_features), 1)


class TransformRowToGraphXyz:
    """
    Transform ASE DB row to graph while keeping the xyz positions of the vertices

    """

    def __init__(self, cutoff=5.0, energy_property="energy", forces_property="forces"):
        self.cutoff = cutoff
        self.energy_property = energy_property
        self.forces_property = forces_property

    def __call__(self, row):
        atoms = row.toatoms()

        if np.any(atoms.get_pbc()):
            atoms.wrap()  # Make sure all atoms are inside unit cell
            edges, edges_displacement = self.get_edges_neighborlist(atoms)
        else:
            edges, edges_displacement = self.get_edges_simple(atoms)

        # Extract energy and forces if they exists
        energy = np.array([0.0])
        try:
            energy = np.array([getattr(row, self.energy_property)])
        except AttributeError:
            pass
        try:
            energy = np.copy([np.squeeze(row.data[self.energy_property])])
        except (KeyError, AttributeError):
            pass
        forces = np.zeros((len(atoms), 3))
        try:
            forces = np.copy(getattr(row, self.forces_property))
        except AttributeError:
            pass
        try:
            forces = np.copy(row.data[self.forces_property])
        except (KeyError, AttributeError):
            pass

        default_type = torch.get_default_dtype()

        # pylint: disable=E1102
        graph_data = {
            "nodes": torch.tensor(atoms.get_atomic_numbers()),
            "nodes_xyz": torch.tensor(atoms.get_positions(), dtype=default_type),
            "num_nodes": torch.tensor(len(atoms.get_atomic_numbers())),
            "edges": torch.tensor(edges),
            "edges_displacement": torch.tensor(edges_displacement, dtype=default_type),
            "cell": torch.tensor(atoms.get_cell(), dtype=default_type),
            "num_edges": torch.tensor(edges.shape[0]),
            "energy": torch.tensor(energy, dtype=default_type),
            "forces": torch.tensor(forces, dtype=default_type),
        }

        return graph_data

    def get_edges_simple(self, atoms):
        # Compute distance matrix
        pos = atoms.get_positions()
        dist_mat = scipy.spatial.distance_matrix(pos, pos)

        # Build array with edges and edge features (distances)
        valid_indices_bool = dist_mat < self.cutoff
        np.fill_diagonal(valid_indices_bool, False)  # Remove self-loops
        edges = np.argwhere(valid_indices_bool)  # num_edges x 2
        edges_displacement = np.zeros((edges.shape[0], 3))

        return edges, edges_displacement

    def get_edges_neighborlist(self, atoms):
        edges = []
        edges_displacement = []
        atom_positions = atoms.get_positions()
        inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

        # Compute neighborlist
        if (
            np.any(atoms.get_cell().lengths() <= 0.0001)
            or (
                np.any(atoms.get_pbc())
                and np.any(_cell_heights(atoms.get_cell()) < self.cutoff)
            )
            or ("asap3" not in sys.modules)
        ):
            neighborlist = AseNeigborListWrapper(self.cutoff, atoms)
        else:
            neighborlist = asap3.FullNeighborList(self.cutoff, atoms)

        for i in range(len(atoms)):
            neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, self.cutoff)

            self_index = np.ones_like(neigh_idx) * i
            this_edges = np.stack((neigh_idx, self_index), axis=1)

            neigh_pos = atom_positions[neigh_idx]
            this_pos = atom_positions[i]
            neigh_origin = neigh_vec + this_pos - neigh_pos
            neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

            ############
            # assert np.allclose(neigh_pos + (neigh_origin_scaled @ atoms.get_cell()) - this_pos, neigh_vec)
            ############

            edges.append(this_edges)
            edges_displacement.append(neigh_origin_scaled)

        return np.concatenate(edges), np.concatenate(edges_displacement)



class TransformAtomsObjectsToGraphXyz:
    """
    Transform Atoms() to graph while keeping the xyz positions of the vertices

    """

    def __init__(self, cutoff=5.0):
        self.cutoff = cutoff

    def __call__(self, atoms):

        if np.any(atoms.get_pbc()):
            atoms.wrap()  # Make sure all atoms are inside unit cell
            edges, edges_displacement = self.get_edges_neighborlist(atoms)
        else:
            edges, edges_displacement = self.get_edges_simple(atoms)

        default_type = torch.get_default_dtype()

        # pylint: disable=E1102
        graph_data = {
            "nodes": torch.tensor(atoms.get_atomic_numbers()),
            "nodes_xyz": torch.tensor(atoms.get_positions(), dtype=default_type),
            "num_nodes": torch.tensor(len(atoms.get_atomic_numbers())),
            "edges": torch.tensor(edges),
            "edges_displacement": torch.tensor(edges_displacement, dtype=default_type),
            "cell": torch.tensor(np.array(atoms.get_cell()), dtype=default_type),
            "num_edges": torch.tensor(edges.shape[0])
        }

        return graph_data

    def get_edges_simple(self, atoms):
        # Compute distance matrix
        pos = atoms.get_positions()
        dist_mat = scipy.spatial.distance_matrix(pos, pos)

        # Build array with edges and edge features (distances)
        valid_indices_bool = dist_mat < self.cutoff
        np.fill_diagonal(valid_indices_bool, False)  # Remove self-loops
        edges = np.argwhere(valid_indices_bool)  # num_edges x 2
        edges_displacement = np.zeros((edges.shape[0], 3))

        return edges, edges_displacement

    def get_edges_neighborlist(self, atoms):
        edges = []
        edges_displacement = []
        atom_positions = atoms.get_positions()
        inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

        # Compute neighborlist
        if (
            np.any(atoms.get_cell().lengths() <= 0.0001)
            or (
                np.any(atoms.get_pbc())
                and np.any(_cell_heights(atoms.get_cell()) < self.cutoff)
            )
            or ("asap3" not in sys.modules)
        ):
            neighborlist = AseNeigborListWrapper(self.cutoff, atoms)
        else:
            neighborlist = asap3.FullNeighborList(self.cutoff, atoms)

        for i in range(len(atoms)):
            neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, self.cutoff)

            self_index = np.ones_like(neigh_idx) * i
            this_edges = np.stack((neigh_idx, self_index), axis=1)

            neigh_pos = atom_positions[neigh_idx]
            this_pos = atom_positions[i]
            neigh_origin = neigh_vec + this_pos - neigh_pos
            neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

            ############
            # assert np.allclose(neigh_pos + (neigh_origin_scaled @ atoms.get_cell()) - this_pos, neigh_vec)
            ############

            edges.append(this_edges)
            edges_displacement.append(neigh_origin_scaled)

        return np.concatenate(edges), np.concatenate(edges_displacement)


class AseDbData(torch.utils.data.Dataset):
    def __init__(self, asedb_path, transformer, **kwargs):
        super().__init__(**kwargs)

        self.asedb_path = asedb_path
        self.asedb_connection = ase.db.connect(asedb_path)
        self.transformer = transformer

    def __len__(self):
        return len(self.asedb_connection)

    def __getitem__(self, key):
        # Note that ASE databases are 1-indexed
        try:
            return self.transformer(self.asedb_connection[key + 1])
        except KeyError:
            raise IndexError("index out of range")


class QM9MetaGGAData(torch.utils.data.Dataset):
    """"""

    def __init__(self, qm9asedb_path, metaggaqm9csv_path, cutoff, **kwargs):
        super().__init__(**kwargs)

        self.asedb_connection = ase.db.connect(qm9asedb_path)
        self.metagga_df = pd.read_csv(metaggaqm9csv_path, index_col="index")
        self.metagga_df.drop(columns=["SOGGA", "SOGGA11"], inplace=True)
        self.transformer = TransformRowToGraph(cutoff=cutoff, targets=[])

    def __len__(self):
        return len(self.asedb_connection)

    def __getitem__(self, key):
        # Note that ASE databases are 1-indexed
        key = key + 1
        try:
            item = self.transformer(self.asedb_connection[key])
            targets = self.metagga_df.loc[key].values
            # pylint: disable=E1102
            item["targets"] = torch.tensor(targets, dtype=torch.float32)
            return item
        except KeyError:
            raise IndexError("index out of range")


class BufferData(torch.utils.data.Dataset):
    """
    Wrapper for a dataset. Loads all data into memory.
    """

    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)

        self.data_objects = [dataset[i] for i in range(len(dataset))]

    def __len__(self):
        return len(self.data_objects)

    def __getitem__(self, index):
        return self.data_objects[index]


def rotating_pool_worker(dataset, rng, queue):
    while True:
        for index in rng.permutation(len(dataset)):
            queue.put(dataset[index])


def transfer_thread(queue: multiprocessing.Queue, datalist: list):
    while True:
        for index in range(len(datalist)):
            datalist[index] = queue.get()


class RotatingPoolData(torch.utils.data.Dataset):
    """
    Wrapper for a dataset that continously loads data into a smaller pool.
    The data loading is performed in a separate process and is assumed to be IO bound.
    """

    def __init__(self, dataset, pool_size, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.parent_data = dataset
        self.rng = np.random.default_rng()
        logging.debug("Filling rotating data pool of size %d" % pool_size)
        self.data_pool = [
            self.parent_data[i]
            for i in self.rng.integers(
                0, high=len(self.parent_data), size=self.pool_size, endpoint=False
            )
        ]
        self.loader_queue = multiprocessing.Queue(2)

        # Start loaders
        self.loader_process = multiprocessing.Process(
            target=rotating_pool_worker,
            args=(self.parent_data, self.rng, self.loader_queue),
        )
        self.transfer_thread = threading.Thread(
            target=transfer_thread, args=(self.loader_queue, self.data_pool)
        )
        self.loader_process.start()
        self.transfer_thread.start()

    def __len__(self):
        return self.pool_size

    def __getitem__(self, index):
        return self.data_pool[index]


def pad_and_stack(tensors: List[torch.Tensor]):
    """ Pad list of tensors if tensors are arrays and stack if they are scalars """
    if tensors[0].shape:
        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=0
        )
    return torch.stack(tensors)


def collate_atomsdata(graphs: List[dict], pin_memory=True):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [dic[k] for dic in graphs] for k in graphs[0]}
    # Convert each list of tensors to single tensor with pad and stack
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x

    collated = {k: pin(pad_and_stack(dict_of_lists[k])) for k in dict_of_lists}
    return collated


class CollateAtomsdata:
    """Callable that applies the collate_atomsdata function."""

    def __init__(self, pin_memory):
        self.pin_memory = pin_memory

    def __call__(self, graphs: List[dict]):
        return collate_atomsdata(graphs, self.pin_memory)
