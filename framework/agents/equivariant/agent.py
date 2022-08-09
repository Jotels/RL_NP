from typing import Tuple, List, Optional

import ase
import ase.data
import numpy as np
import schnetpack as spk
import torch
import torch.distributions
from torch import nn

from molgym.agents.base import AbstractActorCritic
from molgym.agents.internal import zmat
from molgym.modules import MLP, masked_softmax, to_one_hot
from molgym.spaces import ObservationSpace, ObservationType, ActionType, ActionSpace
from molgym.tools.util import to_numpy

from . import layer_painn as layer
from . import data_painn

class PainnAC(AbstractActorCritic):
    def __init__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        min_max_distance: Tuple[float, float],
        actor_network_width: int,
        critic_network_width: int,
        num_interactions: int,
        cutoff: float,
        device: torch.device,
    ):
        # Internal action: stop, focus, element, distance, angle, dihedral, kappa
        super().__init__(observation_space=observation_space, action_space=action_space, internal_action_dim=7)
        self.device = device

        self.num_atoms = self.observation_space.canvas_space.size
        self.num_zs = self.observation_space.bag_space.size

        self.num_afeats = actor_network_width // 2
        self.num_latent_beta = actor_network_width // 4
        self.num_latent = self.num_afeats + self.num_latent_beta

        # PaiNN variables:
        self.transformer = data_painn.TransformAtomsObjectsToGraphXyz(cutoff=cutoff)
        self.hidden_state_size = actor_network_width // 2
        if device==torch.device("cuda"):
            self.pin=True
        else:
            self.pin=False
        self.cutoff = cutoff
        self.distance_embedding_size = 20

        num_embeddings = 119  # atomic numbers + 1
        edge_size = self.distance_embedding_size

        # Setup atom embeddings
        self.atom_embeddings = nn.Embedding(num_embeddings, self.hidden_state_size)

        # Setup interaction networks
        self.interactions = nn.ModuleList(
            [
                layer.PaiNNInteraction(self.hidden_state_size, edge_size, self.cutoff)
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.ModuleList(
            [layer.PaiNNUpdate(self.hidden_state_size) for _ in range(num_interactions)]
        )


        self.converter = spk.AtomsConverter(device=self.device)
        self.embedding_fn = spk.representation.SchNet(n_atom_basis=self.num_afeats)

        # MolGym neural networks
        self.phi_beta = MLP(
            input_dim=self.num_zs,
            hidden_units=(actor_network_width, self.num_latent_beta),
        )

        self.phi_focus = MLP(
            input_dim=self.num_latent,
            hidden_units=(actor_network_width, 1),
        )

        self.phi_element = MLP(
            input_dim=self.num_latent,
            hidden_units=(actor_network_width, self.num_zs),
        )

        self.phi_continuous = MLP(
            input_dim=(self.num_latent + self.num_zs),
            hidden_units=(actor_network_width, 3),
        )

        self.phi_kappa = MLP(
            input_dim=self.num_latent,
            hidden_units=(actor_network_width, 1),
            gate=torch.nn.functional.relu,
        )

        self.log_stds = torch.nn.Parameter(torch.log(torch.tensor([0.15, 0.25, 0.25], dtype=torch.float32)),
                                           requires_grad=True)

        # Reparametrization of continuous variables: values are between [-1, 1] after tanh
        self.min_distance, self.max_distance = min_max_distance
        self.min_angle, self.max_angle = 0, np.pi
        self.min_dihedral, self.max_dihedral = 0, np.pi

        self.action_width = torch.tensor([
            self.max_distance - self.min_distance,
            self.max_angle - self.min_angle,
            self.max_dihedral - self.min_dihedral,
        ])  # (3, )
        self.action_center = 0.5 * torch.tensor([
            self.max_distance + self.min_distance,
            self.max_angle + self.min_angle,
            self.max_dihedral + self.min_dihedral,
        ])  # (3, )

        self.critic = MLP(
            input_dim=self.num_latent,
            hidden_units=(critic_network_width, critic_network_width, 1),
        )

        self.to(device)

    def to_action_space(self, action: np.ndarray, observation: ObservationType) -> ActionType:
        stop, focus, element, distance, angle, dihedral, kappa = action

        if stop:
            return self.action_space.build(ase.Atoms())

        # Round to obtain discrete subactions
        focus = int(round(focus))
        element = int(round(element))
        sign = -1 if int(round(kappa)) else 1

        atoms, bag = self.observation_space.parse(observation)
        positions = [atom.position for atom in atoms]
        position = zmat.position_atom_helper(positions=positions,
                                             focus=focus,
                                             distance=distance,
                                             angle=angle,
                                             dihedral=sign * dihedral)
        #atomic_number_index = self.action_space.zs.index(self.observation_space.bag_space.zs[element])
        #return atomic_number_index, tuple(position)
        new_atom = ase.Atom(symbol=self.observation_space.bag_space.get_symbol(element), position=position)
        # And represent it as an instance of the ActionSpace class:
        return self.action_space.from_atom(new_atom)

    def make_atomic_tensors(
        self, observations: List[ObservationType]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        features = torch.zeros(size=(len(observations), self.num_atoms, self.num_afeats),
                               dtype=torch.float32,
                               device=self.device)
        focus_mask = torch.zeros(size=(len(observations), self.num_atoms), dtype=torch.int, device=self.device)
        focus_mask_next = torch.zeros(size=(len(observations), self.num_atoms), dtype=torch.int, device=self.device)
        element_count = torch.zeros(size=(len(observations), self.num_zs), dtype=torch.float32, device=self.device)
        action_mask = torch.zeros(size=(len(observations), 6), dtype=torch.float32, device=self.device)

        for i, obs in enumerate(observations):
            # Get Atoms() object from observation
            atoms, formula = self.observation_space.parse(obs)
#            bag_tuple = [count for atomic_num, count in formula]
            bag_tuple = self.observation_space.bag_space.from_formula(formula)
            if len(atoms) > 0:
                # Transform to graph dictionary
                graph_state = [self.transformer(atoms)]
                batch_host = data_painn.collate_atomsdata(graph_state, pin_memory=self.pin)
                batch = {
                    k: v.to(device=self.device, non_blocking=True)
                    for (k, v) in batch_host.items()
                }
                # Get PaiNN embeddings for single observation
                nodes_scalar, _, _ = self._get_painn_embeddings(batch)
                # if len(atoms)>4:
                #     print("nodes_scalar : " + str(nodes_scalar.shape))
                #     print(nodes_scalar)
                #     exit()
                features[i, :len(atoms), :] = nodes_scalar # self.embedding_fn(self.converter(atoms))
                #features[i, :len(atoms), :] =  self.embedding_fn(self.converter(atoms))
                focus_mask[i, :len(atoms)] = 1
                focus_mask_next[i, :len(atoms) + 1] = 1
            else:
                focus_mask[i, :1] = 1  # focus null-atom
                focus_mask_next[i, :2] = 1

            element_count[i] = torch.tensor(bag_tuple, dtype=torch.float32, device=self.device)

            # Mask out subactions:
            action_mask[i, 0] = len(atoms) >= 1  # focus
            action_mask[i, 1] = 1.0  # element
            action_mask[i, 2] = len(atoms) >= 1  # distance
            action_mask[i, 3] = len(atoms) >= 2  # angle
            action_mask[i, 4] = len(atoms) >= 3  # dihedral
            action_mask[i, 5] = len(atoms) >= 3  # kappa

        return (
            features,  # n_obs x n_atoms x n_afeats
            focus_mask,  # n_obs x n_atoms
            focus_mask_next,  # n_obs x n_atoms
            element_count,  # n_obs x n_zs
            action_mask,  # n_obs x n_actions
        )

    def surrogate_features(self, observations: List[ObservationType], focus: torch.Tensor, element: torch.Tensor,
                           distance: torch.Tensor, angle: torch.Tensor, dihedral: torch.Tensor) -> torch.Tensor:

        features = torch.zeros(size=(len(observations), self.num_afeats), dtype=torch.float32, device=self.device)
        focus = to_numpy(focus)
        element = to_numpy(element)
        distance = to_numpy(distance)
        angle = to_numpy(angle)
        dihedral = to_numpy(dihedral)

        for i, observation in enumerate(observations):
            atoms, bag = self.observation_space.parse(observation)
            positions = [atom.position for atom in atoms]
            new_position = zmat.position_atom_helper(
                positions=positions,
                focus=int(round(focus[i, 0])),
                distance=distance[i, 0],
                angle=angle[i, 0],
                dihedral=dihedral[i, 0],
            )
            new_element = int(round(element[i, 0]))
#            atomic_number = bag[new_element][0]
#            new_atom = ase.Atom(symbol=ase.data.chemical_symbols[atomic_number], position=new_position)
            new_atom = ase.Atom(symbol=self.observation_space.bag_space.get_symbol(new_element), position=new_position)
            atoms.append(new_atom)
            graph_state = [self.transformer(atoms)]
            batch_host = data_painn.collate_atomsdata(graph_state, pin_memory=self.pin)
            batch = {
                k: v.to(device=self.device, non_blocking=True)
                for (k, v) in batch_host.items()
            }
            nodes_scalar, _, _ = self._get_painn_embeddings(batch)
            features[i] = nodes_scalar[-1, :]

        return features


    def _get_painn_embeddings(self, input_dict: dict) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["edges_displacement"], input_dict["num_edges"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_edges"])

        # Unpad and concatenate all nodes into batch (0th) dimension
        nodes_xyz = layer.unpad_and_cat(
            input_dict["nodes_xyz"], input_dict["num_nodes"]
        )
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes_scalar = self.atom_embeddings(nodes_scalar)
        nodes_vector = torch.zeros(
            (nodes_scalar.shape[0], 3, self.hidden_state_size),
            dtype=nodes_scalar.dtype,
            device=nodes_scalar.device,
        )

        # Compute edge distances
        edges_distance, edges_diff = layer.calc_distance(
            nodes_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_edges"],
            return_diff=True,
        )

        # Expand edge features in Gaussian basis
        edge_state = layer.sinc_expansion(
            edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        # Apply interaction layers
        for int_layer, update_layer in zip(
            self.interactions, self.scalar_vector_update
        ):
            nodes_scalar, nodes_vector = int_layer(
                nodes_scalar,
                nodes_vector,
                edge_state,
                edges_diff,
                edges_distance,
                edges,
            )
            nodes_scalar, nodes_vector = update_layer(nodes_scalar, nodes_vector)

        return nodes_scalar, nodes_vector, edge_offset


    def step(self, observations: List[ObservationType], actions: Optional[np.ndarray] = None) -> dict:
        # atomic_feats: n_obs x n_atoms x n_afeats
        # focus_mask: n_obs x n_atoms
        # focus_mask: n_obs x n_atoms
        # element_count: n_obs x n_zs

        atomic_feats, focus_mask, focus_mask_next, element_count, action_mask = self.make_atomic_tensors(observations)
        element_mask = (element_count > 0).int()

        #print("atomic_feats shape: " + str(atomic_feats.shape))
        #print(atomic_feats)
        #exit()
        # stop: this agent does not stop
        stop = torch.zeros(size=(len(observations), 1), dtype=torch.float, device=self.device)

        # latent states bag
        latent_bag = self.phi_beta(element_count)

        # latent representation of atoms and bag
        latent_bag_tiled = latent_bag.unsqueeze(1)  # n_obs x 1 x n_zs
        latent_bag_tiled = latent_bag_tiled.expand(-1, self.num_atoms, -1)  # n_obs x n_atoms x n_zs

        latent_states = torch.cat([atomic_feats, latent_bag_tiled], dim=-1)  # n_obs x n_atoms x (n_afeats + n_zs)

        # Focus
        focus_logits = self.phi_focus(latent_states)  # n_obs x n_atoms x 1
        focus_logits = focus_logits.squeeze(-1)  # n_obs x n_atoms

        focus_p = masked_softmax(focus_logits, mask=focus_mask.bool())  # n_obs x n_atoms
        focus_dist = torch.distributions.Categorical(probs=focus_p)

        # Cast action to Tensor
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device)

        # focus: n_obs x 1
        if actions is not None:
            focus = torch.round(actions[:, 1:2]).long()
        elif self.training:
            focus = focus_dist.sample().unsqueeze(-1)
        else:
            focus = torch.argmax(focus_p, dim=-1).unsqueeze(-1)

        focus_oh = to_one_hot(focus, num_classes=self.num_atoms, device=self.device)  # n_obs x n_atoms

        # Focused atom is a hard (one-hot) selection over atoms
        focused_atom = (latent_states.transpose(1, 2) @ focus_oh[:, :, None]).squeeze(-1)  # n_obs x n_latent

        # Element
        element_logits = self.phi_element(focused_atom)  # n_obs x n_zs
        element_p = masked_softmax(element_logits, mask=element_mask.bool())  # n_obs x n_zs
        element_dist = torch.distributions.Categorical(probs=element_p)

        # element: n_obs x 1
        if actions is not None:
            element = torch.round(actions[:, 2:3]).long()
        elif self.training:
            element = element_dist.sample().unsqueeze(-1)
        else:
            element = torch.argmax(element_p, dim=-1).unsqueeze(-1)
        element_oh = to_one_hot(element, self.num_zs, device=self.device)  # n_obs x n_zs

        # Continuous variables
        # f: n_obs x (n_latent + n_zs)
        f = torch.cat([focused_atom, element_oh], dim=-1)
        distance_mean, angle_mean, dihedral_mean = torch.split(torch.tanh(self.phi_continuous(f)), 1, dim=-1)

        # Distance
        distance_mean = torch.nan_to_num((distance_mean * self.action_width[0] / 2) + self.action_center[0])
        distance_dist = torch.distributions.Normal(loc=distance_mean, scale=torch.exp(1e-6 + self.log_stds[0]))

        # distance: n_obs x 1
        if actions is not None:
            distance = actions[:, 3:4]
        elif self.training:
            # Ensure that the sampled distance is > 0
            distance = distance_dist.sample().clamp(0.001)
        else:
            distance = distance_mean

        # Angle
        angle_mean = torch.nan_to_num((angle_mean * self.action_width[1] / 2) + self.action_center[1])
        angle_dist = torch.distributions.Normal(loc=angle_mean, scale=torch.exp(1e-6 + self.log_stds[1]))

        # angle: n_obs x 1
        if actions is not None:
            angle = actions[:, 4:5]
        elif self.training:
            angle = angle_dist.sample()
        else:
            angle = angle_mean

        # Dihedral
        dihedral_mean = torch.nan_to_num((dihedral_mean * self.action_width[2] / 2) + self.action_center[2])
        dihedral_dist = torch.distributions.Normal(loc=dihedral_mean, scale=torch.exp(1e-6 + self.log_stds[2]))

        # dihedral: n_obs x 1
        if actions is not None:
            dihedral = actions[:, 5:6]
        elif self.training:
            dihedral = dihedral_dist.sample()
        else:
            dihedral = dihedral_mean

        # Kappa: 0 = keep, 1 = flip
        # surrogate_features: n_obs x n_afeats
        element_count_next = element_count - element_oh
        latent_bag_next = self.phi_beta(element_count_next)

        atomic_feats_next_0 = self.surrogate_features(observations, focus, element, distance, angle, dihedral)
        atomic_feats_next_1 = self.surrogate_features(observations, focus, element, distance, angle, -1 * dihedral)

        v0 = self.phi_kappa(torch.cat([atomic_feats_next_0, latent_bag_next], dim=-1))
        v1 = self.phi_kappa(torch.cat([atomic_feats_next_1, latent_bag_next], dim=-1))

        kappa_logits = torch.cat([v0, v1], dim=-1)
        kappa_dist = torch.distributions.Categorical(logits=kappa_logits)

        # kappa: n_obs x 1
        if actions is not None:
            kappa = torch.round(actions[:, 6:7])
        elif self.training:
            kappa = kappa_dist.sample().unsqueeze(-1)
        else:
            kappa = torch.argmax(kappa_logits, dim=-1).unsqueeze(-1)

        if actions is None:
            actions = torch.cat(
                [stop, focus.float(), element.float(), distance, angle, dihedral,
                 kappa.float()], dim=-1)

        # Critic
        weights = focus_mask.unsqueeze(-1).float()  # n_obs x n_atoms x 1
        weights = weights.transpose(1, 2)  # n_obs x 1 x n_atoms
        sum_atomic_feats = (weights @ atomic_feats).squeeze(1)  # n_obs x n_afeats
        # mean_atomic_feats = sum_atomic_feats / torch.sum(focus_mask, dim=-1, keepdim=True)
        v = self.critic(torch.cat([sum_atomic_feats, latent_bag], dim=-1))

        # Log probabilities
        log_prob_list = [
            focus_dist.log_prob(focus.squeeze(-1)).unsqueeze(-1),
            element_dist.log_prob(element.squeeze(-1)).unsqueeze(-1),
            distance_dist.log_prob(distance),
            angle_dist.log_prob(angle),
            dihedral_dist.log_prob(dihedral),
            kappa_dist.log_prob(kappa.squeeze(-1)).unsqueeze(-1),
        ]
        log_prob = torch.cat(log_prob_list, dim=-1)

        # Mask
        log_prob = log_prob * action_mask

        # Entropies
        entropy_list = [
            focus_dist.entropy().unsqueeze(-1),
            element_dist.entropy().unsqueeze(-1),
            distance_dist.entropy(),
            angle_dist.entropy(),
            dihedral_dist.entropy(),
            kappa_dist.entropy().unsqueeze(-1),
        ]
        entropy = torch.cat(entropy_list, dim=-1)

        # Mask
        entropy = entropy * action_mask

        return {
            'a': actions,  # n_obs x n_subactions
            'logp': log_prob.sum(dim=-1, keepdim=False),  # n_obs
            'ent': entropy[:, 0:2].sum(dim=-1, keepdim=False),  # n_obs
            'v': v.squeeze(-1),  # n_obs

#            # Actions in action space
#            'actions': [self.to_action_space(a, o) for a, o in zip(actions, observations)],
            # Other
            'entropies': entropy,  # n_obs x n_entropies
        }
