from typing import Tuple, List, Optional

import ase
import numpy as np
import schnetpack as spk
import torch
import torch.distributions

from framework.agents.base import AbstractActorCritic
from framework.modules import MLP, masked_softmax, to_one_hot
from framework.spaces import ObservationSpace, ObservationType, ActionType, ActionSpace
from framework.tools import zmat # Imports functions for calculating distance, positions, angles and dihedral angles
from framework.tools.util import to_numpy # Imports a function that converts a torch tensor to a numpy.ndarray


class InternalAC(AbstractActorCritic):
    # Inherit the abstract AC class defined in the base script
    def __init__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        min_max_distance: Tuple[float, float],
        actor_network_width: int,
        critic_network_width: int,
        actor_depth: int,
        device: torch.device,
    ):
        # Internal action: stop, focus, element, distance, angle, dihedral, kappa
        # First inherit properties of the superclass (AbstractActorCritic)
        super().__init__(observation_space=observation_space, action_space=action_space, internal_action_dim=7)
        self.device = device

        # Initialize canvas and bag
        self.num_atoms = self.observation_space.canvas_space.size
        self.num_zs = self.observation_space.bag_space.size

        # Not entirely sure about these 3 features
        self.num_afeats = actor_network_width //2
        self.num_latent_beta = actor_network_width // 4
        self.num_latent = self.num_afeats + self.num_latent_beta

        # Schnet attributes

        #Convert ASE atoms object to an input suitable for the SchNetPack ML models.
        self.converter = spk.AtomsConverter(device=self.device)
        self.embedding_fn = spk.representation.SchNet(
			n_atom_basis=self.num_afeats,
			cutoff=20.0
			)

        # Define neural networks. See Figure 4 in paper for definitions
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

        # kappa is the sign of the dihedral angle
        self.phi_kappa = MLP(
            input_dim=self.num_latent,
            hidden_units=(actor_network_width, 1),
        )
        if actor_depth ==2:
              self.phi_beta = MLP(
              input_dim=self.num_zs,
              hidden_units=(actor_network_width, actor_network_width,self.num_latent_beta),
              )

              self.phi_focus = MLP(
              input_dim=self.num_latent,
              hidden_units=(actor_network_width, actor_network_width, 1),
              )

              self.phi_continuous = MLP(
              input_dim=(self.num_latent + self.num_zs),
              hidden_units=(actor_network_width, actor_network_width, 3),
              )
        self.log_stds = torch.nn.Parameter(torch.log(torch.tensor([0.15, 0.25, 0.25], dtype=torch.float32)),
                                           requires_grad=True)

        # Reparametrization of continuous variables: values are between [-1, 1] after tanh
        self.min_distance, self.max_distance = min_max_distance
        self.min_angle, self.max_angle = 0, np.pi
        self.min_dihedral, self.max_dihedral = 0, np.pi


        # Defines the space wherein an action can be performed
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

        # Define the critic neural network
        self.critic = MLP(
            input_dim=self.num_latent,
            hidden_units=(critic_network_width, critic_network_width, 1),
        )

        self.to(device)

    def to_action_space(self, action: np.ndarray, observation: ObservationType) -> ActionType:

        """
        Given an observation and an action consisting of distances, angles and focal element,
        returns the action space of an instance of the ase.Atom class called new atom
        So this creates our action space for a new atom to be placed from simple input forms
        :param action: np.ndarray consisting of stop, focus, element, distance, angle, dihedral, kappa
        :param observation: An ObservationType
        """

        assert action.shape[0] == self.internal_action_dim
        stop, focus, element, distance, angle, dihedral, kappa = action

        if stop:
            return self.action_space.build(ase.Atoms())

        # Round to obtain discrete subactions
        focus = int(round(focus))
        element = int(round(element))
        sign = -1 if int(round(kappa)) else 1

        atoms, _ = self.observation_space.parse(observation)
        positions = [atom.position for atom in atoms]
        new_position = zmat.position_atom_helper(positions=positions,
                                                 focus=focus,
                                                 distance=distance,
                                                 angle=angle,
                                                 dihedral=sign * dihedral)
        # Create a new ase.Atom() instance below:
        new_atom = ase.Atom(symbol=self.observation_space.bag_space.get_symbol(element), position=new_position)
        # And represent it as an instance of the ActionSpace class:
        return self.action_space.from_atom(new_atom)

    def make_atomic_tensors(
        self, observations: List[ObservationType]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        From a list of ObservationTypes containing MolecularTypes and BagTypes,
        construct features and masks ready for input to the neural nets
        """
        features = torch.zeros(size=(len(observations), self.num_atoms, self.num_afeats),
                               dtype=torch.float32,
                               device=self.device)
        focus_mask = torch.zeros(size=(len(observations), self.num_atoms), dtype=torch.int, device=self.device)
        focus_mask_next = torch.zeros(size=(len(observations), self.num_atoms), dtype=torch.int, device=self.device)
        element_count = torch.zeros(size=(len(observations), self.num_zs), dtype=torch.float32, device=self.device)
        action_mask = torch.zeros(size=(len(observations), 6), dtype=torch.float32, device=self.device)

        for i, observation in enumerate(observations):
            atoms, bag = self.observation_space.parse(observation)
            # Get the BagSpace
            bag_tuple = self.observation_space.bag_space.from_formula(bag)
            if len(atoms) > 0:
                features[i, :len(atoms), :] = self.embedding_fn(self.converter(atoms))
                focus_mask[i, :len(atoms)] = 1
                focus_mask_next[i, :len(atoms) + 1] = 1
            else:
                focus_mask[i, :1] = 1  # focus null-atom
                focus_mask_next[i, :2] = 1

            element_count[i] = torch.tensor(bag_tuple, dtype=torch.float32, device=self.device)

            # Mask out subactions:
            # This is done according to whether these quantities can be calculated.
            # If for example there are less than two atoms, there is no angle to calculate.
            # Similarly, less than 3 atoms means that neither the dihedral nor the kappa value can be calculated.
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
            atoms, _ = self.observation_space.parse(observation)
            positions = [atom.position for atom in atoms]
            new_position = zmat.position_atom_helper(
                positions=positions,
                focus=int(round(focus[i, 0])),
                distance=distance[i, 0],
                angle=angle[i, 0],
                dihedral=dihedral[i, 0],
            )
            new_element = int(round(element[i, 0]))
            new_atom = ase.Atom(symbol=self.observation_space.bag_space.get_symbol(new_element), position=new_position)
            atoms.append(new_atom)
            features[i] = self.embedding_fn(self.converter(atoms))[:, -1, :]

        return features

    def step(self, observations: List[ObservationType], action: Optional[np.ndarray] = None) -> dict:
        """
        This function defines each step in the RL algorithm
        """
        # atomic_feats: n_obs x n_atoms x n_afeats
        # focus_mask: n_obs x n_atoms
        # focus_mask: n_obs x n_atoms
        # element_count: n_obs x n_zs

        atomic_feats, focus_mask, focus_mask_next, element_count, action_mask = self.make_atomic_tensors(observations)
        element_mask = (element_count > 0).int()

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

        focus_p = masked_softmax(focus_logits, mask=focus_mask)  # n_obs x n_atoms
        focus_dist = torch.distributions.Categorical(probs=focus_p)

        # Cast action to Tensor
        if action is not None:
            action = torch.as_tensor(action, device=self.device)

        # focus: n_obs x 1
        if action is not None:
            focus = torch.round(action[:, 1:2]).long()
        elif self.training:
            focus = focus_dist.sample().unsqueeze(-1)
        else:
            focus = torch.argmax(focus_p, dim=-1).unsqueeze(-1)

        focus_oh = to_one_hot(focus, num_classes=self.num_atoms)  # n_obs x n_atoms

        # Focused atom is a hard (one-hot) selection over atoms
        focused_atom = (latent_states.transpose(1, 2) @ focus_oh[:, :, None]).squeeze(-1)  # n_obs x n_latent

        # Element
        element_logits = self.phi_element(focused_atom)  # n_obs x n_zs
        element_p = masked_softmax(element_logits, mask=element_mask)  # n_obs x n_zs
        element_dist = torch.distributions.Categorical(probs=element_p)

        # element: n_obs x 1
        if action is not None:
            element = torch.round(action[:, 2:3]).long()
        elif self.training:
            element = element_dist.sample().unsqueeze(-1)
        else:
            element = torch.argmax(element_p, dim=-1).unsqueeze(-1)

        element_oh = to_one_hot(element, self.num_zs)  # n_obs x n_zs

        # Continuous variables
        # f: n_obs x (n_latent + n_zs)
        f = torch.cat([focused_atom, element_oh], dim=-1)
        distance_mean, angle_mean, dihedral_mean = torch.split(torch.tanh(self.phi_continuous(f)), 1, dim=-1)

        # Distance
        distance_mean = (distance_mean * self.action_width[0] / 2) + self.action_center[0]
        distance_dist = torch.distributions.Normal(loc=distance_mean, scale=torch.exp(1e-6 + self.log_stds[0]))

        # distance: n_obs x 1
        if action is not None:
            distance = action[:, 3:4]
        elif self.training:
            # Ensure that the sampled distance is > 0
            distance = distance_dist.sample().clamp(0.001)
        else:
            distance = distance_mean

        # Angle
        angle_mean = (angle_mean * self.action_width[1] / 2) + self.action_center[1]
        angle_dist = torch.distributions.Normal(loc=angle_mean, scale=torch.exp(1e-6 + self.log_stds[1]))

        # angle: n_obs x 1
        if action is not None:
            angle = action[:, 4:5]
        elif self.training:
            angle = angle_dist.sample()
        else:
            angle = angle_mean

        # Dihedral
        dihedral_mean = (dihedral_mean * self.action_width[2] / 2) + self.action_center[2]
        dihedral_dist = torch.distributions.Normal(loc=dihedral_mean, scale=torch.exp(1e-6 + self.log_stds[2]))

        # dihedral: n_obs x 1
        if action is not None:
            dihedral = action[:, 5:6]
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
        if action is not None:
            kappa = torch.round(action[:, 6:7])
        elif self.training:
            kappa = kappa_dist.sample().unsqueeze(-1)
        else:
            kappa = torch.argmax(kappa_logits, dim=-1).unsqueeze(-1)

        if action is None:
            action = torch.cat([stop, focus.float(), element.float(), distance, angle, dihedral, kappa.float()], dim=-1)

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
            'a': action,  # n_obs x n_subactions
            'logp': log_prob.sum(dim=-1, keepdim=False),  # n_obs
            'ent': entropy[:, 0:2].sum(dim=-1, keepdim=False),  # n_obs
            'v': v.squeeze(-1),  # n_obs

            # Other
            'entropies': entropy,  # n_obs x n_entropies
        }
