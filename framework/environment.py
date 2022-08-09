import abc
import itertools
from typing import Tuple, List

import ase.formula
import gym
import numpy as np
from ase import Atoms, Atom

from framework.reward import InteractionReward
from framework.spaces import ActionSpace, ObservationSpace, ActionType, ObservationType, NULL_SYMBOL
from framework.tools import util


class AbstractMolecularEnvironment(gym.Env, abc.ABC):
    # Negative reward should be on the same order of magnitude as the positive ones.
    # Memory agent on QM9: mean 0.26, std 0.13, min -0.54, max 1.23 (negative reward indeed possible
    # but avoidable and probably due to PM6)

    def __init__(
        self,
        reward: InteractionReward,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        min_atomic_distance=2.4,  # Angstrom
        max_h_distance=2.0,  # Angstrom
        min_reward=-20,  # Hartree
    ):
        self.reward = reward
        self.observation_space = observation_space
        self.action_space = action_space

        self.random_state = np.random.RandomState()

        self.min_atomic_distance = min_atomic_distance
        self.max_h_distance = max_h_distance
        self.min_reward = min_reward

        self.current_atoms = Atoms()
        self.current_formula = ase.formula.Formula()
        self.current_num_bag_atoms = self.current_formula.__len__()

    @abc.abstractmethod
    def reset(self) -> ObservationType:
        raise NotImplementedError

    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, dict]:
        new_atom = self.action_space.to_atom(action)

        # This is how we determine if the iteration is done. If the NULL_SYMBOL is pulled
        # from the bag, then we know that there are no more atoms left to place on the canvas.
        # Remember that NULL_SYMBOL is always the innermost atom in our bag space, meaning
        # that it will always be pulled last, and can therefore serve to keep tabs of
        # whether we have emptied the bag.
        done = new_atom.symbol == NULL_SYMBOL

        if done:
            # Return an observation space that contains the atoms we have pulled.
            return self.observation_space.build(self.current_atoms, self.current_formula), 0.0, done, {}

#        if not self._is_valid_action(current_atoms=self.current_atoms, new_atom=new_atom):
            # Obtain minimum reward if the action is not valid
 #           return (
  #              self.observation_space.build(self.current_atoms, self.current_formula),
   #             self.min_reward,
    #            True,
     #           {},
      #      )

        reward, info = self.reward.calculate(self.current_atoms, new_atom, self.current_num_bag_atoms)

#        if reward < self.min_reward:
 #           done = True
  #          reward = self.min_reward

        # Add new_atom to list of canvas atoms
        self.current_atoms.append(new_atom)

        # Remove the new_atom from current bag and obtain the new bag
        self.current_formula = util.remove_from_formula(self.current_formula, new_atom.symbol)

        # Check if state is terminal
        if self._is_terminal():
            done = True

        # Each step returns the observation space, the reward and
        # info about the elapsed time and whether it was a terminal state
        return self.observation_space.build(self.current_atoms, self.current_formula), reward, done, info

    def _is_terminal(self) -> bool:
        # The state is terminal if the canvas space is filled or if the current_formula list (the bag) is empty
        return len(self.current_atoms) == self.observation_space.canvas_space.size or len(self.current_formula) == 0

    def _is_valid_action(self, current_atoms: Atoms, new_atom: Atom) -> bool:
        # If the new atom is too close to any of the canvas atoms (defined by a simple norm in the function below)
        # then the action is not valid, and the reward is minimized according to the if-statement in line 60 above
        if self._is_too_close(current_atoms, new_atom):
            return False

        return self._all_h_covered(current_atoms, new_atom)

    def _is_too_close(self, existing_atoms: Atoms, new_atom: Atom) -> bool:
        # Check distances between new and old atoms
        for existing_atom in existing_atoms:
            if np.linalg.norm(existing_atom.position - new_atom.position) < self.min_atomic_distance:
                return True

        return False

    def _all_h_covered(self, existing_atoms: Atoms, new_atom: Atom) -> bool:
        # Ensure that H atoms are not too far away from the nearest heavy atom
        if len(existing_atoms) == 0 or new_atom.symbol != 'H':
            return True

        for existing_atom in existing_atoms:
            # Check whether an atom is actually hydrogen
            if existing_atom.symbol == 'H':
                continue

            distance = np.linalg.norm(existing_atom.position - new_atom.position)
            # Then check if this particular hydrogen atom is not too far away
            # as defined by being below the max_h_distance
            if distance < self.max_h_distance:
                return True

        return False

    def render(self, mode='human'):
        pass

    def seed(self, seed=None) -> int:
        # Whether you want to use a specific seed or just a random one.
        seed = seed or np.random.randint(int(1e5))
        self.random_state = np.random.RandomState(seed)
        return seed


class MolecularEnvironment(AbstractMolecularEnvironment):
    def __init__(self, formulas: List[ase.formula.Formula], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.formulas = formulas
        self.formulas_cycle = itertools.cycle(formulas)
        self.reset()

    def reset(self) -> ObservationType:
        # The reset function apparently switches to another bag when called
        self.current_atoms = Atoms()
        self.current_formula = next(self.formulas_cycle)
        self.current_num_bag_atoms = self.current_formula.__len__()
        return self.observation_space.build(self.current_atoms, self.current_formula)
