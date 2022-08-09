import abc
from typing import List, Optional

import numpy as np
import torch.distributions

from framework.spaces import ObservationType, ActionType, ObservationSpace, ActionSpace


class AbstractActorCritic(torch.nn.Module, abc.ABC):
    # The base actor critic. It inherits attributes from both the torch nn module and the abstract base classes.
    # Thus, if an instance of the abstract actor critic does not contain a given attribute, it inherits them from
    # these two classes. It has to be an abstract class because the two methods "step" and "to_action_space" are not
    # defined in the way we want them to be yet. Therefore, whenever we make a new class which inherits from the
    # AbstractActorCritic class, we will have to define the two abstract methods.
    # Thus, "abstract" refers to the fact  that it is not defined yet, but is just an abstraction until we fill out what
    # we actually want the class to contain when we later inherit properties from AbstractActorCritic in new classes.
    # FUTURE CLASSES INHERITING FROM AbstractActorCritic MUST IMPLEMENT THE ABSTRACT METHODS.
    def __init__(self, observation_space: ObservationSpace, action_space: ActionSpace, internal_action_dim: int):
        super().__init__()
        #Define internal action dimensions and observation/action spaces for the actorcritic to work on.
        self.internal_action_dim = internal_action_dim
        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def step(self, observations: List[ObservationType], action: Optional[np.ndarray] = None) -> dict:
        # Whenever we subclass AbstractActorCritic, we will have an error if we do not give the abstractmethods an
        # interpretable definition. Thus, we raise a NotImplementedError, meaning that the abstract method
        # is not yet implemented.
        raise NotImplementedError

    @abc.abstractmethod
    def to_action_space(self, action: np.ndarray, observation: ObservationType) -> ActionType:
        raise NotImplementedError
