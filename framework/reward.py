import abc
import time
from typing import Tuple, Dict
import ase.data
from ase import Atoms, Atom
from ase import geometry
import numpy as np
from ase.calculators.emt import EMT

current_calc=EMT()

class MolecularReward(abc.ABC):
    """
    Define the MolecularReward function as an abstract class
    Any future inheritances from MolecularReward will have to implement the calculate method
    """
    @abc.abstractmethod
    def calculate(self, atoms: Atoms, new_atom: Atom, num_atoms: int) -> Tuple[float, dict]:
        raise NotImplementedError

    @staticmethod
    def get_minimum_spin_multiplicity(atoms) -> int:
        return sum(ase.data.atomic_numbers[atom.symbol] for atom in atoms) % 2 + 1


class InteractionReward(MolecularReward):
    def __init__(self,
                 distance_penalty: float,
                 canvas_size: int,
                 reward_coef: float) -> None:

        self.calculator = current_calc
        self.distance_penalty = distance_penalty
        self.reward_coef = reward_coef
#        self.num_atoms = canvas_size 
        self.settings = {
            'molecular_charge': 0,
            'max_scf_iterations': 128,
            'unrestricted_calculation': 1,
        }

        self.atom_energies: Dict[str, float] = {}
    # Implement the calculate method that is inherited from the superclass:

    def calculate(self, atoms: Atoms, new_atom: Atom, num_atoms: int) -> Tuple[float, dict]:
        """
        Does the energy calculation using the set calculator
        and also returns the elapsed time upon completion
        """
        start = time.time()

        all_atoms = atoms.copy()

        all_atoms.append(new_atom)
      


        if len(all_atoms.numbers) < num_atoms:
                reward= 0                
        else:
                e_tot = self._convert_ev_to_hartree(self._calculate_energy(all_atoms))
                reward = num_atoms/e_tot  
        elapsed = time.time() - start


        info = {
            'elapsed_time': elapsed,
        }

        return reward, info

    def _calculate_atomic_energy(self, atom: Atom) -> float:
        if atom.symbol not in self.atom_energies:
            atoms = Atoms()
            atoms.append(atom)
            self.atom_energies[atom.symbol] = self._calculate_energy(atoms)
        return self.atom_energies[atom.symbol]

    def _calculate_energy(self, atoms: Atoms) -> float:
        if len(atoms) == 0:
            return 0.0
        
        
        atoms.calc = self.calculator


        return atoms.get_potential_energy()

    def _convert_ev_to_hartree(self, energy):
        return energy/27.2107
