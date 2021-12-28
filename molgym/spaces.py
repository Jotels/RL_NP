import sys
from typing import Tuple, List

import gym
import numpy as np
from ase import Atom, Atoms
from ase.data import chemical_symbols, atomic_numbers
from ase.formula import Formula

AtomicType = Tuple[int, Tuple[float, float, float]] #A Tuple containing information about an atom
MolecularType = Tuple[AtomicType, ...] #The molecule is a collection of atoms. The ellipsis object (...) means that the MolecularType
                                        # tuple can contain as many atoms as is desired (naturally, since we want to create large molecules
BagType = Tuple[int, ...]
ActionType = AtomicType #A big part of the action is choosing what atom to select as the next. Thus, the actiontype is also an atomictype,
                        # since when choosing an action we are also choosing what atom to put in.
ObservationType = Tuple[MolecularType, BagType] #The observation type refers to the current molecule canvas and remaining bag. Thus, it is part of the state

NULL_SYMBOL = 'X'


class AtomicSpace(gym.spaces.Tuple):
#Whenever we are inside the class, we call an instance of the object(a specific instance of class AtomicSpace)
#by using the "self" command
# From documentation on gym.spaces.Tuple: "A tuple (i.e., product) of simpler spaces". So the AtomicSpace is a combination of
# several different spaces. This is also evident since there is both gym.spaces.Discrete and gym.spaces.Box directly below
    def __init__(self) -> None:
        element = gym.spaces.Discrete(n=len(atomic_numbers))
        #A discrete space containing all elements. It is n-dimensional with n being the number of different atoms,
        # since atomic_numbers is a dictionary containing all elements and their corresponding atomic number, e.g. 'Cl': 17
        #So n = 118 as a default
        low = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        high = np.array([np.inf, np.inf, np.inf], dtype=float)
        #low and high are the dimensions of the physical space
        position = gym.spaces.Box(low=low, high=high, dtype=float)
        #From gym.spaces.Box documentation:
        #A (possibly unbounded) box in R^n. Specifically, a Box represents the Cartesian product of n closed intervals
        super().__init__((element, position))
        #Here we have used super() to call the __init__() of the gym.spaces.Tuple class,
        # allowing us to use it in the AtomicSpace class without repeating code

## THESE TWO METHODS BELOW CONVERT BETWEEN OUR DEFINED "ATOMICTYPE" AND THE ase.atom.Atom class in a convenient way
    @staticmethod
    def to_atom(sample: AtomicType) -> Atom:
        atomic_number, position = sample
        #to_atom takes an atomictype defined by an element and a position
        if atomic_number < 0:
            raise RuntimeError(f'Invalid atomic number: {atomic_number}')
        #Checks that it's a valid element
        return Atom(symbol=chemical_symbols[atomic_number], position=position)
        #and returns the atom as an Ase Atom

    @staticmethod
    def from_atom(atom: Atom) -> AtomicType:
        """
        Takes in an atom as defined in the ase environment
         and outputs it as our custom class "AtomicType"/"ActionType"
        """
        return int(atomic_numbers[atom.symbol]), tuple(atom.position)  # type: ignore


class MolecularSpace(gym.spaces.Tuple):
    # We create another space which holds different attributes, thus it must be a gym.space.Tuple (since this can hold more types)
    def __init__(self, size: int) -> None:
        self.size = size
        # Sets the size of the created instance of MolecularSpace to the argument
        super().__init__((AtomicSpace(), ) * self.size)

    ## BUILDS ON THE PREVIOUS FUNCTIONS THAT CHANGE BETWEEN THE ASE ATOM CLASS AND THE CUSTOM ATOMICSPACE CLASS
    @staticmethod
    def to_atoms(sample: MolecularType) -> Atoms:
        atoms = Atoms()
        for atomic_sample in sample:
            atom = AtomicSpace.to_atom(atomic_sample)
            if atom.symbol == NULL_SYMBOL:
                break
            atoms.append(atom)
        return atoms

    def from_atoms(self, atoms: Atoms) -> MolecularType:
        if len(atoms) > self.size:
            raise RuntimeError(f'Too many atoms: {len(atoms)} > {self.size}')

        elif len(atoms) < self.size:
            atoms = atoms.copy()

            dummy = Atom(symbol=NULL_SYMBOL, position=(0, 0, 0))
            while len(atoms) < self.size:
                atoms.append(dummy)

        return tuple(AtomicSpace.from_atom(atom) for atom in atoms)


ActionSpace = AtomicSpace
# Naturally, the ActionSpace is an AtomicSpace, since the AtomicSpace contains both the element and the position of the
# current atom under consideration. This completely defines the action - What atom to choose and where to place it.


class SymbolTable:
    def __init__(self, symbols: List[str]):

        # Ensure that all symbols are valid:
        if NULL_SYMBOL in symbols:
            raise RuntimeError(f'Place holder symbol {NULL_SYMBOL} cannot be in list of symbols')

        if len(symbols) < 1:
            raise RuntimeError('List of symbols cannot be empty')


        #Finds indices of all symbols in the "symbols" parameter passed to the SymbolTable
        #.. Apparently doesn't really do anything with it
        for symbol in symbols:
            chemical_symbols.index(symbol)

        # Ensure that there are no duplicates by checking whether the set of unique characters in "symbols"
        # is equal to the length of "symbols" itself.
        if len(set(symbols)) != len(symbols):
            raise RuntimeError(f'List of symbols {symbols} cannot contain duplicates')

        #create the _symbols list which is apparently the NULL_SYMBOL 'X' and then the passed list of symbols
        self._symbols = [NULL_SYMBOL] + symbols

    def get_index(self, symbol: str) -> int:
        #finds the index of a symbol from the _symbols list
        return self._symbols.index(symbol)

    def get_symbol(self, index: int) -> str:
        """Finds the symbol from an index in the _symbols list"""
        if index < 0:
            raise ValueError(f'Index ({index}) cannot be less than zero')
        return self._symbols[index]

    def count(self) -> int:
        #Length of the _symbols space
        return len(self._symbols)


#Create the bag:
class BagSpace(gym.spaces.Tuple):
    def __init__(self, symbols: List[str]):
        #We initialize our bagspace with a symbol_table according to what symbols we're currently passing in:
        self.zs = symbols
        self.symbol_table = SymbolTable(symbols)
        #The size of the bagspace is simply the amount of different elements in the "symbols" list we passed:
        self.size = self.symbol_table.count()
        #Inherits a big discrete space times the number of different elements.
        super().__init__((gym.spaces.Discrete(n=sys.maxsize), ) * self.size)

    def from_formula(self, formula: Formula) -> BagType:
        #BagType is just a tuple defined earlier. So this function inside the BagSpace creates the BagType from a chemical formula
        formula_dict = formula.count()#From ase documentation: "Return dictionary mapping chemical symbol to number of atoms"

        #This one is tricky, but bag is really just a long list of 0's
        bag = [0] * self.symbol_table.count()
        #Next, we look through our formula_dict for what elements we actually have, and make a sort of "one-hot" encoding,
        #but instead of being one-hot it is hot by  the amount of atoms of each element.
        for symbol, value in formula_dict.items():
            bag[self.symbol_table.get_index(symbol)] = value

        #the bag-type is thus this 118-element long list which encodes exactly how many atoms we have to place of each element:
        return tuple(bag)

    #This function takes a BagType  and accomplishes the opposite of the above. Takes a bag and finds  the formula.
    def to_formula(self, bag: 'BagType') -> Formula:
        if len(bag) != self.symbol_table.count():
            raise ValueError(f'Bag {bag} does not fit symbol table')

        d = {self.symbol_table.get_symbol(index): count for index, count in enumerate(bag)}
        return Formula.from_dict(d)

    #The functions below accomplishes the same as the previous get_symbol and get_index inside the other spaces.
    #Given an index or a symbol, it finds the corresponding symbol or index. Thus it is able to match indexes to symbols or reverse.
    def get_symbol(self, index: int) -> str:
        return self.symbol_table.get_symbol(index)

    def get_index(self, symbol: str) -> int:
        return self.symbol_table.get_index(symbol)


#The observationspace is intuitive from the below.
class ObservationSpace(gym.spaces.Tuple):
    def __init__(self, canvas_size: int, symbols: List[str]):
        # create a canvas which is a MolecularSpace of a given size and containing some molecule.
        self.canvas_space = MolecularSpace(size=canvas_size)
        # the observationspace also contains a BagSpace with some number of atoms.
        # Thus, the observationspace is what we pass into our policy and what it  has to make decisions from
        self.bag_space = BagSpace(symbols=symbols)
        super().__init__((self.canvas_space, self.bag_space))

    #Build the space from a given set of atoms and formula to return the canvas
    # and the bag in a way that  conforms to how we defined them above
    def build(self, atoms: Atoms, formula: Formula) -> ObservationType:
        return self.canvas_space.from_atoms(atoms), self.bag_space.from_formula(formula)

    def parse(self, observation: ObservationType) -> Tuple[Atoms, Formula]:
        """From an ObservationType with a canvas and a bag,
        return an ase class of atoms and formula"""

        return self.canvas_space.to_atoms(observation[0]), self.bag_space.to_formula(observation[1])
