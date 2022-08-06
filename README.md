# Reinforcement Learning for Discovering Nanoparticle Structure
This is a modified and adapted version of the original Molgym framework from:

**Reinforcement Learning for Molecular Design Guided by Quantum Mechanics**<br>
Gregor N. C. Simm, Robert Pinsler and José Miguel Hernández-Lobato <br>
*Proceedings of the 37th International Conference on Machine Learning*, Vienna, Austria, PMLR 119, 2020.<br>
https://arxiv.org/abs/2002.07717

However, our framework is different in several ways - both in terms of how reward feedback is provided, the graph representation framework and the actor-critic network architecture. These are designed to tackle the additional challenges that are present for nanoparticles compared to molecules. Firstly, physics-inspired changes include the switch to a positive final reward. This differs from the original negative step-wise reward, which is not suitable for the design of nanoparticles, since intermediate low-energy structures are unlikely to be easily incorporated into a final optimal structure. 
Furthermore, our reward function can easily be applied across nanoparticles of different size using the same hyperparameters, since it is based  on the potential energy per atom of the final nanoparticle configuration. A downside to this sparse reward scheme is that it requires value propagation throughout the decision, making it more technically challenging in terms of training and memory requirement.
Secondly, we moved from an invariant message passing graph representation learning (SchNet) to an equivariant one (PaiNN). The lack of well defined bond lengths and bond angles (unlike molecules) meant it is important for the agent to learn atom positioning decisions that are based on a representation that includes direction of bonds and not just distances. The PaiNN framework also includes invariant features, however.
The third category of improvements we did were non-specific to the nanoparticle building task and inspired from best practices in on-policy RL. The implemented changes include the option for varying network widths and depths for the actor and critic, as well as Tanh activation functions.


## Installation

Dependencies:
* Python  >= 3.6
* PyTorch >= 1.4
* Core dependencies specified in setup.py

# To set up the framework correctly:

1. Create new Python 3.6 environment:
   ```text
   virtualenv --python=python3.6 venv1
   source molgym-venv/bin/activate
   ```

2. Install required packages and library itself:
   ```text
   pip install -r molgym/requirements.txt
   pip install -e molgym/
   ```
## Reward Function
To change from the final to the stepwise reward function, unhash the 4 lines below in reward.py:
```text
       # e_tot = self._calculate_energy(all_atoms)
       # e_parts = self._calculate_energy(atoms) + self._calculate_atomic_energy(new_atom)
       # delta_e = e_tot - e_parts
       # reward = -1 * delta_e/27.21
```
And hash the following lines directly below:
```text
        if len(all_atoms.numbers) < num_atoms:
                reward= 0                
        else:
                e_tot = self._convert_ev_to_hartree(self._calculate_energy(all_atoms))
                reward = num_atoms/e_tot  # - baseline_weight * baseline_reward  
        elapsed = time.time() - start
```
## Usage

Using the framework, a reinforcement learning agent can design a nanoparticle given a set of metal atoms. 

1. To run a default 13 gold atom experiment, use the following command:

    ```
    python3 scripts/run.py --num_steps_per_iter=2048 --vf_coef=0.1 --model=$MODEL$ --ex_dir=$PATH_TO_DESTINATION_FOLDER$ --name=$NAME_OF_EXPERIMENT$ --actor_network_width=64 --critic_network_width=256 --actor_network_depth=1 --optimizer=adam --learning_rate=0.0003 --entropy_coef=0.03 --target_kl=0.005 --min_mean_distance=2.4 --max_mean_distance=3.1 --min_atomic_distance=2.4 --formulas=Au13 --canvas_size=13 --save_rollouts=eval --clip_ratio=0.25 --lam=1.0 --discount=1.0 --num_steps=10000000000 --eval_freq=50
    ```
    
    This will automatically generate an experimental directory according to PATH_TO_DESTINATION_FOLDER and NAME_OF_EXPERIMENT, and place the results in the     directory. 
    The parameter --model should be replaced with either "painn" (PaiNN) or "internal" (SchNet).
    Parameters for the experiments are specified in the appendix to the manuscript.
