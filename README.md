# On the effectiveness of reinforcement learning in nanoparticle design
This is a modified and adapted version of the original Molgym framework from:

**Reinforcement Learning for Molecular Design Guided by Quantum Mechanics**<br>
Gregor N. C. Simm, Robert Pinsler and José Miguel Hernández-Lobato <br>
*Proceedings of the 37th International Conference on Machine Learning*, Vienna, Austria, PMLR 119, 2020.<br>
https://arxiv.org/abs/2002.07717

However, our framework has been adapted to tackle the additional challenges that are present for nanoparticles compared to molecules, most importantly:

1. Switch to a positive final reward instead of the original negative step-wise reward.

2. Switch to an equivariant message passing graph representation learning (PaiNN), in place of the original invariant (SchNet).

3. Changes that are non-specific to the nanoparticle building task are also included, which are inspired from best practices in on-policy RL. The implemented changes include the option for varying network widths and depths for the actor and critic, as well as Tanh activation functions. 
This group of changes is inspired by the arXiv preprint **What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study** by the Google Brain team, located at https://arxiv.org/abs/2006.05990.

In-depth reasoning for these changes are found in the manuscript.


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

## Usage

Using the framework, a reinforcement learning agent can design a nanoparticle given a set of metal atoms. 

1. To run a default 13 gold atom experiment, use the following command:

    ```
    python3 scripts/run.py --num_steps_per_iter=8192 --vf_coef=0.1 --model=$MODEL$ --ex_dir=$PATH_TO_DESTINATION_FOLDER$ --name=$NAME_OF_EXPERIMENT$ --actor_network_width=64 --critic_network_width=256 --actor_network_depth=1 --optimizer=adam --learning_rate=0.0003 --entropy_coef=0.03 --target_kl=0.005 --min_mean_distance=2.4 --max_mean_distance=3.1 --min_atomic_distance=2.4 --formulas=Au13 --canvas_size=13 --save_rollouts=eval --clip_ratio=0.25 --lam=1.0 --discount=1.0 --num_steps=10000000000 --eval_freq=50
    ```
    
    This will automatically generate an experimental directory according to PATH_TO_DESTINATION_FOLDER and NAME_OF_EXPERIMENT, and place the results in the     directory. 
    
    The parameter --model can be replaced with either "painn" (PaiNN) or "internal" (SchNet).
    
    Parameters for all experiments are specified in the appendix to the manuscript.
