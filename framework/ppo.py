# The content of this file is based on: DeepRL https://github.com/ShangtongZhang/DeepRL.
# PROXIMAL POLICY OPTIMIZATION
# Used to learn the weights of the Actor-Critic Neural networks
import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim import Adam

from framework.agents.base import AbstractActorCritic
from framework.buffer import PPOBuffer
from ase.io import write
from framework.environment import AbstractMolecularEnvironment
from framework.tools.mpi import mpi_avg, mpi_avg_grads, get_num_procs, mpi_sum, mpi_mean_std
from framework.tools.util import RolloutSaver, to_numpy, ModelIO, InfoSaver


def compute_loss(ac: AbstractActorCritic, data: dict, clip_ratio: float, vf_coef: float,
                 entropy_coef: float) -> Tuple[torch.Tensor, dict]:
    pred = ac.step(data['obs'], data['act']) # Make a prediction step using the actor critic model

    old_logp = torch.as_tensor(data['logp']) # Get the logp from previous step
    adv = torch.as_tensor(data['adv']) # Get advantage function
    ret = torch.as_tensor(data['ret']) # Get return

    # Policy loss
    ratio = torch.exp(pred['logp'] - old_logp) # This is the same as saying p/old_p
    obj = ratio * adv # Objective defined as the p-ratio times advantage
    clipped_obj = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv # Clipped objective as defined in paper
    policy_loss = -torch.min(obj, clipped_obj).mean() # The PPO policy loss is then the smallest between the real objective
                                                      # and the clipped objective

    # Entropy loss
    entropy_loss = -entropy_coef * pred['ent'].mean() # pred['ent'] is predicted entropy. The entropy_coef is a scaling factor

    # Value loss
    vf_loss = vf_coef * (pred['v'] - ret).pow(2).mean() # Simple quadratic loss function

    # Total loss
    loss = policy_loss + entropy_loss + vf_loss

    # Approximate KL for early stopping
    approx_kl = (old_logp - pred['logp']).mean()  # KL divergence standard formula

    # Extra info
    clipped = ratio.lt(1 - clip_ratio) | ratio.gt(1 + clip_ratio)
    clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean()

    info = dict( # Provides info about the full step, for printing
        policy_loss=mpi_avg(to_numpy(policy_loss)).item(),
        entropy_loss=mpi_avg(to_numpy(entropy_loss)).item(),
        vf_loss=mpi_avg(to_numpy(vf_loss)).item(),
        total_loss=mpi_avg(to_numpy(loss)).item(),
        approx_kl=mpi_avg(to_numpy(approx_kl)).item(),
        clip_fraction=mpi_avg(to_numpy(clip_fraction)).item(),
    )

    return loss, info


# Train policy with multiple steps of gradient descent
def train(ac: AbstractActorCritic, optimizer: Adam, data: Dict[str, torch.Tensor], clip_ratio: float, target_kl: float,
          vf_coef: float, entropy_coef: float, gradient_clip: float, max_num_steps: int) -> dict:
    infos = {}

    start_time = time.time()

    i = -1
    for i in range(max_num_steps):
        # Compute loss
        loss, loss_info = compute_loss(ac, data, clip_ratio=clip_ratio, vf_coef=vf_coef, entropy_coef=entropy_coef)

        # Check KL
        if loss_info['approx_kl'] > 1.5 * target_kl: # Means we stray  too far from a previous policy, thus violating PPO
                                                     # Note that we use the approx_kl from above
            logging.info(f'Early stopping at step {i} due to reaching max KL.')
            break

        # Take gradient step
        optimizer.zero_grad()
        loss.backward()
        mpi_avg_grads(ac)  # average grads across MPI processes
        # Clip gradients, just to be sure
        torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=gradient_clip)
        optimizer.step()

        # Logging
        logging.debug(f'Optimization step {i}: {loss_info}')
        infos.update(loss_info)

    infos['num_opt_steps'] = i + 1
    infos['time'] = time.time() - start_time
    return infos


def rollout(ac: AbstractActorCritic,
            env: AbstractMolecularEnvironment,
            buffer: PPOBuffer,
            vis_dir: str, # Directory in which the rollout visualizations are placed
            iteration_ctr: Optional[int]=None,
            struc_vis_save: bool = False,
            num_steps: Optional[int] = None,
            num_episodes: Optional[int] = None) -> dict:
    assert num_steps or num_episodes # Either num_steps or num_episodes must be specified
    num_steps = num_steps if num_steps is not None else np.inf # If num_steps is not specified it's set to infinity
    num_episodes = num_episodes if num_episodes is not None else np.inf # If num_episodes is not specified it's set to infinty
    env.current_atoms.pbc = (False,False,False)
    #env.current_atoms.center(vacuum=10)
    obs = env.reset() # Resets the state of the environment and returns an initial observation

    ep_returns = []
    ep_lengths = []

    ep_length = 0
    ep_counter = 0
    step = 0
    start_time = time.time()

    while step < num_steps and ep_counter < num_episodes:
        pred = ac.step([obs]) # Make a step according to current observation using the AbstractActorCritic

        a = to_numpy(pred['a'][0]) # Get action
        next_obs, reward, done, _ = env.step(ac.to_action_space(action=a, observation=obs))
        # Get next observation, reward and boolean for "done" using action and observation
        buffer.store(obs=obs,
                     act=a,
                     reward=reward,
                     next_obs=next_obs,
                     terminal=done,
                     value=pred['v'].item(),
                     logp=pred['logp'].item())
        # Store information in buffer

        obs = next_obs # step to next observation

        step += 1 # Update number of steps
        ep_length += 1 # Update number of episodes

        last_step = step == num_steps - 1
        if done or last_step:
            # if trajectory didn't reach terminal state, bootstrap value target of next observation
            if not done:
                pred = ac.step([obs])
                value = float(pred['v'])
            else:
                value = 0

            ep_return = buffer.finish_path(value)

            if done:
                ep_returns.append(ep_return)
                ep_lengths.append(ep_length)
                ep_counter += 1

            ## CREATE STRUCTURE VISUALIZATION
            # use ending .png while testing
            # change to .traj once satisfied with algorithm
            if struc_vis_save == True:
                write(vis_dir + '/Structure_iter' + str(iteration_ctr) + '.traj',
                      env.current_atoms)

                print('------------------------------------Created trajectory------------------------------------')

            obs = env.reset()
            ep_length = 0

    # Compute statistics
    return_mean, return_std = mpi_mean_std(np.asarray(ep_returns), axis=0)
    ep_length_mean, ep_length_std = mpi_mean_std(np.asarray(ep_lengths), axis=0)

    value_mean, value_std = mpi_mean_std(buffer.val_buf[:buffer.ptr], axis=0)
    logp_mean, logp_std = mpi_mean_std(buffer.logp_buf[:buffer.ptr], axis=0)

    return {
        'time': time.time() - start_time,
        'num_steps': mpi_sum(np.asarray(step)).item(),
        'return_mean': return_mean.item(),
        'return_std': return_std.item(),
        'value_mean': value_mean.item(),
        'value_std': value_std.item(),
        'logp_mean': logp_mean.item(),
        'logp_std': logp_std.item(),
        'episode_length_mean': ep_length_mean.item(),
        'episode_length_std': ep_length_std.item(),
    }


def ppo(
    env: AbstractMolecularEnvironment,
    eval_env: AbstractMolecularEnvironment,
    ac: AbstractActorCritic,
    vis_dir: str,
    optimizer: str,
    gamma=0.99,
    start_num_steps=0,
    max_num_steps=4096,
    num_steps_per_iter=200,
    clip_ratio=0.2,
    learning_rate=3e-4,
    vf_coef=0.5,
    entropy_coef=0.0,
    max_num_train_iters=80,
    lam=0.97,
    target_kl=0.01,
    gradient_clip=0.5,
    save_freq=10,
    model_handler: Optional[ModelIO] = None,
    eval_freq=10,
    num_eval_episodes=1,
    rollout_saver: Optional[RolloutSaver] = None,
    save_train_rollout=False,
    save_eval_rollout=True,
    info_saver: Optional[InfoSaver] = None,
):
    """
    Proximal Policy Optimization (by clipping), with early stopping based on approximate KL

    Args:
        :param env: MolecularEnvironment for training.

        :param eval_env: MolecularEnvironment for evaluation.

        :param ac: Instance of an AbstractActorCritic

        :param vis_dir: Directory for structure visualizations

        :param num_steps_per_iter: Number of agent-environment interaction steps per iteration.

        :param start_num_steps: Initial number of steps

        :param max_num_steps: Maximum number of steps

        :param gamma: Discount factor. (Always between 0 and 1.)

        :param clip_ratio: Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        :param learning_rate: Learning rate for policy optimizer.

        :param vf_coef: coefficient for value function loss term

        :param entropy_coef: coefficient for entropy loss term

        :param learning_rate: Learning rate for optimizer.

        :param gradient_clip: clip norm of gradients before optimization step is taken

        :param max_num_train_iters: Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)

        :param lam: Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)

        :param target_kl: Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        :param eval_freq: How often to evaluate the policy

        :param num_eval_episodes: Number of evaluation episodes

        :param model_handler: Save model to file

        :param save_freq: How often the model is saved

        :param rollout_saver: Saves rollout buffers

        :param save_train_rollout: Save training rollout

        :param save_eval_rollout: Save evaluation rollout

        :param info_saver: Save statistics
    """
    eval_buffer_size = 1000

    # Set up experience buffer
    local_steps_per_iter = int(num_steps_per_iter / get_num_procs())
    buffer = PPOBuffer(int_act_dim=ac.internal_action_dim, size=local_steps_per_iter, gamma=gamma, lam=lam)

    # Set up optimizers for policy and value function
    if optimizer == 'adam':
        amsgrad = False
    elif optimizer == 'amsgrad':
        amsgrad = True
    else:
        raise RuntimeError(f"Unknown optimizer '{name}'")

    optimizer= Adam(ac.parameters(), lr=learning_rate, amsgrad=amsgrad)

    # Total number of steps
    total_num_steps = start_num_steps
    num_steps_per_iter = get_num_procs() * local_steps_per_iter
   # Modified manually by a factor of 3 below compared to molgym suite.
    max_num_iterations = (max_num_steps - total_num_steps) // num_steps_per_iter

    # Main loop
    for iteration in range(max_num_iterations):
        logging.info(f'Iteration: {iteration}/{max_num_iterations-1}, steps: {total_num_steps}')

        # Training rollout
        rollout_info = rollout(ac=ac, env=env, buffer=buffer, vis_dir = vis_dir, struc_vis_save =  False, num_steps=local_steps_per_iter)

        rollout_info['iteration'] = iteration
        rollout_info['total_num_steps'] = total_num_steps
        logging.info('Training rollout: ' + str(rollout_info))
        if info_saver:
            info_saver.save(rollout_info, name='train')

        # Safe training buffer
        if rollout_saver and save_train_rollout:
            rollout_saver.save(buffer, num_steps=total_num_steps, info='train')

        # Obtain (standardized) data for training
        data = buffer.get()

        # Train policy
        train_info = train(ac=ac,
                           optimizer=optimizer,
                           data=data,
                           clip_ratio=clip_ratio,
                           vf_coef=vf_coef,
                           entropy_coef=entropy_coef,
                           target_kl=target_kl,
                           gradient_clip=gradient_clip,
                           max_num_steps=max_num_train_iters)

        train_info['iteration'] = iteration
        train_info['total_num_steps'] = total_num_steps
        logging.info('Optimization: ' + str(train_info))
        if info_saver:
            info_saver.save(train_info, name='opt')

        # Update number of steps taken / trained
        total_num_steps += num_steps_per_iter

        # Evaluate policy
        if (iteration % eval_freq == 0) or (iteration == max_num_iterations - 1):
            # Create new buffer every time as it's not filled
            eval_buffer = PPOBuffer(int_act_dim=ac.internal_action_dim, size=eval_buffer_size, gamma=gamma, lam=lam)

            with torch.no_grad():
                ac.training = False
                rollout_info = rollout(ac, eval_env, eval_buffer, vis_dir = vis_dir, struc_vis_save =  True,
                                       iteration_ctr = iteration, num_episodes=num_eval_episodes)
                ac.training = True

            # Log information
            rollout_info['iteration'] = iteration
            rollout_info['total_num_steps'] = total_num_steps
            logging.info('Evaluation rollout: ' + str(rollout_info))
            if info_saver:
                info_saver.save(rollout_info, name='eval')

            # Safe evaluation buffer
            if rollout_saver and save_eval_rollout:
                rollout_saver.save(eval_buffer, num_steps=total_num_steps, info='eval')

        # Save model
        if model_handler and ((iteration % save_freq == 0) or (iteration == max_num_iterations - 1)):
            model_handler.save(ac, num_steps=total_num_steps)
