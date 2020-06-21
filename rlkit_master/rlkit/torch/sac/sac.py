from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,
            spectrum_loss_coef=0.,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.spectrum_loss_coef = spectrum_loss_coef
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        ###
        self._save_actions_info = True

    def train_from_torch(self, batch):
        batch_new = batch[0]
        paths = batch[1]

        batch = np_to_pytorch_batch(batch_new)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # print(len(batch))
        # # print(len(batch[0]))
        # # print(len(batch[0]))
        # # print((batch[0]))
        # batch_o = np_to_pytorch_batch(batch[0])
        # rewards = batch_o['rewards']
        # terminals = batch_o['terminals']
        # obs = batch_o['observations']
        # actions = batch_o['actions']
        # next_obs = batch_o['next_observations']
        # for i in range(1,len(batch)):
        #     path = batch[i]
        #     rewards = np.append(rewards,path['rewards'])
        #     terminals = np.append(terminals,path['terminals'])
        #     obs = np.append(terminals,path['observations'])
        #     actions = np.append(terminals,path['actions'])
        #     next_obs = np.append(terminals,path['next_observations'])

        # rewards = np_to_pytorch_batch(rewards)
        # terminals = np_to_pytorch_batch(terminals)
        # obs = np_to_pytorch_batch(obs)
        # actions = np_to_pytorch_batch(actions)
        # next_obs = np_to_pytorch_batch(next_obs)
        # print(type(rewards), rewards.shape)
        # print(a)
        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        """
        Entropy loss
        """
        entropy_path_arr = []
        # print(len(paths))
        for path_num, path in enumerate(paths):
            # print("path_num = ", path_num)
            # print(path.keys())
            # print(path['actions'].shape)

            for act_num in range(path['actions'].shape[1]):
                spectrum = torch.stft(torch.from_numpy(path["actions"][:, act_num]), n_fft=80, hop_length=6,
                                      win_length=40, normalized=True)
                entropy_act = torch.distributions.Categorical(probs=(spectrum.abs().mean(axis = 1)[:, 0])).entropy()
                # print("entropy_act = ", entropy_act)
                if act_num == 0:
                    entropy_by_act = torch.tensor([entropy_act])
                else:
                    entropy_by_act = torch.cat((entropy_by_act, torch.tensor([entropy_act])), 0)
                # print("entropy_by_act = ", entropy_by_act)

            entropy_path = entropy_by_act.sum()
            entropy_path_arr.append(entropy_path)

        spectrum_loss = torch.tensor(entropy_path_arr).mean()

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        #
        # obs_traj = batch_traj['observations']
        # new_actions_traj, *_ = self.policy(obs_traj, reparameterize=True, return_log_prob=True)
        # new_actions_traj = new_actions_traj.reshape(-1, 1000, new_actions_traj.shape[-1])
        # new_actions_traj = new_actions_traj.transpose(1, 2)
        # f = torch.rfft(new_actions_traj, 1)
        # amp_f = f[1:].norm(dim=-1)
        # spectrum_loss = amp_f.norm(p=1, dim=-1).mean()


        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        (policy_loss + self.spectrum_loss_coef * spectrum_loss).backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        ###
        # if self._save_actions_info:
        #     log_actions_file = '../data/name-of-experiment/actions_log.pkl'
        #     if os.path.exists(log_actions_file):
        #         actions_log = pd.read_pickle(log_actions_file)
        #     else:
        #         actions_log = pd.DataFrame()
        #     columns = ['action_' + str(i) for i in range((new_obs_actions.shape[1]))]
        #     actions_log_new = pd.DataFrame(new_obs_actions, columns=columns)
        #
        #     actions_log = pd.concat([actions_log,actions_log_new])
        #     print(actions_log.shape)
        #     print(actions_log.head(3))
        #     actions_log.to_pickle(log_actions_file)

        ###
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'entropy loss',
                spectrum_loss.item(),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )
