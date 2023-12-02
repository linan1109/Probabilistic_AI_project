import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode
from torch.nn import functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, hidden_activation, 
                                output_activation):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        self._nn = nn.Sequential()
        self._nn.add_module('input', nn.Linear(input_dim, hidden_size))
        self._nn.add_module('input_activation', hidden_activation)
        for i in range(hidden_layers):
            self._nn.add_module('hidden_{}'.format(i), nn.Linear(hidden_size, hidden_size))
            self._nn.add_module('hidden_activation_{}'.format(i), hidden_activation)
        self._nn.add_module('output', nn.Linear(hidden_size, output_dim))
        self._nn.add_module('output_activation', output_activation)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        output = self._nn(s)
        return output

    
class Actor:
    def __init__(self,hidden_size: int=256, hidden_layers: int=1, actor_lr: float=0.001,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py. 
        self._net = NeuralNetwork(self.state_dim, 1, self.hidden_size, self.hidden_layers-1, nn.ReLU(), nn.Tanh())
        self._target_net = NeuralNetwork(self.state_dim, 1, self.hidden_size, self.hidden_layers-1, nn.ReLU(), nn.Tanh())
        self.optimizer = optim.Adam(self._net.parameters(), lr=self.actor_lr)
        
        self._log_std = nn.Parameter(torch.zeros(1, self.action_dim))
        self.optimizer_log_std = optim.Adam([self._log_std], lr=self.actor_lr)


class Critic:
    def __init__(self, hidden_size: int=256, 
                 hidden_layers: int=1, critic_lr: int=0.001, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self._net = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers-1, nn.ReLU(), nn.Identity())
        self._target_net = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers-1, nn.ReLU(), nn.Identity())
        self.optimizer = optim.Adam(self._net.parameters(), lr=self.critic_lr)
        
        
class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.   
        self._actor = Actor(actor_lr = 0.00005, device=self.device)
        self._critic = Critic(device=self.device)
        self._gamma = 0.99
        self._tau = 0.001
        

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        # action = np.random.uniform(-1, 1, (1,))
        
        if train:
            # sample from policy
            action = self._actor._net(torch.tensor(s, dtype=torch.float, device=self.device))
            action = Normal(action, torch.exp(self._actor._log_std)).sample()
            action = action.cpu().detach().numpy()
            action = action.reshape(1,)
            action = np.clip(action, -1, 1)
            
        else:
            action = self._actor._target_net(torch.tensor(s, dtype=torch.float, device=self.device))
            action = action.cpu().detach().numpy()
            action = action.reshape(1,)     
                 
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        for i in range(20):
            batch = self.memory.sample(self.batch_size)
            s_batch, a_batch, r_batch, s_prime_batch = batch
            
            Q_prime = self._critic._target_net(torch.cat((s_prime_batch, self._actor._target_net(s_prime_batch)), dim=1))
            y = r_batch + self._gamma * Q_prime

            # TODO: Implement Critic(s) update here.
            Q_a = self._critic._net(torch.cat((s_batch, a_batch), dim=1))
            loss_critic = F.mse_loss(Q_a, y.detach())
            self.run_gradient_update_step(self._critic, loss_critic)
            
            # TODO: Implement Policy update here
            Q_pai = self._critic._net(torch.cat((s_batch, self._actor._net(s_batch)), dim=1))
            loss_actor = -Q_pai.mean()
            self.run_gradient_update_step(self._actor, loss_actor)
            
            # update log_std for actor
            actions = self._actor._net(s_batch)
            log_prob = Normal(actions, torch.exp(self._actor._log_std)).log_prob(a_batch)
            loss_log_std = -log_prob.mean()
            self._actor.optimizer_log_std.zero_grad()
            loss_log_std.backward()
            self._actor.optimizer_log_std.step()
            
            # update target network
            self.critic_target_update(self._critic._net, self._critic._target_net, self._tau, True)
            self.critic_target_update(self._actor._net, self._actor._target_net, self._tau, True)



# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
