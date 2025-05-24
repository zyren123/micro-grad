import gymnasium as gym
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import imageio.v2 as imageio
import os
import pickle  # 添加pickle库

# 导入用户的框架
from tinygrad.tensor import Tensor
from tinygrad.module import  MLP

# 设置随机种子以便复现结果
np.random.seed(42)
random.seed(42)

# 创建保存GIF的目录
os.makedirs("gifs", exist_ok=True)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为numpy数组
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.vstack(next_states)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN代理"""
    def __init__(self, state_size, action_size, hidden_layers=[256, 256,256], 
                 learning_rate=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995):
        """
        初始化DQN代理
        
        参数:
            state_size: 状态空间维度
            action_size: 动作空间维度
            hidden_layers: 隐藏层的节点数列表
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 探索率
            epsilon_min: 最小探索率
            epsilon_decay: 探索率衰减
        """
        # 创建Q网络
        self.q_network = MLP(state_size, hidden_layers + [action_size], init_method='xavier')
        
        # 创建目标Q网络
        self.target_network = MLP(state_size, hidden_layers + [action_size], init_method='xavier')
        self.update_target_network()  # 初始化目标网络权重
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
    def update_target_network(self):
        """更新目标网络的权重为当前Q网络的权重"""
        for target_param, param in zip(self.target_network.params, self.q_network.params):
            target_param.data = np.copy(param.data)
    
    def choose_action(self, state):
        """选择动作，使用epsilon-greedy策略"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # 将状态转换为Tensor
        state_tensor = Tensor(state.reshape(1, -1))
        
        # 前向传播获取Q值
        q_values = self.q_network(state_tensor)
        
        # 返回最大Q值对应的动作
        return np.argmax(q_values.data.flatten())
    
    def train(self, state, action, reward, next_state, done):
        """训练Q网络
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        # 将输入转换为Tensor
        state_tensor = Tensor(state)
        next_state_tensor = Tensor(next_state)
        
        # 计算当前状态的Q值
        q_values = self.q_network(state_tensor)
        
        # 使用目标网络计算下一个状态的Q值
        next_q_values = self.target_network(next_state_tensor)
        
        # 为每个样本创建目标Q值
        target_q = np.copy(q_values.data)
        batch_size = state.shape[0]
        
        for i in range(batch_size):
            if done[i]:
                target_q[i, action[i]] = reward[i]
            else:
                # Q-learning更新规则
                target_q[i, action[i]] = reward[i] + self.gamma * np.max(next_q_values.data[i])
        
        # 将目标Q值转换为Tensor
        target_q_tensor = Tensor(target_q)
        
        # 计算MSE损失
        loss = ((q_values - target_q_tensor) ** 2).mean()
        
        # 反向传播和更新
        self.q_network.zero_grad()
        loss.backward()
        
        # 简单的SGD更新
        for param in self.q_network.params:
            if param.grad is not None:
                param.data -= self.learning_rate * param.grad
        
        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.data

    def save(self, filename):
        """保存模型权重
        
        使用pickle保存权重列表，处理不同形状的参数
        """
        weights = [p.data for p in self.q_network.params]
        with open(filename, 'wb') as f:
            pickle.dump(weights, f)
        
    def load(self, filename):
        """加载模型权重
        
        使用pickle加载权重列表
        """
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        for i, param in enumerate(self.q_network.params):
            param.data = weights[i]

def record_gif(agent, env_name, gif_path, num_frames=200):
    """记录代理的表现并保存为GIF
    
    参数:
        agent: 要评估的DQN代理
        env_name: 环境名称
        gif_path: 保存GIF的路径
        num_frames: 要捕获的最大帧数
    """
    try:
        # 创建带有rgb_array渲染模式的环境
        env = gym.make(env_name, render_mode="rgb_array")
        
        state, _ = env.reset()
        frames = []
        total_reward = 0
        
        # 记录一个回合的表现
        for t in range(num_frames):
            # 渲染并捕获帧
            frame = env.render()
            frames.append(frame)
            
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            
            # 更新状态和累计奖励
            state = next_state
            total_reward += reward
            
            # 如果回合结束，重置环境
            if done or truncated:
                print(f"回合在第{t+1}步结束，总奖励: {total_reward:.2f}")
                break
        
        # 关闭环境
        env.close()
        
        if len(frames) > 0:
            # 保存为GIF
            imageio.mimsave(gif_path, frames, fps=30)
            print(f"GIF已保存到 {gif_path}，总奖励: {total_reward:.2f}")
        else:
            print(f"未能捕获任何帧，无法创建GIF")
        
        return total_reward
        
    except Exception as e:
        print(f"创建GIF时发生错误: {str(e)}")
        print("继续训练但跳过GIF生成...")
        return 0.0

def train_lunar_lander(agent, env, num_episodes=100, batch_size=64, 
                      update_target_every=100, replay_buffer_size=100000,
                      min_replay_size=1000, render=False):
    """
    训练Lunar Lander
    
    参数:
        agent: DQN代理
        env: Gymnasium环境
        num_episodes: 训练的总回合数
        batch_size: 每次更新的批量大小
        update_target_every: 更新目标网络的频率
        replay_buffer_size: 经验回放缓冲区大小
        min_replay_size: 开始训练前的最小缓冲区大小
        render: 是否渲染环境
    """
    # 创建经验回放缓冲区
    replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
    
    # 记录每个回合的总奖励
    rewards_history = []
    
    # 最近100回合的平均奖励
    avg_rewards_history = []
    
    # 用于绘图
    episodes = []
    
    # 最佳平均奖励
    best_avg_reward = -float('inf')
    
    # 计算1/5训练进度的回合数
    checkpoint_interval = max(1, num_episodes // 5)
    
    # 进度条
    pbar = tqdm(range(num_episodes), desc="训练中")
    
    for episode in pbar:
        # 重置环境
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (done or truncated):
            # 如果渲染，则显示环境
            if render and episode % 50 == 0:
                env.render()
                time.sleep(0.01)
            
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            
            # 保存经验
            replay_buffer.add(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            
            # 累加奖励
            total_reward += reward
            steps += 1
            
            # 如果缓冲区大小足够，则进行训练
            if len(replay_buffer) > min_replay_size:
                # 从缓冲区采样
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # 训练代理
                loss = agent.train(states, actions, rewards, next_states, dones)
                
                # 更新目标网络
                if steps % update_target_every == 0:
                    agent.update_target_network()
        
        # 记录回合奖励
        rewards_history.append(total_reward)
        
        # 计算最近100回合的平均奖励
        avg_reward = np.mean(rewards_history[-100:])
        avg_rewards_history.append(avg_reward)
        episodes.append(episode)
        
        # 更新进度条
        pbar.set_postfix({
            '回合': episode, 
            '奖励': f'{total_reward:.2f}',
            '平均(100)': f'{avg_reward:.2f}',
            'Epsilon': f'{agent.epsilon:.4f}'
        })
        
        # 保存最佳模型
        if avg_reward > best_avg_reward and episode >= 100:
            best_avg_reward = avg_reward
            agent.save('best_dqn_model.pkl')
            print(f"\n保存了新的最佳模型，平均奖励: {best_avg_reward:.2f}")
        
        # 每1/5的训练进度记录GIF
        if (episode + 1) % checkpoint_interval == 0 or episode == num_episodes - 1:
            progress_percent = (episode + 1) / num_episodes * 100
            # 保存当前模型
            agent.save(f'model_checkpoint_{episode+1}.pkl')
            
            # 记录GIF
            gif_path = f"gifs/lunar_lander_progress_{progress_percent:.0f}percent.gif"
            print(f"\n已完成{progress_percent:.0f}%的训练，正在生成训练进度动画...")
            reward = record_gif(agent, "LunarLander-v3", gif_path)
            print(f"在{progress_percent:.0f}%进度下的表现: 奖励 = {reward:.2f}")
    
    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards_history, label='reward')
    plt.plot(episodes, avg_rewards_history, label='average reward(100)', color='red')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.title('LunarLander-v3的DQN训练')
    plt.savefig('lunar_lander_rewards.png')
    plt.show()
    
    return rewards_history, avg_rewards_history

def evaluate_agent(agent, env, num_episodes=10, render=True):
    """评估训练好的代理"""
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            if render:
                env.render()
                time.sleep(0.01)
            
            # 使用训练好的策略选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            
            # 更新状态和累加奖励
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        print(f"Episode {episode+1}: Reward = {total_reward:.2f}")
    
    print(f"Average Reward over {num_episodes} episodes: {np.mean(rewards):.2f}")
    return rewards

if __name__ == "__main__":
    # 创建Lunar Lander环境
    env = gym.make("LunarLander-v3")
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"状态空间大小: {state_size}")
    print(f"动作空间大小: {action_size}")
    
    # 创建DQN代理
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_layers=[256, 256,256],
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # 训练代理
    print("开始训练DQN代理...")
    train_lunar_lander(
        agent=agent,
        env=env,
        num_episodes=300,  # 减少回合数以加快训练
        batch_size=64,
        update_target_every=50,
        replay_buffer_size=100000,
        min_replay_size=1000,
        render=False  # 训练时通常不渲染
    )
    
    print("训练完成！")
    
    # 加载最佳模型
    agent.load('best_dqn_model.pkl')
    
    # 创建新环境进行评估
    eval_env = gym.make("LunarLander-v3", render_mode="human")
    
    # 评估代理
    print("开始评估DQN代理...")
    evaluate_agent(agent, eval_env, num_episodes=5, render=True)
    
    # 关闭环境
    env.close()
    eval_env.close() 