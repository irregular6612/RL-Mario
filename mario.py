import torch
from torch import nn
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, time, os
import multiprocessing as mp

# Gym은 강화학습을 위한 OpenAI 툴킷입니다.
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# OpenAI Gym을 위한 NES 에뮬레이터
from nes_py.wrappers import JoypadSpace

# OpenAI Gym에서의 슈퍼 마리오 환경 세팅
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, LazyTensorStorage
from tqdm import tqdm

import matplotlib.pyplot as plt
import gc


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """모든 `skip` 프레임만 반환합니다."""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """행동을 반복하고 포상을 더합니다."""
        total_reward = 0.0
        for _ in range(self._skip):
            # 포상을 누적하고 동일한 작업을 반복합니다.
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        # dtype: uint8 -> float32: 수렴성을 위해?
        self.observation_space = Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def permute_orientation(self, observation):
        # [H, W, C] 배열을 [C, H, W] 텐서로 바꿉니다.
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float32)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Compose([
            T.Grayscale(),
        ])
        observation = transform(observation)
        observation /= 255.0 # Normalize 0~255 -> 0~1
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 1)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

class Mario:
    def __init__(self, state_dim, action_dim, save_dir, performance_mode="balanced"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.performance_mode = performance_mode  # 성능 모드 저장

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # 마리오의 DNN은 최적의 행동을 예측합니다 - 이는 학습하기 섹션에서 구현합니다.
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e4  # Mario Net 저장 사이의 경험 횟수
        
        # info와 ep_rewards 초기화
        self.info = {}
        self.ep_rewards = []

        # 배치 사이즈를 GPU 메모리에 맞게 증가
        self.batch_size = 128  # 32에서 128로 증가

    def act(self, state):
        """
        주어진 상태에서, 입실론-그리디 행동(epsilon-greedy action)을 선택하고, 스텝의 값을 업데이트 합니다.

        입력값:
        state (``LazyFrame``): 현재 상태에서의 단일 상태(observation)값을 말합니다. 차원은 (state_dim)입니다.
        출력값:
        ``action_idx`` (int): Mario가 수행할 행동을 나타내는 정수 값입니다.
        """
        # 임의의 행동을 선택하기
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # 최적의 행동을 이용하기
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            state /= 255.0 # Normalize 0~255 -> 0~1
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # exploration_rate 감소하기
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # 스텝 수 증가하기
        self.curr_step += 1
        return action_idx

class Mario(Mario):  # 연속성을 위한 하위 클래스입니다.
    def __init__(self, state_dim, action_dim, save_dir, performance_mode="balanced"):
        super().__init__(state_dim, action_dim, save_dir, performance_mode)
        
        # 성능 모드에 따른 메모리 버퍼 설정
        current_process = mp.current_process()
        is_worker = current_process.name != 'MainProcess'
        
        if is_worker:
            # 워커 프로세스에서 성능 모드에 따른 메모리 버퍼 선택
            if performance_mode == "speed":
                # 속도 우선: GPU 메모리 버퍼 시도
                try:
                    if self.device == "cuda":
                        self.memory = TensorDictReplayBuffer(
                            storage=LazyTensorStorage(50000, device=self.device)
                        )
                        print(f"워커 {current_process.name}: 속도 우선 - GPU 메모리 버퍼 사용")
                    else:
                        raise RuntimeError("CUDA 없음")
                except Exception as e:
                    print(f"워커 {current_process.name}: GPU 메모리 버퍼 실패 ({e}), CPU 메모리 버퍼 사용")
                    self.memory = TensorDictReplayBuffer(
                        storage=LazyMemmapStorage(50000, device=torch.device("cpu"))
                    )
                    if self.device in ["cuda", "mps"]:
                        self.memory.append_transform(lambda x: x.to(self.device))
                        
            elif performance_mode == "memory_safe":
                # 메모리 안전: 항상 CPU 메모리 버퍼
                print(f"워커 {current_process.name}: 메모리 안전 - CPU 메모리 버퍼 강제 사용")
                self.memory = TensorDictReplayBuffer(
                    storage=LazyMemmapStorage(50000, device=torch.device("cpu"))
                )
                if self.device in ["cuda", "mps"]:
                    self.memory.append_transform(lambda x: x.to(self.device))
                    
            else:  # balanced
                # 균형 모드: GPU 메모리 시도 후 실패 시 CPU 메모리
                try:
                    if self.device == "cuda":
                        # 작은 메모리 버퍼로 시작 (30000 -> 25000)
                        self.memory = TensorDictReplayBuffer(
                            storage=LazyTensorStorage(25000, device=self.device)
                        )
                        print(f"워커 {current_process.name}: 균형 모드 - 작은 GPU 메모리 버퍼 사용")
                    else:
                        raise RuntimeError("CUDA 없음")
                except Exception as e:
                    print(f"워커 {current_process.name}: GPU 메모리 버퍼 실패 ({e}), CPU 메모리 버퍼 사용")
                    self.memory = TensorDictReplayBuffer(
                        storage=LazyMemmapStorage(50000, device=torch.device("cpu"))
                    )
                    if self.device in ["cuda", "mps"]:
                        self.memory.append_transform(lambda x: x.to(self.device))
        else:
            # 메인 프로세스에서는 기존 로직 사용
            if self.device == "mps":
                # MPS는 통합 메모리이므로 CPU 메모리 버퍼 사용 (데이터 이동 불필요)
                self.memory = TensorDictReplayBuffer(
                    storage=LazyMemmapStorage(50000, device=torch.device("cpu"))
                )
                print(f"MPS 통합 메모리 사용: CPU 메모리 버퍼 (데이터 이동 없음)")
            else:
                # CUDA 또는 CPU의 경우 기존 로직 사용
                try:
                    # LazyTensorStorage 사용 시도 (CUDA의 경우)
                    self.memory = TensorDictReplayBuffer(
                        storage=LazyTensorStorage(50000, device=self.device)
                    )
                    print(f"GPU 메모리 버퍼 사용: {self.device}")
                except (ImportError, AttributeError, RuntimeError) as e:
                    print(f"GPU 메모리 버퍼 사용 실패 ({e}), CPU 메모리 버퍼 + transform 사용")
                    # LazyTensorStorage가 없거나 오류 발생 시 기존 방식 + transform 사용
                    self.memory = TensorDictReplayBuffer(
                        storage=LazyMemmapStorage(50000, device=torch.device("cpu"))
                    )
                    # CUDA의 경우에만 GPU로 데이터 이동하는 transform 추가
                    if self.device == "cuda":
                        self.memory.append_transform(lambda x: x.to(self.device))
        
        self.prev_info = None

    def cache(self, state, next_state, action, reward, done, info):
        """
        Store the experience to self.memory (replay buffer)

        입력값:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """

        # info를 self.info로 업데이트
        self.info = info.copy()
        
        # 리워드 계산
        reward = self.calculate_reward(reward, info, done, self.prev_info)
        self.prev_info = info.copy()
        
        # 기존 캐시 로직 - GPU에서 직접 텐서 생성
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        
        # numpy 배열 변환을 최소화하고 직접 GPU 텐서로 생성
        state_array = first_if_tuple(state).__array__()
        next_state_array = first_if_tuple(next_state).__array__()

        if self.device == "mps":
            # MPS는 통합 메모리이므로 CPU에서 텐서 생성 (자동으로 MPS에서 접근 가능)
            state = torch.tensor(state_array, dtype=torch.float32)
            next_state = torch.tensor(next_state_array, dtype=torch.float32)
            action = torch.tensor([action], dtype=torch.int64)
            reward = torch.tensor([reward], dtype=torch.float32)
            done = torch.tensor([done])
        else:
            # CUDA 또는 CPU의 경우 해당 디바이스에서 직접 텐서 생성
            state = torch.tensor(state_array, dtype=torch.float32, device=self.device)
            next_state = torch.tensor(next_state_array, dtype=torch.float32, device=self.device)
            action = torch.tensor([action], dtype=torch.int64, device=self.device)
            reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
            done = torch.tensor([done], device=self.device)

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        """
        메모리에서 일련의 경험들을 검색합니다.
        """
        # 배치 데이터를 미리 GPU로 로드
        if self.device == "cuda":
            batch = self.memory.sample(self.batch_size).to(self.device, non_blocking=True)
        else:
            batch = self.memory.sample(self.batch_size)
        
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

class MarioNet(nn.Module):
    """작은 CNN 구조
  입력 -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> 출력
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target 매개변수 값은 고정시킵니다.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model="online"):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir, performance_mode="balanced"):
        super().__init__(state_dim, action_dim, save_dir, performance_mode)
        self.gamma = 0.9
        self.reward_scale = 0.1  # 보상 스케일링 파라미터
        self.q_scale = 1.0      # Q값 스케일링 파라미터
        
        # 보상 정규화를 위한 통계
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_count = 0
        
        # Q값 정규화를 위한 통계
        self.q_mean = 0
        self.q_std = 1
        self.q_count = 0
    
    def update_statistics(self, value, is_reward=True):
        """온라인 통계 업데이트"""
        if is_reward:
            self.reward_count += 1
            delta = value - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = value - self.reward_mean
            self.reward_std += delta * delta2
        else:
            self.q_count += 1
            delta = value - self.q_mean
            self.q_mean += delta / self.q_count
            delta2 = value - self.q_mean
            self.q_std += delta * delta2

    def normalize_value(self, value, is_reward=True):
        """값 정규화"""
        if is_reward:
            return (value - self.reward_mean) / (self.reward_std + 1e-8)
        else:
            return (value - self.q_mean) / (self.q_std + 1e-8)

    def calculate_reward(self, reward, info, done, prev_info=None):
        """보상 계산 및 스케일링"""
        total_reward = reward * self.reward_scale
        
        if info["flag_get"]:    # 정복
            total_reward += 10000.0
        if prev_info and info["coins"] > prev_info["coins"]:    # 코인 획득
            total_reward += 100.0
        if not done:    # 진행도
            total_reward += 1.0
        if prev_info and info["x_pos"] > prev_info["x_pos"]:    # 진행도 보너스
            total_reward += 10.0
        if prev_info and info["x_pos"] <= prev_info["x_pos"]:    # 진행도 감점
            total_reward -= 10.0
        if prev_info and info["life"] < prev_info["life"]:    # 생명 감소시 200점 패널티
            total_reward -= 200.0
        if prev_info and info["y_pos"] < prev_info["y_pos"]:  # 위로 올라갈 때
            total_reward += 5.0
        if prev_info and info["x_pos"] - prev_info["x_pos"] > 5:  # 빠른 이동
            total_reward += 15.0
        # 보상 정규화
        normalized_reward = self.normalize_value(total_reward, is_reward=True)
        self.update_statistics(total_reward, is_reward=True)
        
        return normalized_reward


    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        
        # 정규화된 값으로 TD 타겟 계산
        normalized_Q = self.normalize_value(next_Q, is_reward=False)

        #td_target = (reward + (1 - done.float()) * self.gamma * next_Q).float()
        td_target = (reward + (1 - done.float()) * self.gamma * normalized_Q).float()
        
        return td_target

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir, performance_mode="balanced"):
        super().__init__(state_dim, action_dim, save_dir, performance_mode)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir, performance_mode="balanced"):
        super().__init__(state_dim, action_dim, save_dir, performance_mode)
        self.best_reward = float('-inf')
        self.success_count = 0
        self.save_conditions = {
            'performance': False,  # 성능 기반 저장
            'stability': False,    # 안정성 기반 저장
            'periodic': False      # 주기적 저장
        }
    
    def save(self):
        # ep_rewards가 비어있지 않을 때만 평균 계산
        if len(self.ep_rewards) >= 10:
            current_reward = np.mean(self.ep_rewards[-10:])
        elif len(self.ep_rewards) > 0:
            current_reward = np.mean(self.ep_rewards)
        else:
            current_reward = float('-inf')
        
        # 성능 기반 저장
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.save_conditions['performance'] = True
        
        # 안정성 기반 저장
        if self.info.get("flag_get", False):
            self.success_count += 1
            if self.success_count >= 5:
                self.save_conditions['stability'] = True
        
        # 주기적 저장
        if self.curr_step % self.save_every == 0:
            self.save_conditions['periodic'] = True
        
        # 저장 조건 충족 시 저장
        if any(self.save_conditions.values()):
            save_path = self.save_dir / f"mario_net_{self.curr_step}.chkpt"
            torch.save({
                'model': self.net.state_dict(),
                'exploration_rate': self.exploration_rate,
                'best_reward': self.best_reward,
                'success_count': self.success_count,
                'save_conditions': self.save_conditions
            }, save_path)
            
            # 조건 초기화
            self.save_conditions = {k: False for k in self.save_conditions}

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir, performance_mode="balanced"):
        super().__init__(state_dim, action_dim, save_dir, performance_mode)
        self.burnin = 1e3  # 학습을 진행하기 전 최소한의 경험값. 1e4 -> 1e3 학습을 더 빠르게 시작.
        self.learn_every = 1  # Q_online 업데이트 사이의 경험 횟수. 매번 학습 3 -> 1
        self.sync_every = 1e3  # Q_target과 Q_online sync 사이의 경험 수 1e4 -> 1e3

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # 메모리로부터 샘플링을 합니다.
        state, next_state, action, reward, done = self.recall()

        # TD 추정값을 가져옵니다.
        td_est = self.td_estimate(state, action)

        # TD 목표값을 가져옵니다.
        td_tgt = self.td_target(reward, next_state, done)

        # 실시간 Q(Q_online)을 통해 역전파 손실을 계산합니다.
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


class MetricLogger:
    def __init__(self, save_dir, model=None, device=None):
        self.save_log = save_dir / "log"
        self.writer = SummaryWriter(os.path.join(save_dir, "tensorboard"))

        self.max_history_length = 1000  # 최대 저장할 에피소드 수
        self.cleanup_interval = 100     # 몇 에피소드마다 정리할지
        self.last_cleanup = 0           # 마지막 정리 시점

        # model 기록
        if model is None:
            raise ValueError("model is required")
        
        dummy_input = torch.zeros((1, 4, 84, 84)).to(device)
        self.writer.add_graph(model, dummy_input)  # 모델 구조 기록
        # 모델 파라미터 수 계산 및 기록
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.writer.add_text('Model/Parameters', f'Total Parameters: {total_params:,}\nTrainable Parameters: {trainable_params:,}')
        
        # 각 레이어의 파라미터 수 기록
        for name, param in model.named_parameters():
            self.writer.add_text('Model/Layers', f'{name}: {param.numel():,} parameters')
        

        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # 지표(Metric)와 관련된 리스트입니다.
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # 모든 record() 함수를 호출한 후 이동 평균(Moving average)을 계산합니다.
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # 현재 에피스드에 대한 지표를 기록합니다.
        self.init_episode()

        # 시간에 대한 기록입니다.
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "에피스드의 끝을 표시합니다."
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step, model=None):
        self.cleanup_old_data(episode)

        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        # tensorboard에 기록
        self.writer.add_scalar('Metrics/Mean Reward', mean_ep_reward, episode)
        self.writer.add_scalar('Metrics/Mean Length', mean_ep_length, episode)
        self.writer.add_scalar('Metrics/Mean Loss', mean_ep_loss, episode)
        self.writer.add_scalar('Metrics/Mean Q Value', mean_ep_q, episode)
        self.writer.add_scalar('Metrics/Epsilon', epsilon, episode)
        self.writer.add_scalar('Metrics/Step', step, episode)

        # 모델 파라미터의 그래디언트와 가중치 분포 기록
        if model is not None:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, episode)
                self.writer.add_histogram(f'Weights/{name}', param.data, episode)    

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))
            self.writer.add_figure(f'Plots/{metric}', plt.gcf(), episode)
    
    def cleanup_old_data(self, current_episode):
        """오래된 데이터 정리"""
        if current_episode - self.last_cleanup >= self.cleanup_interval:
            # 이동 평균 데이터는 유지하고 나머지 데이터 정리
            if len(self.ep_rewards) > self.max_history_length:
                self.ep_rewards = self.ep_rewards[-self.max_history_length:]
            if len(self.ep_lengths) > self.max_history_length:
                self.ep_lengths = self.ep_lengths[-self.max_history_length:]
            if len(self.ep_avg_losses) > self.max_history_length:
                self.ep_avg_losses = self.ep_avg_losses[-self.max_history_length:]
            if len(self.ep_avg_qs) > self.max_history_length:
                self.ep_avg_qs = self.ep_avg_qs[-self.max_history_length:]
            
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.last_cleanup = current_episode
    def close(self):
        self.writer.close()


def main():
    # 슈퍼 마리오 환경 초기화하기 (in v0.26 change render mode to 'human' to see results on the screen)
    # numpy 2.0 version은 uint8에서 crash
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)

    # 상태 공간을 2가지로 제한하기
    #   0. 오른쪽으로 걷기
    #   1. 오른쪽으로 점프하기
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    # 래퍼를 환경에 적용합니다.
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    print(f"CUDA-available: {use_cuda}, MPS-available: {use_mps}")

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, performance_mode="balanced")

    logger = MetricLogger(save_dir, model = mario.net, device = mario.device)

    episodes = 10000
    for e in range(episodes):

        state = env.reset()
        episode_reward = 0  # 에피소드 보상 추적

        # 게임을 실행시켜봅시다!
        while True:

            # 현재 상태에서 에이전트 실행하기
            action = mario.act(state)

            # 에이전트가 액션 수행하기
            next_state, reward, done, trunc, info = env.step(action)

            # 에피소드 보상 누적
            episode_reward += reward

            # 기억하기
            mario.cache(state, next_state, action, reward, done, info)

            # 배우기
            q, loss = mario.learn()

            # 기록하기
            logger.log_step(reward, loss, q)

            # 상태 업데이트하기
            state = next_state

            # 게임이 끝났는지 확인하기
            if done or info["flag_get"]:
                break

        # 에피소드 보상을 mario.ep_rewards에 추가
        mario.ep_rewards.append(episode_reward)
        
        logger.log_episode()

        if (e % 5 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step, model=mario.net)
    logger.close()

if __name__ == "__main__":
    main()