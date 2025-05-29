import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import numpy as np
from pathlib import Path
import datetime
import time
import gc
import gym
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from torch.utils.tensorboard import SummaryWriter

# mario.py에서 필요한 클래스들 import
from mario import (
    SkipFrame,
    GrayScaleObservation, 
    ResizeObservation,
    Mario,
    MetricLogger
)

class ParallelTrainer:
    def __init__(self, num_workers=None, episodes_per_worker=100, performance_mode="balanced"):
        """
        병렬 학습을 위한 트레이너 클래스
        
        Args:
            num_workers: 워커 프로세스 수 (None이면 성능 모드에 따라 자동 설정)
            episodes_per_worker: 각 워커당 실행할 에피소드 수
            performance_mode: "balanced", "speed", "memory_safe" 중 선택
        """
        self.performance_mode = performance_mode
        
        # 성능 모드에 따른 동적 워커 수 설정
        if num_workers is None:
            num_workers = self._get_optimal_workers()
        
        self.num_workers = num_workers
        self.episodes_per_worker = episodes_per_worker
        
        # 환경 설정
        self.world = 1
        self.stage = 1
        self.frame_skip = 4
        self.resize_shape = 84
        self.render_mode = 'rgb_array'  # 병렬 처리에서는 rgb_array 사용
        
        # 학습 하이퍼파라미터 (mario.py와 동일)
        self.state_dim = (4, 84, 84)
        
        print(f"병렬 학습 초기화: {self.num_workers}개 워커, 워커당 {self.episodes_per_worker} 에피소드")
        print(f"성능 모드: {self.performance_mode}")
        print(f"CPU 코어 수: {mp.cpu_count()}, 선택된 워커 수: {self.num_workers}")
    
    def _get_optimal_workers(self):
        """성능 모드와 하드웨어를 고려한 최적 워커 수 계산"""
        cpu_count = mp.cpu_count()
        
        # GPU 메모리 정보 확인
        if torch.cuda.is_available():
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU 메모리: {gpu_memory_gb:.1f} GB")
            except:
                gpu_memory_gb = 20  # 기본값
        else:
            gpu_memory_gb = 0
        
        # 성능 모드별 워커 수 계산
        if self.performance_mode == "speed":
            # 속도 우선: GPU 메모리 최대 활용, 적은 워커 수
            if gpu_memory_gb >= 20:
                optimal_workers = 2  # GPU 메모리 집중 사용
                print("속도 우선 모드: 2개 워커로 GPU 메모리 최대 활용")
            else:
                optimal_workers = 1
                print("속도 우선 모드: GPU 메모리 부족으로 1개 워커")
                
        elif self.performance_mode == "memory_safe":
            # 메모리 안전: CPU 메모리 사용, 많은 워커 수
            if cpu_count >= 16:
                optimal_workers = min(6, cpu_count - 2)
            elif cpu_count >= 8:
                optimal_workers = min(4, cpu_count - 2)
            else:
                optimal_workers = min(3, cpu_count - 1)
            print(f"메모리 안전 모드: {optimal_workers}개 워커로 CPU 메모리 사용")
            
        else:  # balanced
            # 균형 모드: GPU 메모리와 워커 수의 균형
            if gpu_memory_gb >= 20:
                if cpu_count >= 16:
                    optimal_workers = 3  # 16 vCPU에서 균형점
                elif cpu_count >= 8:
                    optimal_workers = 2  # 8 vCPU에서 균형점
                else:
                    optimal_workers = 2  # 기본 균형점
            else:
                optimal_workers = min(4, cpu_count - 1)
            print(f"균형 모드: {optimal_workers}개 워커로 GPU/CPU 메모리 혼합 사용")
        
        return optimal_workers
    
    @staticmethod
    def get_cpu_recommendations():
        """GCP에서 L4 GPU와 함께 사용할 수 있는 CPU 옵션별 권장 워커 수"""
        recommendations = {
            "현재 설정 (4 vCPU)": {
                "machine_type": "g2-standard-4",
                "recommended_workers": 3,  # GPU 메모리 제한 고려
                "expected_performance": "기준 성능",
                "cost_efficiency": "높음"
            },
            "업그레이드 옵션 1 (8 vCPU)": {
                "machine_type": "g2-standard-8", 
                "recommended_workers": 4,  # GPU 메모리 제한 고려
                "expected_performance": "1.3-1.5배 향상",
                "cost_efficiency": "중간"
            },
            "업그레이드 옵션 2 (16 vCPU)": {
                "machine_type": "g2-standard-16",
                "recommended_workers": 5,  # GPU 메모리 제한 고려
                "expected_performance": "1.5-2배 향상", 
                "cost_efficiency": "중간"
            },
            "고성능 옵션 (32 vCPU)": {
                "machine_type": "g2-standard-32",
                "recommended_workers": 6,  # GPU 메모리 제한 고려
                "expected_performance": "2-2.5배 향상",
                "cost_efficiency": "낮음 (고비용)"
            }
        }
        
        print("\n=== GCP L4 GPU + CPU 조합별 권장 설정 (GPU 메모리 제한 고려) ===")
        for config_name, config in recommendations.items():
            print(f"\n{config_name}:")
            print(f"  - 머신 타입: {config['machine_type']}")
            print(f"  - 권장 워커 수: {config['recommended_workers']}")
            print(f"  - 예상 성능: {config['expected_performance']}")
            print(f"  - 비용 효율성: {config['cost_efficiency']}")
        
        print(f"\n중요 사항:")
        print(f"  - L4 GPU 메모리 제한(22GB)으로 인해 워커 수가 제한됨")
        print(f"  - 각 워커당 약 2-3GB GPU 메모리 사용")
        print(f"  - CPU 코어 수보다 GPU 메모리가 병목이 될 수 있음")
        print(f"  - 권장: g2-standard-8 또는 g2-standard-16")
        
        return recommendations

    def create_environment(self):
        """마리오 환경 생성"""
        env_id = f"SuperMarioBros-{self.world}-{self.stage}-v0"
        
        if gym.__version__ < '0.26':
            env = gym_super_mario_bros.make(env_id, new_step_api=True)
        else:
            env = gym_super_mario_bros.make(env_id, render_mode=self.render_mode, apply_api_compatibility=True)
        
        # 액션 공간 제한
        env = JoypadSpace(env, [["right"], ["right", "A"]])
        
        # 환경 래퍼 적용
        env = SkipFrame(env, skip=self.frame_skip)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=self.resize_shape)
        
        if gym.__version__ < '0.26':
            env = FrameStack(env, num_stack=4, new_step_api=True)
        else:
            env = FrameStack(env, num_stack=4)
            
        return env

    def worker_process(self, worker_id, result_queue, save_dir, tensorboard_dir):
        """
        개별 워커 프로세스에서 실행되는 학습 함수
        
        Args:
            worker_id: 워커 ID
            result_queue: 결과를 전송할 큐
            save_dir: 체크포인트 저장 디렉토리
            tensorboard_dir: TensorBoard 로그 디렉토리
        """
        print(f"워커 {worker_id}: 학습 시작 (성능 모드: {self.performance_mode})")
        
        try:
            # 환경 생성
            env = self.create_environment()
            
            # 워커별 저장 디렉토리 생성
            worker_save_dir = save_dir / f"worker_{worker_id}"
            worker_save_dir.mkdir(parents=True, exist_ok=True)
            
            # 워커별 TensorBoard 로거 생성
            worker_tb_dir = tensorboard_dir / f"worker_{worker_id}"
            worker_writer = SummaryWriter(log_dir=str(worker_tb_dir))
            
            # 마리오 에이전트 생성 (성능 모드 전달)
            mario = Mario(
                state_dim=self.state_dim,
                action_dim=env.action_space.n,
                save_dir=worker_save_dir,
                performance_mode=self.performance_mode  # 성능 모드 전달
            )
            
            # 모델 구조를 TensorBoard에 기록
            try:
                dummy_input = torch.zeros((1, 4, 84, 84)).to(mario.device)
                worker_writer.add_graph(mario.net, dummy_input)
                
                # 모델 파라미터 정보 기록
                total_params = sum(p.numel() for p in mario.net.parameters())
                trainable_params = sum(p.numel() for p in mario.net.parameters() if p.requires_grad)
                worker_writer.add_text('Model/Info', 
                                     f'Worker {worker_id}\n'
                                     f'Total Parameters: {total_params:,}\n'
                                     f'Trainable Parameters: {trainable_params:,}')
            except Exception as e:
                print(f"워커 {worker_id}: 모델 그래프 기록 실패 - {e}")
            
            # 에피소드 실행
            for episode in range(self.episodes_per_worker):
                episode_reward = 0
                episode_length = 0
                episode_losses = []
                episode_q_values = []
                
                # 환경 리셋
                state = env.reset()
                
                while True:
                    # 액션 선택
                    action = mario.act(state)
                    
                    # 환경에서 액션 실행
                    next_state, reward, done, trunc, info = env.step(action)
                    
                    # 경험 저장
                    mario.cache(state, next_state, action, reward, done, info)
                    
                    # 학습
                    q, loss = mario.learn()
                    
                    # 메트릭 수집
                    if loss is not None:
                        episode_losses.append(loss)
                    if q is not None:
                        episode_q_values.append(q)
                    
                    # 상태 업데이트
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                    
                    # 에피소드 종료 조건
                    if done or info.get("flag_get", False):
                        break
                
                # 에피소드 보상을 mario.ep_rewards에 추가
                mario.ep_rewards.append(episode_reward)
                
                # 에피소드별 메트릭을 TensorBoard에 기록
                global_step = worker_id * self.episodes_per_worker + episode
                
                try:
                    # 기본 메트릭
                    worker_writer.add_scalar('Episode/Reward', float(episode_reward), global_step)
                    worker_writer.add_scalar('Episode/Length', int(episode_length), global_step)
                    worker_writer.add_scalar('Episode/Epsilon', float(mario.exploration_rate), global_step)
                    worker_writer.add_scalar('Episode/Steps', int(mario.curr_step), global_step)
                    
                    # 학습 메트릭
                    if episode_losses:
                        avg_loss = float(np.mean(episode_losses))
                        worker_writer.add_scalar('Episode/AvgLoss', avg_loss, global_step)
                    
                    if episode_q_values:
                        avg_q = float(np.mean(episode_q_values))
                        worker_writer.add_scalar('Episode/AvgQValue', avg_q, global_step)
                    
                    # 게임 정보 기록
                    worker_writer.add_scalar('Game/XPosition', float(info.get('x_pos', 0)), global_step)
                    worker_writer.add_scalar('Game/YPosition', float(info.get('y_pos', 0)), global_step)
                    worker_writer.add_scalar('Game/Coins', int(info.get('coins', 0)), global_step)
                    worker_writer.add_scalar('Game/Life', int(info.get('life', 0)), global_step)
                    worker_writer.add_scalar('Game/FlagGet', int(info.get('flag_get', False)), global_step)
                    
                    # 모델 파라미터와 그래디언트 상세 분석 (10 에피소드마다로 변경)
                    if episode % 10 == 0:
                        total_grad_norm = 0.0
                        total_param_norm = 0.0
                        layer_count = 0
                        
                        for name, param in mario.net.named_parameters():
                            if param.requires_grad:
                                # 파라미터 통계 (GPU에서 직접 계산)
                                param_norm = float(param.data.norm())
                                worker_writer.add_scalar(f'Parameters/{name}/Norm', param_norm, global_step)
                                total_param_norm += param_norm ** 2
                                
                                # 그래디언트 통계 (그래디언트가 있는 경우)
                                if param.grad is not None:
                                    grad_norm = float(param.grad.norm())
                                    worker_writer.add_scalar(f'Gradients/{name}/Norm', grad_norm, global_step)
                                    total_grad_norm += grad_norm ** 2
                                
                                layer_count += 1
                        
                        # 전체 그래디언트 및 파라미터 노름
                        worker_writer.add_scalar('Model/TotalGradientNorm', float(total_grad_norm ** 0.5), global_step)
                        worker_writer.add_scalar('Model/TotalParameterNorm', float(total_param_norm ** 0.5), global_step)
                        worker_writer.add_scalar('Model/LayerCount', layer_count, global_step)
                    
                    # 모델 파라미터 히스토그램 기록 (20 에피소드마다로 변경)
                    if episode % 20 == 0:
                        for name, param in mario.net.named_parameters():
                            if param.grad is not None:
                                worker_writer.add_histogram(f'Gradients/{name}', param.grad.cpu(), global_step)
                            worker_writer.add_histogram(f'Weights/{name}', param.data.cpu(), global_step)
                        
                        # 학습률 기록
                        current_lr = mario.optimizer.param_groups[0]['lr']
                        worker_writer.add_scalar('Training/LearningRate', float(current_lr), global_step)
                        
                        # 옵티마이저 상태 기록 (Adam의 경우)
                        if hasattr(mario.optimizer, 'state') and mario.optimizer.state:
                            for group_idx, group in enumerate(mario.optimizer.param_groups):
                                for param_idx, param in enumerate(group['params']):
                                    if param in mario.optimizer.state:
                                        state = mario.optimizer.state[param]
                                        if 'exp_avg' in state:
                                            exp_avg_norm = state['exp_avg'].norm().item()
                                            worker_writer.add_scalar(f'Optimizer/ExpAvgNorm_Group{group_idx}_Param{param_idx}', 
                                                                   float(exp_avg_norm), global_step)
                                        if 'exp_avg_sq' in state:
                                            exp_avg_sq_norm = state['exp_avg_sq'].norm().item()
                                            worker_writer.add_scalar(f'Optimizer/ExpAvgSqNorm_Group{group_idx}_Param{param_idx}', 
                                                                   float(exp_avg_sq_norm), global_step)
                    
                    # 네트워크 레이어별 활성화 통계 (10 에피소드마다)
                    if episode % 10 == 0:
                        # 온라인 네트워크와 타겟 네트워크 파라미터 차이
                        param_diff_norm = 0.0
                        for online_param, target_param in zip(mario.net.online.parameters(), mario.net.target.parameters()):
                            diff = (online_param.data - target_param.data).norm().item()
                            param_diff_norm += diff ** 2
                        
                        worker_writer.add_scalar('Model/OnlineTargetParamDiff', float(param_diff_norm ** 0.5), global_step)
                        
                        # 메모리 사용량 기록
                        if torch.cuda.is_available():
                            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                            worker_writer.add_scalar('System/GPU_Memory_Allocated_GB', memory_allocated, global_step)
                            worker_writer.add_scalar('System/GPU_Memory_Reserved_GB', memory_reserved, global_step)
                    
                    # 플러시하여 즉시 기록
                    worker_writer.flush()
                    
                except Exception as e:
                    print(f"워커 {worker_id}: TensorBoard 기록 실패 - {e}")
                    import traceback
                    traceback.print_exc()
                
                # 주기적 메모리 정리
                if episode % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                
                # 결과 전송 (TensorBoard 메트릭 포함)
                result_data = {
                    'worker_id': worker_id,
                    'episode': episode,
                    'reward': float(episode_reward),
                    'length': int(episode_length),
                    'steps': int(mario.curr_step),
                    'epsilon': float(mario.exploration_rate),
                    'flag_get': bool(info.get("flag_get", False)),
                    'avg_loss': float(np.mean(episode_losses)) if episode_losses else 0.0,
                    'avg_q': float(np.mean(episode_q_values)) if episode_q_values else 0.0,
                    'x_pos': float(info.get('x_pos', 0)),
                    'coins': int(info.get('coins', 0)),
                    'life': int(info.get('life', 0))
                }
                result_queue.put(result_data)
                
                # 진행 상황 출력
                if (episode + 1) % 10 == 0:
                    print(f"워커 {worker_id}: {episode + 1}/{self.episodes_per_worker} 에피소드 완료 "
                          f"(보상: {episode_reward:.2f}, 길이: {episode_length}, 스텝: {mario.curr_step})")
            
            # 워커별 TensorBoard 로거 종료
            worker_writer.close()
            env.close()
            print(f"워커 {worker_id}: 학습 완료")
            
        except Exception as e:
            print(f"워커 {worker_id} 오류: {e}")
            import traceback
            traceback.print_exc()
            
            # 오류 정보를 큐에 전송
            try:
                result_queue.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'completed_episodes': episode if 'episode' in locals() else 0,
                    'traceback': traceback.format_exc()
                })
            except Exception as queue_error:
                print(f"워커 {worker_id}: 오류 정보 전송 실패 - {queue_error}")

    def train(self):
        """병렬 학습 실행"""
        print("병렬 학습 시작")
        
        # 저장 디렉토리 생성
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir = Path("checkpoints_parallel") / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard 디렉토리 생성
        tensorboard_dir = Path("runs_parallel") / timestamp
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # 메인 TensorBoard 로거 생성
        main_writer = SummaryWriter(log_dir=str(tensorboard_dir / "main"))
        
        # 로깅을 위한 더미 환경 및 에이전트 생성
        temp_env = self.create_environment()
        dummy_mario = Mario(
            state_dim=self.state_dim,
            action_dim=temp_env.action_space.n,
            save_dir=save_dir / "main"
        )
        temp_env.close()
        
        # 메인 TensorBoard에 모델 그래프 기록
        try:
            dummy_input = torch.zeros((1, 4, 84, 84)).to(dummy_mario.device)
            main_writer.add_graph(dummy_mario.net, dummy_input)
            
            # 모델 구조 상세 정보 기록
            total_params = sum(p.numel() for p in dummy_mario.net.parameters())
            trainable_params = sum(p.numel() for p in dummy_mario.net.parameters() if p.requires_grad)
            
            # 온라인 네트워크와 타겟 네트워크 파라미터 수 분석
            online_params = sum(p.numel() for p in dummy_mario.net.online.parameters())
            target_params = sum(p.numel() for p in dummy_mario.net.target.parameters())
            
            model_info = f"""
                    ## 모델 구조 정보

                    ### 전체 네트워크 파라미터 수        
                        - **총 파라미터 수**: {total_params:,}
                        - **학습 가능한 파라미터 수**: {trainable_params:,}
                        - **고정된 파라미터 수**: {total_params - trainable_params:,}

                    ## 네트워크 구성
                    - **온라인 네트워크 파라미터**: {online_params:,}
                    - **타겟 네트워크 파라미터**: {target_params:,}

                    ## 레이어별 파라미터 수
                    """
            
            # 레이어별 파라미터 수 계산
            for name, param in dummy_mario.net.named_parameters():
                model_info += f"- **{name}**: {param.numel():,} parameters (shape: {list(param.shape)})\n"
            
            # 모델 구조 정보를 텍스트로 기록
            main_writer.add_text('Model/Architecture', model_info)
            
            # 모델 크기 정보를 스칼라로 기록
            main_writer.add_scalar('Model/TotalParameters', total_params, 0)
            main_writer.add_scalar('Model/TrainableParameters', trainable_params, 0)
            main_writer.add_scalar('Model/OnlineNetworkParameters', online_params, 0)
            main_writer.add_scalar('Model/TargetNetworkParameters', target_params, 0)
            
            # 모델 메모리 사용량 추정 (MB)
            model_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
            main_writer.add_scalar('Model/EstimatedSizeMB', model_size_mb, 0)
            
            print(f"모델 그래프 및 구조 정보가 TensorBoard에 기록되었습니다.")
            print(f"총 파라미터 수: {total_params:,}")
            print(f"추정 모델 크기: {model_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"메인 모델 그래프 기록 실패: {e}")
            import traceback
            traceback.print_exc()
        
        # 하이퍼파라미터를 TensorBoard에 기록
        hparams = {
            'num_workers': self.num_workers,
            'episodes_per_worker': self.episodes_per_worker,
            'world': self.world,
            'stage': self.stage,
            'frame_skip': self.frame_skip,
            'resize_shape': self.resize_shape,
            'exploration_rate_decay': dummy_mario.exploration_rate_decay,
            'gamma': dummy_mario.gamma,
            'batch_size': dummy_mario.batch_size,
            'learning_rate': dummy_mario.optimizer.param_groups[0]['lr']
        }
        
        main_writer.add_hparams(hparams, {})
        
        # 결과 큐 생성
        result_queue = Queue()
        
        # 워커 프로세스 시작
        processes = []
        for i in range(self.num_workers):
            p = Process(
                target=self.worker_process,
                args=(i, result_queue, save_dir, tensorboard_dir)
            )
            p.start()
            processes.append(p)
        
        # 결과 수집
        total_episodes = self.num_workers * self.episodes_per_worker
        completed_episodes = 0
        all_rewards = []
        all_lengths = []
        all_losses = []
        all_q_values = []
        success_count = 0
        worker_stats = {i: {'rewards': [], 'episodes': 0} for i in range(self.num_workers)}
        
        start_time = time.time()
        
        while completed_episodes < total_episodes:
            try:
                # 결과 대기 (타임아웃 120초로 증가)
                result = result_queue.get(timeout=120)
                
                # 오류 처리
                if 'error' in result:
                    print(f"워커 {result['worker_id']} 오류: {result['error']}")
                    # 오류가 발생한 워커의 에피소드 수를 차감하여 무한 대기 방지
                    total_episodes -= (self.episodes_per_worker - result.get('completed_episodes', 0))
                    continue
                
                # 메인 TensorBoard에 통합 메트릭 기록
                try:
                    main_writer.add_scalar('Aggregate/Reward', result['reward'], completed_episodes)
                    main_writer.add_scalar('Aggregate/Length', result['length'], completed_episodes)
                    main_writer.add_scalar('Aggregate/Epsilon', result['epsilon'], completed_episodes)
                    main_writer.add_scalar('Aggregate/Steps', result['steps'], completed_episodes)
                    
                    if result.get('avg_loss', 0) > 0:
                        main_writer.add_scalar('Aggregate/Loss', result['avg_loss'], completed_episodes)
                    if result.get('avg_q', 0) > 0:
                        main_writer.add_scalar('Aggregate/QValue', result['avg_q'], completed_episodes)
                    
                    # 워커별 통계 기록
                    worker_id = result['worker_id']
                    worker_stats[worker_id]['rewards'].append(result['reward'])
                    worker_stats[worker_id]['episodes'] += 1
                    
                    main_writer.add_scalar(f'Workers/Worker_{worker_id}_Reward', result['reward'], completed_episodes)
                    main_writer.add_scalar(f'Workers/Worker_{worker_id}_AvgReward', 
                                         np.mean(worker_stats[worker_id]['rewards'][-10:]), completed_episodes)
                    
                    main_writer.flush()
                    
                except Exception as e:
                    print(f"메인 TensorBoard 기록 실패: {e}")
                
                # 통계 업데이트
                all_rewards.append(result['reward'])
                all_lengths.append(result['length'])
                if result.get('avg_loss', 0) > 0:
                    all_losses.append(result['avg_loss'])
                if result.get('avg_q', 0) > 0:
                    all_q_values.append(result['avg_q'])
                
                if result.get('flag_get', False):
                    success_count += 1
                
                completed_episodes += 1
                
                # 진행 상황 출력 및 TensorBoard 기록
                if completed_episodes % 20 == 0:
                    elapsed_time = time.time() - start_time
                    avg_reward = np.mean(all_rewards[-100:])
                    avg_length = np.mean(all_lengths[-100:])
                    success_rate = success_count / completed_episodes * 100
                    
                    # 통합 통계를 TensorBoard에 기록
                    try:
                        main_writer.add_scalar('Progress/AvgReward_100', avg_reward, completed_episodes)
                        main_writer.add_scalar('Progress/AvgLength_100', avg_length, completed_episodes)
                        main_writer.add_scalar('Progress/SuccessRate', success_rate, completed_episodes)
                        main_writer.add_scalar('Progress/ElapsedTime', elapsed_time, completed_episodes)
                        
                        if all_losses:
                            avg_loss = np.mean(all_losses[-100:])
                            main_writer.add_scalar('Progress/AvgLoss_100', avg_loss, completed_episodes)
                        
                        if all_q_values:
                            avg_q = np.mean(all_q_values[-100:])
                            main_writer.add_scalar('Progress/AvgQValue_100', avg_q, completed_episodes)
                        
                        main_writer.flush()
                    except Exception as e:
                        print(f"진행 상황 TensorBoard 기록 실패: {e}")
                    
                    print(f"진행 상황: {completed_episodes}/{total_episodes} "
                          f"(평균 보상: {avg_reward:.2f}, 성공률: {success_rate:.1f}%, "
                          f"경과 시간: {elapsed_time:.1f}초)")
                
            except Exception as e:
                print(f"결과 수집 중 오류: {e}")
                print(f"오류 타입: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                
                # 프로세스 상태 확인
                alive_processes = [p.is_alive() for p in processes]
                print(f"살아있는 프로세스: {sum(alive_processes)}/{len(processes)}")
                
                if not any(alive_processes):
                    print("모든 워커 프로세스가 종료됨")
                    break
                
                # 큐가 비어있는지 확인
                try:
                    queue_size = result_queue.qsize()
                    print(f"큐 크기: {queue_size}")
                except:
                    print("큐 크기 확인 불가")
                
                # 타임아웃이 계속 발생하면 중단
                if "timeout" in str(e).lower():
                    print("타임아웃이 계속 발생하여 학습을 중단합니다.")
                    break
        
        # 프로세스 정리
        print("워커 프로세스 종료 대기 중...")
        for p in processes:
            p.join(timeout=30)
            if p.is_alive():
                print(f"프로세스 {p.pid} 강제 종료")
                p.terminate()
                p.join()
        
        # 로거 종료
        main_writer.close()
        
        # 최종 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # 결과 요약
        total_time = time.time() - start_time
        final_success_rate = success_count / completed_episodes * 100 if completed_episodes > 0 else 0
        final_avg_reward = np.mean(all_rewards) if all_rewards else 0
        
        print("\n=== 병렬 학습 완료 ===")
        print(f"총 에피소드: {completed_episodes}")
        print(f"평균 보상: {final_avg_reward:.2f}")
        print(f"성공률: {final_success_rate:.1f}%")
        print(f"총 소요 시간: {total_time:.1f}초")
        print(f"저장 디렉토리: {save_dir}")
        print(f"TensorBoard 로그: {tensorboard_dir}")
        print(f"\nTensorBoard 실행 명령:")
        print(f"tensorboard --logdir {tensorboard_dir} --port 6006")


def main():
    """메인 함수"""
    # PyTorch CUDA 메모리 관리 최적화
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # multiprocessing 설정
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 이미 설정된 경우 무시
    
    # CPU 권장사항 출력
    ParallelTrainer.get_cpu_recommendations()
    
    print("\n=== 성능 모드 선택 ===")
    print("1. speed: 2개 워커 + GPU 메모리 (최고 속도)")
    print("2. balanced: 3개 워커 + GPU/CPU 메모리 혼합 (균형)")
    print("3. memory_safe: 5-6개 워커 + CPU 메모리 (안전)")
    
    # 성능 모드별 트레이너 생성
    performance_modes = ["speed", "balanced", "memory_safe"]
    
    print(f"\n권장: 16 vCPU 환경에서는 'speed' 모드 추천 (GPU 연산 최대 활용)")
    
    # 기본적으로 speed 모드 사용 (가장 빠른 성능)
    selected_mode = "speed"
    print(f"선택된 모드: {selected_mode}")
    
    # 병렬 트레이너 생성 및 실행
    trainer = ParallelTrainer(
        num_workers=8,  # 성능 모드에 따라 자동 설정
        episodes_per_worker=600,  # 워커 수가 적으므로 에피소드 수 증가
        performance_mode=selected_mode
    )
    
    trainer.train()


if __name__ == "__main__":
    main() 