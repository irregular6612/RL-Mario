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
    def __init__(self, num_workers=None, episodes_per_worker=100):
        """
        병렬 학습을 위한 트레이너 클래스
        
        Args:
            num_workers: 워커 프로세스 수 (None이면 CPU 코어 수의 80% 사용)
            episodes_per_worker: 각 워커당 실행할 에피소드 수
        """
        self.num_workers = num_workers or max(1, int(mp.cpu_count() * 0.8))
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
        print(f"워커 {worker_id}: 학습 시작")
        
        try:
            # 환경 생성
            env = self.create_environment()
            
            # 워커별 저장 디렉토리 생성
            worker_save_dir = save_dir / f"worker_{worker_id}"
            worker_save_dir.mkdir(parents=True, exist_ok=True)
            
            # 워커별 TensorBoard 로거 생성
            worker_tb_dir = tensorboard_dir / f"worker_{worker_id}"
            worker_writer = SummaryWriter(log_dir=str(worker_tb_dir))
            
            # 마리오 에이전트 생성
            mario = Mario(
                state_dim=self.state_dim,
                action_dim=env.action_space.n,
                save_dir=worker_save_dir
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
                    
                    # 모델 파라미터와 그래디언트 상세 분석 (매 에피소드마다)
                    total_grad_norm = 0.0
                    total_param_norm = 0.0
                    layer_count = 0
                    
                    for name, param in mario.net.named_parameters():
                        if param.requires_grad:
                            # 파라미터 통계
                            param_data = param.data.cpu()
                            worker_writer.add_scalar(f'Parameters/{name}/Mean', float(param_data.mean()), global_step)
                            worker_writer.add_scalar(f'Parameters/{name}/Std', float(param_data.std()), global_step)
                            worker_writer.add_scalar(f'Parameters/{name}/Min', float(param_data.min()), global_step)
                            worker_writer.add_scalar(f'Parameters/{name}/Max', float(param_data.max()), global_step)
                            worker_writer.add_scalar(f'Parameters/{name}/Norm', float(param_data.norm()), global_step)
                            
                            total_param_norm += param_data.norm().item() ** 2
                            
                            # 그래디언트 통계 (그래디언트가 있는 경우)
                            if param.grad is not None:
                                grad_data = param.grad.cpu()
                                worker_writer.add_scalar(f'Gradients/{name}/Mean', float(grad_data.mean()), global_step)
                                worker_writer.add_scalar(f'Gradients/{name}/Std', float(grad_data.std()), global_step)
                                worker_writer.add_scalar(f'Gradients/{name}/Min', float(grad_data.min()), global_step)
                                worker_writer.add_scalar(f'Gradients/{name}/Max', float(grad_data.max()), global_step)
                                worker_writer.add_scalar(f'Gradients/{name}/Norm', float(grad_data.norm()), global_step)
                                
                                total_grad_norm += grad_data.norm().item() ** 2
                            
                            layer_count += 1
                    
                    # 전체 그래디언트 및 파라미터 노름
                    worker_writer.add_scalar('Model/TotalGradientNorm', float(total_grad_norm ** 0.5), global_step)
                    worker_writer.add_scalar('Model/TotalParameterNorm', float(total_param_norm ** 0.5), global_step)
                    worker_writer.add_scalar('Model/LayerCount', layer_count, global_step)
                    
                    # 모델 파라미터 히스토그램 기록 (5 에피소드마다)
                    if episode % 5 == 0:
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
            result_queue.put({
                'worker_id': worker_id,
                'error': str(e)
            })

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
# 모델 구조 정보

## 전체 네트워크
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
                # 결과 대기 (타임아웃 60초)
                result = result_queue.get(timeout=60)
                
                # 오류 처리
                if 'error' in result:
                    print(f"워커 {result['worker_id']} 오류: {result['error']}")
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
                # 프로세스 상태 확인
                alive_processes = [p.is_alive() for p in processes]
                if not any(alive_processes):
                    print("모든 워커 프로세스가 종료됨")
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
    # multiprocessing 설정
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 이미 설정된 경우 무시
    
    # 병렬 트레이너 생성 및 실행
    trainer = ParallelTrainer(
        num_workers=6,  # 테스트를 위해 워커 수 줄임
        episodes_per_worker=50  # 테스트를 위해 에피소드 수 줄임
    )
    
    trainer.train()


if __name__ == "__main__":
    main() 