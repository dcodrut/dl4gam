import argparse
import os
import subprocess
import time


def run_task(cmd, gpu_id):
    # add the gpu id to the command
    cmd += f" --gpu_id={gpu_id}"
    env = os.environ.copy()
    new_process = subprocess.Popen(cmd, shell=True, env=env)
    running_processes.append((new_process, gpu_id))
    gpu_task_count[gpu_id] += 1

    return running_processes


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ensemble of models')
    parser.add_argument('--config_fp', type=str, required=True, help='Path to the config file')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation iterations')
    parser.add_argument('--ensemble_size', type=int, default=5, help='Number of models in the ensemble')
    parser.add_argument('--n_gpus', type=int, default=1, help='Number of GPUs available')
    parser.add_argument('--max_tasks_per_gpu', type=int, default=1, help='Maximum number of models per GPU')
    parser.add_argument('--seed_base', type=int, default=1, help='Base seed for the ensemble')
    args = parser.parse_args()

    config_fp = args.config_fp
    n_splits = args.n_splits
    ensemble_size = args.ensemble_size
    n_gpus = args.n_gpus
    max_tasks_per_gpu = args.max_tasks_per_gpu
    seed_base = args.seed_base

    # generate all the commands
    commands = []
    for seed in range(seed_base, seed_base + ensemble_size):
        for i_split in range(1, n_splits + 1):
            # get the settings from the environment
            cmd = f"python main_train.py --config_fp={config_fp} --split=split_{i_split} --seed={seed}"
            commands.append(cmd)
    print(f"Generated {len(commands)} commands")

    # GPU task counters
    gpu_task_count = [0] * n_gpus
    running_processes = []

    # loop to manage and distribute tasks
    task_index = 0
    total_tasks = len(commands)

    while task_index < total_tasks or len(running_processes) > 0:
        # launch new tasks if any GPUs have available slots
        for gpu_id in range(n_gpus):
            while gpu_task_count[gpu_id] < max_tasks_per_gpu and task_index < total_tasks:
                print(f"Launching task {task_index} ({commands[task_index]}) on GPU {gpu_id}")
                running_processes = run_task(commands[task_index], gpu_id)
                task_index += 1
                time.sleep(1)

        time.sleep(1)

        # check for completed tasks
        for process, gpu_id in running_processes[:]:
            if process.poll() is not None:  # process has completed
                print(f"Task on GPU {gpu_id} completed")
                gpu_task_count[gpu_id] -= 1
                running_processes.remove((process, gpu_id))

    print("All tasks have been completed.")
