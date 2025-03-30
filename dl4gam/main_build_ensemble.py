import argparse
import os
import subprocess
import time
from pathlib import Path

import pandas as pd


def run_task(task_index, cmd, gpu_id):
    # add the gpu id to the command
    cmd += f" --gpu_id={gpu_id}"
    env = os.environ.copy()
    start_time = time.time()
    new_process = subprocess.Popen(cmd, shell=True, env=env)
    # store process with additional info: gpu_id, task_index, command and start time
    running_processes.append((new_process, gpu_id, task_index, cmd, start_time))
    gpu_task_count[gpu_id] += 1
    return running_processes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=r'Train ensemble of models')
    parser.add_argument(r'--config_fp', type=str, required=True, help=r'Path to the config file')
    parser.add_argument(r'--n_splits', type=int, default=5, help=r'Number of cross-validation iterations')
    parser.add_argument(r'--ensemble_size', type=int, default=5, help=r'Number of models in the ensemble')
    parser.add_argument(r'--n_gpus', type=int, default=1, help=r'Number of GPUs available')
    parser.add_argument(
        r'--max_tasks_per_gpu_train', type=int, default=1, help=r'Maximum number of models per GPU while training'
    )
    parser.add_argument(
        r'--max_tasks_per_gpu_infer', type=int, default=1, help=r'Maximum number of models per GPU while inference'
    )
    parser.add_argument(r'--seed_base', type=int, default=1, help=r'Base seed for the ensemble')
    args = parser.parse_args()

    config_fp = args.config_fp
    n_splits = args.n_splits
    ensemble_size = args.ensemble_size
    n_gpus = args.n_gpus
    max_tasks_per_gpu_train = args.max_tasks_per_gpu_train
    max_tasks_per_gpu_infer = args.max_tasks_per_gpu_infer
    seed_base = args.seed_base

    # generate all the model training commands
    commands = []  # list of (command_str, is_train)
    for seed in range(seed_base, seed_base + ensemble_size):
        for i_split in range(1, n_splits + 1):
            cmd = f"python main_train.py --config_fp={config_fp} --split=split_{i_split} --seed={seed}"
            commands.append((cmd, True))  # True = training

    model_version = "version_0"
    for seed in range(seed_base, seed_base + ensemble_size):
        for i_split in range(1, n_splits + 1):
            for fold in ['s_valid', 's_test']:
                for sub_dir in ['inv', '2023']:
                    cmd = (
                        f"python main_test.py"
                        f" --checkpoint_dir=../data/external/_experiments/s2_alps_plus/unet/split_{i_split}/seed_{seed}/{model_version}/checkpoints"
                        f" --fold={fold}"
                        f" --test_per_glacier true"
                        f" --split_fp=../data/external/wd/s2_alps_plus/cv_split_outlines/map_all_splits_all_folds.csv"
                        f" --rasters_dir=../data/external/wd/s2_alps_plus/{sub_dir}/glacier_wide"
                    )
                    commands.append((cmd, False))  # False = inference

    print(f"Generated {len(commands)} commands")

    gpu_task_count = [0] * n_gpus
    running_processes = []
    finished_tasks = []

    task_index = 0
    total_tasks = len(commands)

    while task_index < total_tasks or len(running_processes) > 0:
        for gpu_id in range(n_gpus):
            while task_index < total_tasks:
                cmd, is_train = commands[task_index]
                max_tasks = args.max_tasks_per_gpu_train if is_train else args.max_tasks_per_gpu_infer

                if gpu_task_count[gpu_id] < max_tasks:
                    print(f"Launching task {task_index} ({cmd}) on GPU {gpu_id}")
                    running_processes = run_task(task_index, cmd, gpu_id)
                    task_index += 1
                    time.sleep(1)
                else:
                    break  # This GPU is full, move to next

        time.sleep(1)

        for process_tuple in running_processes[:]:
            process, gpu_id, i_task, cmd, start_time = process_tuple
            if process.poll() is not None:
                finish_time = time.time()
                duration = finish_time - start_time
                print(f"Task {i_task + 1} / {len(commands)} on GPU {gpu_id} completed in {duration / 3600:.2f} hours")
                gpu_task_count[gpu_id] -= 1
                running_processes.remove(process_tuple)
                finished_tasks.append((gpu_id, i_task, cmd, start_time, finish_time, duration))

    print("All tasks have been completed.")

    # Export finished task metadata
    cols = ["gpu_id", "i_task", "command", "start_time", "finish_time", "duration_sec"]
    df = pd.DataFrame(finished_tasks, columns=cols)
    df = df.sort_values(by="start_time")
    fn = f"task_times_models_{model_version}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    fp_out = Path(f'../data/external/_experiments/s2_alps_plus/unet/') / fn
    df.to_csv(fp_out, index=False)
    print(df)
    print(f"Task timing details exported to task_times.csv")
