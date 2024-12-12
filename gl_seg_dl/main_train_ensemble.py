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
    num_gpus = 4  # number of GPUs available
    max_tasks_per_gpu = 2  # maximum number of tasks per GPU
    base_settings_file = './configs/unet.yaml'

    # generate all the commands
    ensemble_size = 10
    seed = 1
    commands = []
    for seed in range(seed, seed + ensemble_size):
        for i_split in range(1, 6):
            # get the settings from the environment
            cmd = f"python main_train.py --setting={base_settings_file} --split=split_{i_split} --seed={seed}"
            commands.append(cmd)
    print(f"Generated {len(commands)} commands")

    # GPU task counters
    gpu_task_count = [0] * num_gpus
    running_processes = []

    # loop to manage and distribute tasks
    task_index = 0
    total_tasks = len(commands)

    while task_index < total_tasks or len(running_processes) > 0:
        # launch new tasks if any GPUs have available slots
        for gpu_id in range(num_gpus):
            while gpu_task_count[gpu_id] < max_tasks_per_gpu and task_index < total_tasks:
                print(f"Launching task {task_index} ({commands[task_index]}) on GPU {gpu_id}")
                running_processes = run_task(commands[task_index], gpu_id)
                task_index += 1

                # give some time to the training to start (first one takes longer to start)
                time.sleep(300 if task_index == 1 else 10)

        # take a break
        time.sleep(1)

        # check for completed tasks
        for process, gpu_id in running_processes[:]:
            if process.poll() is not None:  # process has completed
                print(f"Task on GPU {gpu_id} completed")
                gpu_task_count[gpu_id] -= 1
                running_processes.remove((process, gpu_id))

    print("All tasks have been completed.")
