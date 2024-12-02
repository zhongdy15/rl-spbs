import os
import time


env_list = ["SemiPhysBuildingSim-v0"]
time_flag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
algo = "a2c"
algo_filename = algo + "_" + time_flag
seed_list = [0,10,20]

env_num = len(env_list)
gpu_list = [6]
assert len(gpu_list) == env_num, "gpu_setting is incorrect"

for seed in seed_list:
    for env_id in range(env_num):
        cmd_line = f"CUDA_VISIBLE_DEVICES={gpu_list[env_id]} " \
                   f"python train.py " \
                   f" --algo {algo} " \
                   f" --env {env_list[env_id]} " \
                   f" --seed {seed} " \
                   f" -f logs/{algo_filename}/ " \
                   f" -tb logs/{algo_filename}/tb/ &"
        print(cmd_line)
        os.system(cmd_line)
        time.sleep(5)

# python rl_zoo3/enjoy.py --algo dqn_original --env SemiPhysBuildingSim-v0 --folder logs/dqn_original_2024-11-11-15-40-28//dqn_original/SemiPhysBuildingSim-v0_10/ -n 5000