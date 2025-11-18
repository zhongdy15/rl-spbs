import os
import time

algo_list = ["dqn_mask"] #["dqn_original"] #["dqn"]
seed_list = [0]
gpu_list = [5]
reward_mode = "Baseline_OCC_PPD_with_energy"
tradeoff_constant = 10

env_list = ["SemiPhysBuildingSim-v0"]
time_flag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

reward_mode_list = ["Baseline_without_energy",
                    "Baseline_with_energy",
                    "Baseline_OCC_PPD_without_energy",
                    "Baseline_OCC_PPD_with_energy",
                    ]


env_kwargs = f"reward_mode:\"'{reward_mode}'\" tradeoff_constant:{tradeoff_constant}"


# algo_filename = algo + "_" + reward_mode + \
#                 "_" + str(tradeoff_constant) + \
#                 "_" +time_flag

algo_num = len(algo_list)

assert len(gpu_list) == algo_num, "gpu_setting is incorrect"

for seed in seed_list:
    for algo_id in range(algo_num):
        algo_filename = algo_list[algo_id] + "_" + reward_mode + \
                        "_" + str(tradeoff_constant) + \
                        "_" + time_flag
        cmd_line = f"CUDA_VISIBLE_DEVICES={gpu_list[algo_id]} " \
                   f"python train.py " \
                   f" --algo {algo_list[algo_id]} " \
                   f" --env {env_list[0]} " \
                   f" --env-kwargs {env_kwargs} " \
                   f" --seed {seed} " \
                   f" -f logs/{algo_filename}/ " \
                   f" -tb logs/{algo_filename}/tb/ &"
        print(cmd_line)
        os.system(cmd_line)
        time.sleep(5)

# python rl_zoo3/enjoy.py --algo dqn_original --env SemiPhysBuildingSim-v0 --folder logs/dqn_original_2024-11-11-15-40-28//dqn_original/SemiPhysBuildingSim-v0_10/ -n 5000