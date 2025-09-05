import os


def get_commands(idx, ckpt_name):
    cuda_idx = idx % 2
    return f"CUDA_VISIBLE_DEVICES={cuda_idx} python ./legged_gym/scripts/data_collector_runner.py \
  --task h1int --num_envs 100 --headless \
  --load_checkpoint /root/workspace/HugWBC/logs/h1_interrupt/Aug21_13-31-13_/{ckpt_name} \
  --num_total 2000 --switch_prob 1 --episode_length_s 4 --output_root {ckpt_name.split('.')[0]} --overwrite"

ckpt_paths = "logs/h1_interrupt/Aug21_13-31-13_"
ckpts = os.listdir(ckpt_paths)
ckpts = [ckptn for ckptn in ckpts if ckptn.endswith(".pt")]
ckpts = sorted(ckpts, key=lambda x: int(x.split("_")[1].split(".")[0]))
commands_list = ["" for _ in range(8)]
for idx, ckpt in enumerate(ckpts):
    commands_list[idx % 8] += get_commands(idx, ckpt) + "\n"

with open("commands_0.sh", "w") as f:
    f.write(commands_list[0])
with open("commands_1.sh", "w") as f:
    f.write(commands_list[1])
with open("commands_2.sh", "w") as f:
    f.write(commands_list[2])
with open("commands_3.sh", "w") as f:
    f.write(commands_list[3])
with open("commands_4.sh", "w") as f:
    f.write(commands_list[4])
with open("commands_5.sh", "w") as f:
    f.write(commands_list[5])
with open("commands_6.sh", "w") as f:
    f.write(commands_list[6])
with open("commands_7.sh", "w") as f:
    f.write(commands_list[7])