import os

seed_list = [770, 771,772,773,774]
data_list = [1,2,3,4,5]

# seed_list = [770]
# data_list = [1]


format = "CUDA_VISIBLE_DEVICES=2 python train_instantaneous.py -e A "

for seed in seed_list:
    for data in data_list:

        command = format + " -s " + str(seed) + " --data A_" + str(data)
        os.system(command)
        pass

    # python loop.py

    # CUDA_VISIBLE_DEVICES=3 python train_instantaneous.py -e A -s 770 -d A_1 