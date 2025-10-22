import os


format2 = "python3 train_instantaneous.py -s %d -d G_%d -e G"

for i in range (770, 1070):
    command2 = format2 % (i, i)
    os.system(command2)