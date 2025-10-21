import os

format1 = "python3 gen.py --seed %d"

for i in range (770, 1070):
    command1 = format1 % i
    os.system(command1)