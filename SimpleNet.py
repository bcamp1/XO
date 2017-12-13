import math
import random
from colorama import Fore, Back, Style
import sys
import os

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

b = 0
values = 28*28

def load_weights(weights):
    global w
    w = weights

def sig(x):
    return 1 / (1 + math.exp(-x))

def dsig(x):
    return sig(x) * (1 - sig(x))


def NN(x):
    global w
    global b
    global values
    z = 0
    for i in range(values):
        z += x[i] * w[i]
    z += b
    return z

def test(x, sigmoid=True):
    if sigmoid:
        output = sig(NN(x))
        if output  > 0.5:
            print("Looks like an " + Fore.GREEN + color.BOLD + "X" + color.END + Fore.WHITE + " to me! (I'm", Fore.GREEN + color.BOLD + str(int(100*output))  + color.END + Fore.WHITE,"% sure)")
        else:
            print("Looks like an " + Fore.GREEN + color.BOLD + "O" + color.END + Fore.WHITE + " to me! (I'm", Fore.GREEN + color.BOLD + str(int(100 - 100*output)) + color.END + Fore.WHITE,"% sure)")
    else:
        output = NN(x)
        print(definition[0],":",output)

# training loop
def train(iterations, learning_rate):
    global w
    global b
    global data
    global answer
    global values
    global parameters

    values = len(data[0])

    w = []
    for i in range(values):
        w.append(0)

    costs = []

    c = 0;
    for i in range(iterations):
        if (i + 1) % (iterations / 50) == 0:
            c += 1
            os.system('setterm -cursor off')
            sys.stdout.write(Fore.WHITE + Back.RESET + "\rTraining |" + Back.WHITE + " " * c + Back.BLACK + " " * (50 - c) + Back.RESET + "|")
            sys.stdout.flush()


        ri = random.randint(0, len(data) -1)
        point = data[ri]

        z = NN(point)
        pred = sig(z)
        target = answer[ri]
        cost = (pred - target) ** 2
        costs.append(cost)

        dcost_dpred = 2 * (pred - target)
        dpred_dz = dsig(z)

        dz_dw = []
        for i in range(values):
            dz_dw.append(point[i])
        dz_db = 1

        dcost_dz = dcost_dpred * dpred_dz

        dcost_dw = []
        for i in range(values):
            dcost_dw.append(dcost_dz * dz_dw[i])

        dcost_db = dcost_dz * dz_db

        for i in range(values):
            w[i] = w[i] - learning_rate * dcost_dw[i]
        b = b - learning_rate * dcost_db
    os.system('setterm -cursor on')
    print(Back.RESET + "\nFinal Cost: " + Fore.GREEN + color.BOLD + str(cost) + color.END)
    print();
    f = open("Weights.txt", "w")
    f.write(str(w))
    f.close()



############################################################################
data=[[3, 1, 7],
      [1, 0, 5],
      [6, 1, 8],
      [2, 2, 3],
      [4, 1, 7],
      [0, 0, 4],
      [7, 2, 9],
      [3, 2, 3]]

answer = [1,
          0,
          1,
          0,
          1,
          0,
          1,
          0]

definition = ["O", "X"]

############################################################################
