#!/usr/bin/python3
#Project XO by Branson Camp, 12-12-2017

from PIL import Image
import SimpleNet as net
import os
from colorama import Fore, Back, Style

os.chdir(os.path.dirname(__file__))
dir = os.getcwd()

red = Fore.RED
green = Fore.GREEN;
blue = Fore.BLUE;
yellow = Fore.YELLOW;
cyan = Fore.CYAN;
magenta = Fore.MAGENTA;
white = Fore.WHITE;

def t():
    pic = []
    im = Image.open(dir + "/test.png")
    pix = im.load()

    for y in range(28):
        for x in range(28):
            pic.append(pix[x, y][0] / 255)
    net.test(pic)

pic = []
data = []
answer = []

X = ["x0.png", "x1.png", "x2.png", "x3.png", "x4.png", "x5.png", "x6.png", "x7.png", "x8.png", "x9.png", "x10.png", "x11.png", "x12.png", "x13.png", "x14.png", ]
O = ["o0.png", "o1.png", "o2.png", "o3.png", "o4.png", "o5.png", "o6.png", "o7.png", "o8.png", "o9.png", "o10.png", "o11.png", "o12.png", "o13.png", "o14.png", ]

for fname in X:
    im = Image.open(dir + "/TrainingImages/" + fname)
    pix = im.load()

    for y in range(28):
        for x in range(28):
            pic.append(pix[x, y][0] / 255)

    data.append(pic)
    answer.append(1)
    pic = []

for fname in O:
    im = Image.open(dir + "/TrainingImages/" + fname)
    pix = im.load()

    for y in range(28):
        for x in range(28):
            pic.append(pix[x, y][0] / 255)

    data.append(pic)
    answer.append(0)
    pic = []
net.data = data
net.answer = answer
net.train(20000, 1)

#input
choice = "";
while choice != "q":
    print(green + "[T] " + white + " Test with test.png");
    print(green + "[R] " + white + " Redo Traning Process");
    print(green + "[C] " + white + " Custom Train");
    print(green + "[E] " + white + " Edit Test Image");
    print(green + "[Q] " + white + " Quit");
    choice = input(green + "\n>> " + Fore.WHITE).lower();
    print();

    if choice == "t":
        t()
    elif choice == "e":
        print("Edit the image. Save and close when finished.")
        os.system("pinta $HOME/Documents/Python/XO/test.png")
    elif choice == "r":
        net.train(20000, 1)
    elif choice == "c":
        iterations = int(input("Training iterations (Default 20000): "))
        stepsize = int(input("Step Size (Default 1): "))
        print(green + "Training with", iterations, "Iterations and", stepsize, "Step Size" + white)
        net.train(iterations, stepsize)
