import math
import random

# Константы
minimumDepth = 70
maximumDepth = 120

firstSigma = 1
secondSigma = 1

adultAlpha = 0.006
juniorAlpha = 0.0016

adultBeta = 0.000000075
juniorBeta = 0.0000007

adultDelta = 0.00006
juniorDelta = 0.000016

adultGamma = 0.004
juniorGamma = 0.00008


# Функции окружиющей среды
def foodDistribution(arg):
    return firstSigma * (arg + maximumDepth) # -maximumDepth < arg < 0


def predatorDistribution(arg):
    return secondSigma * (arg + maximumDepth) # -maximumDepth < arg < 0


def predatorActivity(argT):
    return math.cos(2*math.pi*argT) + 1 # 0 < arg < 1


def additionalMortality(arg):
    return (arg + minimumDepth)**2


# Функция стратегии поведения
def behaviorStrategy(argA, argB, argT):
    return argA + argB * math.cos(2*math.pi*argT)


# Функция генерации А(глубина) и В(амплитуда погружения)
def randomArgs():
    a = random.uniform((-maximumDepth + 1) / 2, -1)
    b = random.uniform((-maximumDepth + 1) / 2, -1)
    return a,b


# Функции макропараметров
def macroparameterM1(argA):
    return firstSigma * (argA + maximumDepth)


def macroparameterM2(argA, argB):
    return -secondSigma * (argA + maximumDepth + argB/2)


def macroparameterM3(argB):
    return -2*math.pi**2 * argB**2


def macroparameterM4(argA, argB):
    return -(argA + minimumDepth)**2 - argB**2/2


# Функция фитнеса
def fitness(argA1, argB1, argA2, argB2):
    p = juniorAlpha * macroparameterM1(argA1) + juniorBeta * macroparameterM3(argB1) + juniorDelta * macroparameterM4(argA1,argB1)
    q = juniorGamma * macroparameterM2(argA1,argB1)

    r = adultAlpha * macroparameterM1(argA2) + adultBeta * macroparameterM3(argB2) + adultDelta * macroparameterM4(argA2,argB2)
    s = adultGamma * macroparameterM2(argA2,argB2)

    return -s-p-q + math.sqrt( 4*r*p+(p+q-s)**2 ), p, q, r, s


arrayOfAdultA = []
arrayOfAdultB = []
arrayOfJuniorA = []
arrayOfJuniorB = []
for i in range(0,100000):
    a, b = randomArgs()
    arrayOfAdultA.append(a)
    arrayOfJuniorA.append(b)
    a, b = randomArgs()
    arrayOfAdultB.append(a)
    arrayOfJuniorB.append(b)

fitnessArray = []
for i in range(0,100000):
    fitnessArray.append(fitness(arrayOfJuniorA[i], arrayOfJuniorB[i], arrayOfAdultA[i], arrayOfAdultB[i]))

# print(fitnessArray)
