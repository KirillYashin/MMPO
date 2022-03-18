import math
import random

def trapezoidalIntegration(function, lower, upper, delta, argA, argB):
    sum = 0.5 * (function(argA, argB, lower) + function(argA, argB, upper))
    x = lower + delta
    while (x <= upper - delta):
        sum += function(argA, argB, x)
        x += delta
    return delta * sum


# Константы
minimumDepth = 40
optimumDepth = 80
maximumDepth = 140

firstSigma = 0.25
secondSigma = 0.003

adultAlpha = 0.006
juniorAlpha = 0.0016

adultBeta = 0.000000075
juniorBeta = 0.0000007

adultDelta = 0.00006
juniorDelta = 0.000016

adultGamma = 0.004
juniorGamma = 0.00008

firstEpsilon = 0.025
secondEpsilon = 0.025
thirdEpsilon = 0.2

# Функции окружиющей среды
def foodDistribution(argA, argB, argT):
    return firstSigma * (math.tanh(firstEpsilon * behaviorStrategy(argA, argB, argT)  + minimumDepth) + 1)  # -minimumDepth < arg < 0

def predatorDistribution(argA, argB, argT):
    return secondSigma * (math.tanh(secondEpsilon * behaviorStrategy(argA, argB, argT) + minimumDepth) + 1)

def predatorActivity(argT):
    return math.cos(2*math.pi*argT) + 1  # 0 < argT < 1

def additionalMortality(argA, argB, argT):
    return math.cosh(thirdEpsilon * (behaviorStrategy(argA, argB, argT) + optimumDepth))

# Функция стратегии поведения
def behaviorStrategy(argA, argB, argT):
    return argA + argB * math.cos(2*math.pi*argT)  # 0 < argT < 1

# Производная от функции стратегии поведения
def derivativeBehaviorStrategyInSecondDegree(argA, argB, argT):
    return (-2 * math.pi * argB * math.sin(2*math.pi*argT))**2  # 0 < argT < 1

# Комбинированная функция хищника
def combinedPredatorFunction(argA, argB, argT):
    return predatorActivity(argT) * predatorDistribution(argA, argB, argT)

# Функция генерации А(глубина) и В(амплитуда погружения)
def randomArgs():
    a = random.uniform((-maximumDepth + 1) / 2, -1)
    b = random.uniform((-maximumDepth + 1) / 2, -1)
    return a,b

# Функции макропараметров
def macroparameterM1(argA, argB):
    return trapezoidalIntegration(foodDistribution, 0, 1, 0.001, argA, argB)

def macroparameterM2(argA, argB):
    return - trapezoidalIntegration(combinedPredatorFunction, 0, 1, 0.001, argA, argB)

def macroparameterM3(argA, argB):
    return - trapezoidalIntegration(derivativeBehaviorStrategyInSecondDegree, 0, 1, 0.001, argA, argB)

def macroparameterM4(argA, argB):
    return - trapezoidalIntegration(additionalMortality, 0, 1, 0.001, argA, argB)

# Функция фитнеса
def fitness(argA1, argB1, argA2, argB2):
    p = juniorAlpha * macroparameterM1(argA1, argB1) + juniorBeta * macroparameterM3(0, argB1) + juniorDelta * macroparameterM4(argA1, argB1)
    q = juniorGamma * macroparameterM2(argA1, argB1)

    r = adultAlpha * macroparameterM1(argA2, argB2) + adultBeta * macroparameterM3(0, argB2) + adultDelta * macroparameterM4(argA2, argB2)
    s = adultGamma * macroparameterM2(argA2, argB2)

    if(4*r*p+(p+q-s)**2 > 0):
        return -s-p-q + math.sqrt( 4*r*p+(p+q-s)**2 )


arrayOfA1 = []
arrayOfB1 = []
arrayOfA2 = []
arrayOfB2 = []
for i in range(0,100):
    a, b = randomArgs()
    arrayOfA1.append(a)
    arrayOfB1.append(b)
    a, b = randomArgs()
    arrayOfA2.append(a)
    arrayOfB2.append(b)

fitnessArray = []
for i in range(0,100):
    fitnessArray.append(fitness(arrayOfA1[i], arrayOfB1[i], arrayOfA2[i], arrayOfB2[i]))

trueArrayOfA1 = []
trueArrayOfB1 = []
trueArrayOfA2 = []
trueArrayOfB2 = []
trueFitnessArray = []

for i in range(0, len(fitnessArray)):
    if(fitnessArray[i] != None):
        trueArrayOfA1.append(arrayOfA1[i])
        trueArrayOfB1.append(arrayOfB1[i])
        trueArrayOfA2.append(arrayOfA2[i])
        trueArrayOfB2.append(arrayOfB1[i])
        trueFitnessArray.append(fitnessArray[i])

for i in range(0, len(trueFitnessArray)):
    print(trueArrayOfA1[i], trueArrayOfB1[i], trueArrayOfA2[i], trueArrayOfB2[i], trueFitnessArray[i])
