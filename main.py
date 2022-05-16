import random
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from restoration_of_fitness_function.linearQuadraticApproximations import randomArgs, fitness, behaviorStrategy, \
    macroparameterM1, macroparameterM2, macroparameterM3, macroparameterM4


def generate_data(size):
    # Генерация набора данных
    arrayOfA1 = []
    arrayOfB1 = []
    arrayOfA2 = []
    arrayOfB2 = []
    for i in range(0, size):
        a, b = randomArgs()
        arrayOfA1.append(a)
        arrayOfB1.append(b)
        a, b = randomArgs()
        arrayOfA2.append(a)
        arrayOfB2.append(b)
    fitnessArray = []
    pqrs = []
    for i in range(0, size):
        f, p, q, r, s = fitness(arrayOfA1[i], arrayOfB1[i], arrayOfA2[i], arrayOfB2[i])
        fitnessArray.append(f)
        pqrs.append([p, q, r, s])
    trueArrayOfA1 = []
    trueArrayOfB1 = []
    trueArrayOfA2 = []
    trueArrayOfB2 = []
    strategies = []
    trueFitnessArray = []
    truepqrs = []
    test = []
    m1 = []
    m2 = []
    m3 = []
    m4 = []
    m5 = []
    m6 = []
    m7 = []
    m8 = []
    for i in range(0, len(fitnessArray)):
        if fitnessArray[i] is not None:
            trueArrayOfA1.append(arrayOfA1[i])
            trueArrayOfB1.append(arrayOfB1[i])
            trueArrayOfA2.append(arrayOfA2[i])
            trueArrayOfB2.append(arrayOfB1[i])
            trueFitnessArray.append(fitnessArray[i])
            truepqrs.append(pqrs[i])
    for i in range(0, len(trueFitnessArray)):
        strategies.append((behaviorStrategy(trueArrayOfA1[i], trueArrayOfB1[i], i / 100), (behaviorStrategy(trueArrayOfA2[i], trueArrayOfB2[i], i / 100))))
        test.append([trueArrayOfA1, trueArrayOfB1, trueArrayOfA2, trueArrayOfB2])
        m1.append(macroparameterM1(trueArrayOfA1[i]))
        m2.append(macroparameterM2(trueArrayOfA1[i], trueArrayOfB1[i]))
        m3.append(macroparameterM3(trueArrayOfB1[i]))
        m4.append(macroparameterM4(trueArrayOfA1[i], trueArrayOfB1[i]))
        m5.append(macroparameterM1(trueArrayOfA2[i]))
        m6.append(macroparameterM2(trueArrayOfA2[i], trueArrayOfB2[i]))
        m7.append(macroparameterM3(trueArrayOfB2[i]))
        m8.append(macroparameterM4(trueArrayOfA2[i], trueArrayOfB2[i]))
    return strategies, m1, m2, m3, m4, m5, m6, m7, m8, fitnessArray, pqrs, test


def prep_data(strats, m1, m2, m3, m4, m5, m6, m7, m8, fitnesses):
    macroparams = []
    macroparams.extend(m1)
    macroparams.extend(m2)
    macroparams.extend(m3)
    macroparams.extend(m4)
    macroparams.extend(m5)
    macroparams.extend(m6)
    macroparams.extend(m7)
    macroparams.extend(m8)
    return strats, macroparams, fitnesses


def binary_classificator(strats, macroparams, fitnesses):
    macro = []
    targets = []
    global_fitches = []
    for i in range(len(strats)):
        local_macro = []
        for j in range(8):
            local_macro.append(macroparams[i*8+j])
        #print(local_macro)
        macro.append(local_macro)
        for j in range(0, len(strats)):
            if i != j:
                try:
                    if fitnesses[i] > fitnesses[j]:
                        targets.append(1)
                    else:
                        targets.append(-1)
                except:
                    continue
                fitches = []
                for k in range(8):
                    fitches.append(macroparams[8 * i + k] - macroparams[8 * j + k])
                    for l in range(8):
                        if k != l:
                            fitches.append(
                                macroparams[8 * i + k] * macroparams[8 * i + l] - macroparams[8 * j + k] * macroparams[
                                    8 * j + l])
                global_fitches.append(fitches)

    df_targets = pd.DataFrame(data=targets)
    df_fitches = pd.DataFrame(data=global_fitches)
    df_macro = pd.DataFrame(data=macro)
    data_frame = pd.concat([df_fitches, df_targets], ignore_index=True, axis=1)
    data_frame = pd.concat([data_frame, df_macro], ignore_index=True, axis=1)
    return data_frame


def do_lda_classification():
    ogo_size = 100
    data = generate_data(ogo_size)
    pqrs = data[10]
    test = data[11]
    prepared_data = prep_data(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9])
    classif = binary_classificator(prepared_data[0], prepared_data[1], prepared_data[2])

    test_size = int(np.round(len(classif) * 0.2, 0))
    x_train = classif[:-test_size].iloc[:, 0:64].values
    y_train = classif[:-test_size].iloc[:, 64].values
    x_test = classif[-test_size:].iloc[:, 0:64].values
    y_test = classif[-test_size:].iloc[:, 64].values

    macro = classif.iloc[:, 65:73].values
    # print(len(macro))
    # print(macro)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    lda = LDA()
    x_train = lda.fit_transform(x_train, y_train)
    x_test = lda.transform(x_test)

    coefs = lda.scalings_
    for i in range(len(coefs)):
        coefs[i] = coefs[i]/10000
    globalF = 0
    globalm18 = []
    for k in range(100):
        m18 = []
        F = 0
        for i in range(8):
            F += coefs[i * 8] * macro[k*64+i][i]
            m18.append(macro[k*64+i][i])
            for j in range(8):
                F += coefs[i * 8 + j] * macro[k*64+i][i] * macro[k*64+j][j]
        if F > globalF:
            globalF = F
            globalm18 = m18

    lda.fit(x_train, y_train)
    y_pred = lda.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    if 0 < globalF < 2:
        '''print(globalF)
        print(coefs)
        print(cm)'''
        accuracy = str(accuracy_score(y_test, y_pred))
        '''print(f'Accuracy: {accuracy}')
        print('--------------------------------------------------------------------------------------------------------------------')'''
        return globalF, coefs, cm, accuracy, pqrs, globalm18, test
    else:
        return -1, -1, -1, -1, -1, -1, -1


def do_svm_classification():
    ogo_size = 100
    data = generate_data(ogo_size)
    prepared_data = prep_data(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9])
    classif = binary_classificator(prepared_data[0], prepared_data[1], prepared_data[2])
    test_size = int(np.round(len(classif) * 0.2, 0))
    x_train = classif[:-test_size].iloc[:, 0:64].values
    y_train = classif[:-test_size].iloc[:, 64]
    x_test = classif[-test_size:].iloc[:, 0:64].values
    y_test = classif[-test_size:].iloc[:, 64]
    model = svm.SVC(decision_function_shape='ovo')
    model.fit(x_train, y_train)
    predictions_lin = model.predict(x_test)

    cm = confusion_matrix(y_test, predictions_lin)
    accuracy_lin = accuracy_score(y_test, predictions_lin)
    print(cm)
    print("Accuracy: " + str(accuracy_lin))
    return cm, str(accuracy_lin)


if __name__ == '__main__':
    max_globalF, max_coefs, max_cm, max_accuracy, max_pqrs, max_macro, max_test = [], [], [], [], [], [], []
    for i in range(1000):
        globalF, coefs, cm, accuracy, pqrs, macro, test = do_lda_classification()
        if globalF != -1:
            max_globalF.append(globalF)
            max_coefs.append(coefs)
            max_cm.append(cm)
            max_accuracy.append(accuracy)
            max_pqrs.append(pqrs)
            max_macro.append(macro)
            max_test.append(test)
    max_a = max(max_accuracy)
    ind = max_accuracy.index(max_a)
    print(max_a, max_coefs[max_accuracy.index(max_a)],
          max_cm[max_accuracy.index(max_a)], max_globalF[max_accuracy.index(max_a)])
    print(max_macro[max_accuracy.index(max_a)])
    print(max_pqrs[max_accuracy.index(max_a)])
    print(max_test[max_accuracy.index(max_a)])
    with open('data.txt', 'w') as file:
        for i in range(len(max_test[max_accuracy.index(max_a)])):
            file.write(max_test[max_accuracy.index(max_a)])

    '''max_cm, max_accuracy = [], []
    for i in range(100):
        a, b = do_svm_classification()
        max_cm.append(a)
        max_accuracy.append(b)
    print(max(max_accuracy), max_cm[max_accuracy.index(max(max_accuracy))])'''