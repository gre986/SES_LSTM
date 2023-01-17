from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from numpy import genfromtxt
from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import numpy
import codecs
import csv
import random
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,auc,accuracy_score, RocCurveDisplay
from sklearn.metrics import precision_score,recall_score,classification_report
import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np
import socket
import pandas as pd
import ast
import json

import os
import io
import requests

# Assigning Data and variables
data = genfromtxt('data.csv', delimiter=',')


m =[data[i][-1] for i in range(1,len(data))]
Y = np_utils.to_categorical(m)

# classification instance
def get_sequence(steps,time):
    x = [data[index] for index in range((time*10)+1 ,(time*10)+10+1)]
    x = numpy.delete(x, (data.shape[1]-1), axis=1)
    x = numpy.array(x)
    y = [Y[index] for index in range((time*10) ,(time*10)+10)]
    y = numpy.array(y)
    X = x.reshape(1, steps, (data.shape[1]-1))
    y = y.reshape(1, steps, y.shape[1])
    return X, y

# LSTM training function
def trainLSTM(model, t_size):
    for epoch in range(0,t_size):
        X,y = get_sequence(10,epoch)
        model.fit(X, y,batch_size=1, verbose=0)
    return model

# LSTM prediction function
def testLSTM(model, start, end):
    pred_prob = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    yactual = []
    ypredicted = []
    ya = []
    yp = []
    
    for d in range(start,end):
        X,y = get_sequence(10,d)
        yhat = model.predict(X,verbose=0)[0]
        pred_prob = numpy.append(pred_prob, yhat, axis=0)
        for index1 in range(10):
            i = np.where(yhat[index1] == yhat[index1].max())
            hin = i[0]
            for index2 in range(6):
                if(index2==hin):
                    yhat[index1][index2]=1
                else:
                    yhat[index1][index2]=0
        j= yhat
        k= y[0]
        for index1 in range(10):
            ypredicted.append(j[index1])
            yactual.append(k[index1])
    
    for index1 in range(len(ypredicted)):
        if (ypredicted[index1][0]==1 and ypredicted[index1][1]==0 and ypredicted[index1][2]==0 and ypredicted[index1][3]==0 and ypredicted[index1][4]==0 and ypredicted[index1][5]==0):
            yp.append(0)
        if (ypredicted[index1][0]==0 and ypredicted[index1][1]==1 and ypredicted[index1][2]==0 and ypredicted[index1][3]==0 and ypredicted[index1][4]==0 and ypredicted[index1][5]==0):
            yp.append(1)
        if (ypredicted[index1][0]==0 and ypredicted[index1][1]==0 and ypredicted[index1][2]==1 and ypredicted[index1][3]==0 and ypredicted[index1][4]==0 and ypredicted[index1][5]==0):
            yp.append(2)
        if (ypredicted[index1][0]==0 and ypredicted[index1][1]==0 and ypredicted[index1][2]==0 and ypredicted[index1][3]==1 and ypredicted[index1][4]==0 and ypredicted[index1][5]==0):
            yp.append(3)
        if (ypredicted[index1][0]==0 and ypredicted[index1][1]==0 and ypredicted[index1][2]==0 and ypredicted[index1][3]==0 and ypredicted[index1][4]==1 and ypredicted[index1][5]==0):
            yp.append(4)
        if (ypredicted[index1][0]==0 and ypredicted[index1][1]==0 and ypredicted[index1][2]==0 and ypredicted[index1][3]==0 and ypredicted[index1][4]==0 and ypredicted[index1][5]==1):
            yp.append(5)

    for index1 in range(len(yactual)):
        if (yactual[index1][0]==1 and yactual[index1][1]==0 and yactual[index1][2]==0 and yactual[index1][3]==0 and yactual[index1][4]==0 and yactual[index1][5]==0):
            ya.append(0)
        if (yactual[index1][0]==0 and yactual[index1][1]==1 and yactual[index1][2]==0 and yactual[index1][3]==0 and yactual[index1][4]==0 and yactual[index1][5]==0):
            ya.append(1)
        if (yactual[index1][0]==0 and yactual[index1][1]==0 and yactual[index1][2]==1 and yactual[index1][3]==0 and yactual[index1][4]==0 and yactual[index1][5]==0):
            ya.append(2)
        if (yactual[index1][0]==0 and yactual[index1][1]==0 and yactual[index1][2]==0 and yactual[index1][3]==1 and yactual[index1][4]==0 and yactual[index1][5]==0):
            ya.append(3)
        if (yactual[index1][0]==0 and yactual[index1][1]==0 and yactual[index1][2]==0 and yactual[index1][3]==0 and yactual[index1][4]==1 and yactual[index1][5]==0):
            ya.append(4)   
        if (yactual[index1][0]==0 and yactual[index1][1]==0 and yactual[index1][2]==0 and yactual[index1][3]==0 and yactual[index1][4]==0 and yactual[index1][5]==1):
            ya.append(5)
                
    pred_prob = pred_prob[1:]
    
    return ya, yp, pred_prob

# DIY rudimentary fivefold cross validation
def fiveFold_CVal(size):
    s = int(size + (size*0.2))
    ds = list(range(s))
    avgAcc = 0
    model = None
    
    # Loop for testing 
    for i in range(5):
        ya = []
        yp = [] 
        pred_prob = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ypredicted =  []
        yactual = []
        
        # Rudimentary way to reinitalize LSTM model
        model = Sequential()
        model.add(LSTM(30,input_shape = (None, (data.shape[1]-1)),return_sequences=True))
        model.add(Dropout(0.25))
        model.add(Dense(6, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        
        testingSection = ds[ int((len(ds)/5*i)) : int((len(ds)/5*(i+1))) ]
        
        # Loop for training
        for j in range(5):
            if(j != i):
                trainingSections = ds[ int((len(ds)/5*j)) : int((len(ds)/5*(j+1))) ]
                for epoch in trainingSections:
                    X,y = get_sequence(10, epoch)
                    model.fit(X, y, batch_size=1, verbose=0)
                    
        ya, yp, pred_prob = testLSTM(model, testingSection[0], testingSection[-1])
        acc = accuracy_score(ya, yp)
        print(acc)
        avgAcc += acc
    
    return avgAcc/5


# Function for ROC plot calculations 
def plotROC(l, ya, yp, pred_prob):
    labelFPR = []
    labelFNR = []
    conffpr = []
    conftpr = []
    
    # Store ya and yp in other variables to convert into binary arrays
    bya = ya.copy()
    byp = yp.copy()
    
    if (l == 0):
        for i in range(len(bya)):
            if (bya[i] == 0):
                bya[i] = 1
            else:
                bya[i] = 0
        for j in range(len(byp)):
            if (byp[j] == 0):
                byp[j] = 1
            else:
                byp[j] = 0
    else:
        for i in range(len(bya)):
            if (bya[i] != 0 and bya[i] != l):
                bya[i] = 0
        for j in range(len(byp)):
            if (byp[j] != 0 and byp[j] != l):
                byp[j] = 0      
                
    # ROC statistics
    confidence = []
    if (l == 0):
        for ind in range(len(ya)):
            confidence.append(pred_prob[ind][0])
    else:
        for ind in range(len(ya)):
            confidence.append(pred_prob[ind][l])
    fpr, tpr, thresholds_keras = roc_curve(bya, confidence, pos_label=l if l > 0 else 1)
    roc_auc = auc(fpr, tpr)
    conffpr.append(fpr.tolist())
    conftpr.append(tpr.tolist())
    
    # Print out Label information
    ltrue = 0
    lActCount = 0
    lPredCount = 0
    for e in range(len(ya)):
        if(ya[e] == l):
            lActCount += 1
        if(yp[e] == l):
            lPredCount += 1
        if(ya[e] == l and yp[e] == l):
            ltrue += 1

    print("Actual number of label " + (str(l)) + " : " + str(lActCount))
    print("Number of times label " + (str(l)) + " was predicted: " + str(lPredCount))
    print("Correct predictions: " + str(ltrue))
    print("Label False Positives: " + str(lActCount - ltrue))
    print("Label False Negatives: " + str(lPredCount - ltrue))
    
    labelFPR.append(1 - ((lActCount - ltrue) / lActCount))

    if (lPredCount != 0):
        labelFNR.append(1 - ((lPredCount - ltrue) / lPredCount))
    else:
        labelFNR.append(0)
    print()
    
    return [labelFPR, labelFNR, conffpr, conftpr]


# Function for ROC plot calculations (binary classification)
def importantLabelROC(ya, yp, pred_prob):
    bya = ya.copy()
    byp = yp.copy()

    for i in range(len(bya)):
        if (bya[i] != 0 and bya[i] != 1):
            if (bya[i] == 2):
                bya[i] = 1
            else:
                bya[i] = 0
    for j in range(len(byp)):
        if (byp[j] != 0 and byp[j] != 1):
            if (byp[j] == 2):
                byp[j] = 1
            else:
                byp[j] = 0      

    confidence = []
    for ind in range(len(ya)):
        confidence.append(pred_prob[ind][1])
    fpr, tpr, thresholds_keras = roc_curve(bya, confidence, pos_label=1)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="Label 1/2 confidence")
    display.plot()
    plt.savefig('Label12Conf.png')
    plt.show()
    
    # Print out Label information
    ltrue = 0
    lActCount = 0
    lPredCount = 0
    for e in range(len(ya)):
        if(ya[e] == 1):
            lActCount += 1
        if(yp[e] == 1):
            lPredCount += 1
        if(ya[e] == 1 and yp[e] == 1):
            ltrue += 1

    print("Actual number of label: " + str(lActCount))
    print("Number of times label was predicted: " + str(lPredCount))
    print("Correct predictions: " + str(ltrue))
    print("Label False Positives: " + str(lActCount - ltrue))
    print("Label False Negatives: " + str(lPredCount - ltrue))


# Plot the relevant data 
def combineROC(instance, ya, yp, pred_prob):
    
    # [labelFPR, labelFNR, conffpr, conftpr]
    cumLabelData = [[], [], [], []]
    for i in range(0, 6):
        newLabelData = plotROC(i, ya, yp, pred_prob)
        for x in range(len(newLabelData)):
            cumLabelData[x].extend(newLabelData[x])
    
    confM = confusion_matrix(ya, yp)
    print(confM)
    
    for i in range(len(cumLabelData[3])):
        # roc_auc = auc(cumLabelData[2][i], cumLabelData[3][i])
        plt.plot(cumLabelData[2][i],cumLabelData[3][i], label="Label "+str(i))
    plt.xlabel('False Positive Rate', fontsize = 13)
    plt.ylabel('True Positive Rate', fontsize = 13)
    plt.legend() 
    plt.savefig('ROC_Curves' + str(instance) + '.png')
    plt.show()
    
    
    # set width of bar
    barWidth = 0.3
    fig = plt.subplots()

    # Set position of bar on X axis
    br1 = np.arange(6)
    br2 = [x + barWidth for x in br1]
    
    # Make the plot
    plt.bar(br1, cumLabelData[0], width = barWidth, edgecolor ='grey', label ='True Positive')
    plt.bar(br2, cumLabelData[1], width = barWidth, edgecolor ='grey', label ='True Negative')

    # Adding Xticks
    plt.xlabel('Labels', fontsize = 13)
    plt.ylabel('Score', fontsize = 13)
    plt.xticks([r + 0.15 for r in range(6)], ['L0', 'L1', 'L2', 'L3', 'L4', 'L5'])
    
    plt.ylim(0,1)
    
    plt.legend()
    plt.savefig('ConfMatrix'+str(instance)+'Bars.png')
    plt.show()


# Final classification report and accuracy
def LSTMReport(ya, yp):
    print(classification_report(ya, yp))
    print("Final accuracy: " + str(accuracy_score(ya,yp)))    


# --------------------------------------------------------------------------- #

# Neutral Model

# LSTM model initialization
Nmodel = Sequential()
Nmodel.add(LSTM(30, input_shape = (None, (data.shape[1]-1)),return_sequences=True))
Nmodel.add(Dropout(0.25))
Nmodel.add(Dense(6, activation='softmax'))
Nmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Training, testing, plotting
Nmodel = trainLSTM(Nmodel, 100) 
# testLSTM(534, 575)
ya, yp, pred_prob = testLSTM(Nmodel, 500, 600)
combineROC(1000, ya, yp, pred_prob)
LSTMReport(ya, yp)


# --------------------------------------------------------------------------- #

# Base Model

# # LSTM model initialization
# Bmodel = Sequential()
# Bmodel.add(LSTM(30,input_shape = (None, (data.shape[1]-1)),return_sequences=True))
# Bmodel.add(Dropout(0.25))
# Bmodel.add(Dense(6, activation='softmax'))
# Bmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# # Training, testing, plotting
# Bmodel = trainLSTM(Bmodel, 50) 
# ya, yp, pred_prob = testLSTM(Bmodel, 100, 150)
# combineROC(500, ya, yp, pred_prob)
# LSTMReport(ya, yp)
# importantLabelROC(ya, yp, pred_prob)


# --------------------------------------------------------------------------- #

# Heavy Model

# # LSTM model initialization
# Hmodel = Sequential()
# Hmodel.add(LSTM(30,input_shape = (None, (data.shape[1]-1)),return_sequences=True))
# Hmodel.add(Dropout(0.25))
# Hmodel.add(Dense(6, activation='sigmoid'))
# Hmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# # Training, testing, plotting
# Hmodel = trainLSTM(Hmodel, 250) 
# ya, yp, pred_prob = testLSTM(Hmodel, 300, 600)
# combineROC(2500, ya, yp, pred_prob)
# LSTMReport(ya, yp)


# --------------------------------------------------------------------------- #

# Direct Testing

# test_data = data[-1][:36]
# test_data2 = np.array(test_data)
# test_data3 = test_data2.reshape(1, 1, 36)

# resultnewTest = model.predict(test_data3, verbose=0)[0]

# print(resultnewTest)


# for index1 in range(1):
#     i = np.where(resultnewTest[index1] == resultnewTest[index1].max())
#     hin = i[0]
#     for index2 in range(6):
#         if(index2==hin):
#             resultnewTest[index1][index2]=1
#         else:
#             resultnewTest[index1][index2]=0


# print(resultnewTest)

# test_data = data[7799][:36]
# test_data2 = np.array(test_data)
# test_data3 = test_data2.reshape(1, 1, 36)

# resultnewTest = model.predict(test_data3, verbose=0)[0]

# print(resultnewTest)


# for index1 in range(1):
#     i = np.where(resultnewTest[index1] == resultnewTest[index1].max())
#     hin = i[0]
#     for index2 in range(6):
#         if(index2==hin):
#             resultnewTest[index1][index2]=1
#         else:
#             resultnewTest[index1][index2]=0


# print(resultnewTest)


# --------------------------------------------------------------------------- #

# Five-fold classification (Neutral model)
valScore = fiveFold_CVal(100)

print("FiveFold Cross Validation: "+str(valScore))

# --------------------------------------------------------------------------- #

# Flask Connection

# app = Flask(__name__)
# api = Api(app)

# @app.route('/predict', methods=['POST'])
# def predict():
#     request_data = request.get_json()

#     test_data = [request_data['kurtosisx'],request_data['kurtosisy'],request_data['kurtosisz'],request_data['kurtosisf'],request_data['abs_kurtosisx'],request_data['abs_kurtosisy'],request_data['abs_kurtosisz'],request_data['abs_kurtosisf'],request_data['minx'],request_data['miny'],request_data['minz'],request_data['minf'],request_data['abs_minx'],request_data['abs_miny'],request_data['abs_minz'],request_data['abs_minf'],request_data['maxx'],request_data['maxy'],request_data['maxz'],request_data['maxf'],request_data['abs_maxx'],request_data['abs_maxy'],request_data['abs_maxz'],request_data['abs_maxf'],request_data['meanx'],request_data['meany'],request_data['meanz'],request_data['meanf'],request_data['abs_meanx'],request_data['abs_meany'],request_data['abs_meanz'],request_data['abs_meanf'],request_data['medianx'],request_data['mediany'],request_data['medianz'],request_data['medianf']]

#     #test_data = [3.496975571,5.23133881,2.969923024,1.403945156,2.586841992,3.263263203,3.875656459,1.427367267,-0.00310496,-0.008150457,-0.004602477,-0.008577343,-0.001675536,-0.005400046,-0.005581164,-0.003245549,0.993325865,0.991415445,0.996473748,0.998845225,0.996452155,0.995826012,0.994582555,0.995877992,0.679476704,0.227064595,0.321708889,0.436399674,0.683794901,0.228052776,0.326327732,0.439317916,0.697691169,0.190961651,0.31343034,0.423416975]
#     test_data2 = np.array(test_data)
#     test_data3 = test_data2.reshape(1, 1, 36)
#     # Change 'Nmodel' to used model name
#     resultnewTest = Nmodel.predict(test_data3, verbose=0)[0]
#     for index1 in range(1):
#         i = np.where(resultnewTest[index1] == resultnewTest[index1].max())
#         hin = i[0]
#         for index2 in range(6):
#             if(index2==hin):
#                 resultnewTest[index1][index2]=1
#             else:
#                 resultnewTest[index1][index2]=0
#     return {"t1": str(resultnewTest[0][0]), "t2": str(resultnewTest[0][1]), "t3": str(resultnewTest[0][2]), "t4": str(resultnewTest[0][3]), "t5": str(resultnewTest[0][4]), "t6": str(resultnewTest[0][5])}, 200

# if __name__ == "__main__":
#     app.run()























