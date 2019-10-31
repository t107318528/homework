import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing

trainPath = 'data/train-v3.csv'
validPath = 'data/valid-v3.csv'
testPath = 'data/test-v3.csv'

trainRawData = pd.read_csv(trainPath)
validRawData = pd.read_csv(validPath)
testRawData = pd.read_csv(testPath)


def getColValue(data, col):
    prices = data[col].values
    return prices


def getInputData(data, hasPrice=True):
    if hasPrice :
        data = data.drop(['price', 'id'], axis=1)
    else:
        data = data.drop(['id'], axis=1)
    dataSet = preprocessing.scale(data, with_mean=True, with_std=True, copy=True)
    return dataSet


trainPrice = getColValue(trainRawData, 'price')
trainData = getInputData(trainRawData)
validPrice = getColValue(validRawData, 'price')
validData = getInputData(validRawData)
testID = getColValue(testRawData, 'id')
testData = getInputData(testRawData, False)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_dim=trainData.shape[1]),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mae',
                  optimizer=optimizer)
    return model


model = build_model()

model.summary()

train_history = model.fit(x=trainData, y=trainPrice, validation_data=(validData, validPrice), epochs=200, batch_size=100)
predictPrices = model.predict(testData)


output = np.column_stack((testID, predictPrices))

np.savetxt('testPredict.csv', output, delimiter=',', header='id,price', comments='')

