import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def calculateLinearRegression(playerTrainingData : list[list[float]], playerDraftPick : list[int]) -> LinearRegression:

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    # X = Array of arrays of data (will be height, weight, long jump dist, etc. in each inner array)
    # y = Array of predicted values (will be pick #)
    # LinearRegression().fit() = Calculating the actual linear regression line
    # .predict() = Uses linear regression line to predict y values using passed in X
    # .score() = Calculates R^2 given X and real y

    # X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = np.dot(X, np.array([1, 2])) + 3
    
    linReg = LinearRegression().fit(playerTrainingData, playerDraftPick)

    #print(playerDraftPick)
    #print(linReg.predict([[1, 3], [2, 8]]))

    #print(linReg.score([[1,3], [2,9]], [10, 21]))

    return linReg


# xAxisDataIndex : Index in array of parsed data that should be used as the x axis variable when displaying linear regression line
def plotLinearRegression(linearRegression : LinearRegression, playerTrainingData : list[list[float]], playerDraftPick : list[int], xAxisDataIndex : int):

    scatterXData : list[float] = []

    for data in playerTrainingData:
        scatterXData.append(data[xAxisDataIndex])

    lineYData : list[float] = []

    print(linearRegression.coef_)
    print(linearRegression.intercept_)

    #for i in scatterXData:
        #lineYData.append(linearRegression.coef_[xAxisDataIndex] * i + linearRegression.intercept_)

    # for i in range(len(scatterXData)):
    #     lineYData.append(linearRegression.predict([playerTrainingData[i]]))

    lineYData = linearRegression.predict(playerTrainingData)

    print(scatterXData)
    print(lineYData)

    plt.scatter(scatterXData, playerDraftPick)
    plt.plot(scatterXData, lineYData)
    plt.show()

    pass
    

testX = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
testY = np.dot(testX, np.array([1, 2])) + 3

linReg = calculateLinearRegression(testX, testY)
plotLinearRegression(linReg, testX, testY, 0)


# TODO: fix this to read in correct data
def readCSV(file_path: Path) -> tuple[list[float], list[float]]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        xs = []
        ys = []
        for row in reader:
            xs.append(float(row[0]))
            ys.append(float(row[1]))
        return (xs, ys)