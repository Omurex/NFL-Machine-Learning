import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from sklearn.linear_model import LinearRegression
import re
import random


# Read all the data from the CSV file and put into an array
# Returns list of each player's data used for prediction, and player pick number
def readCSV(file_path: Path) -> tuple[list[list[float]], list[int]]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        # for index, header in enumerate(headers):
        #     print(f"{header}: {index}")
        selected_cols = [4, 5, 6, 7, 8, 9, 10, 11, 13]
        pick_data_col = 12 # Column that pick data can be found in
        data = []

        player_data = [] # Holds return data for player stats
        pick_number = [] # Holds return for pick number

        for row in reader:
            pick_str = row[pick_data_col]
            
            if pick_str == "NA":
                continue

            # https://stackoverflow.com/questions/1450897/remove-characters-except-digits-from-string-using-python
            # Gets rid of all non-digits in the string
            # Example: Arizona Cardinals / 1st / 31st pick / 2009
            pick_str = pick_str.split("/")[2]
            pick_number.append(int(re.sub('\D', '', pick_str)))
                        
            row_vals = [row[col] for col in selected_cols]
            player_data.append(row_vals)


        return player_data, pick_number


# Delete any data points that have "NA" for any of the physical tests
# Completed with the help of ChatGPT
def remove_na_players(player_data, pick_data) -> list[list[float]]:
    filtered_player_data = [[float(val) for val in row] for row in player_data if "NA" not in row]
    filtered_pick_data = []

    for i in range(len(player_data)):
        row = player_data[i]
        if "NA" not in row:
            filtered_pick_data.append(pick_data[i])

    return filtered_player_data, filtered_pick_data


# Replace every instance of "NA" with a 0
# Completed with the help of ChatGPT
def replace_na_with_zero(data) -> list[list[float]]:
    filtered_data = [[float(0) if val == "NA" else float(val) for val in row] for row in data]
    return filtered_data


# Replace every instance of "NA" with the column average
# Completed with the help of ChatGPT
def replace_na_with_averages(no_na_data, data) -> list[list[float]]:
    col_avg = [sum(col) / len(col) for col in no_na_data]
    filtered_data = [[col_avg if val == "NA" else float(val) for val, col_avg in zip(row, col_avg)] for row in data]
    return filtered_data


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


# The problem with this function is that we are trying to make a straight line in two dimensions out of a straight line in 8 dimensions. So, not very useful as a visual
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
    

# testX = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# testY = np.dot(testX, np.array([1, 2])) + 3

# linReg = calculateLinearRegression(testX, testY)
# plotLinearRegression(linReg, testX, testY, 0)



if __name__ == '__main__':
    p = Path(__file__).with_name('NFL.csv')

    # Contains all data points, don't use
    full_player_data, pick_data = readCSV(p.absolute())

    joined_data = list(zip(full_player_data, pick_data))
    random.shuffle(joined_data)
    full_player_data, pick_data = zip(*joined_data)

    # Contains only the data points with every column completed
    no_na_player_data, no_na_pick_data = remove_na_players(full_player_data, pick_data)
    #print(no_na_player_data[:5])

    # Changed all "NA" data points to be 0
    na_to_0_data = replace_na_with_zero(full_player_data)
    #print(na_to_0_data[:5])

    # Changed all "NA" data points to be the column average
    na_to_avg_data = replace_na_with_averages(no_na_player_data, full_player_data)
    #print(na_to_avg_data[:5])


    train_data = no_na_player_data[:800]
    train_answers = no_na_pick_data[:800]

    test_data = no_na_player_data[800:]
    test_answers = no_na_pick_data[800:]

    # # train_data = na_to_0_data[:800]
    # # train_answers = pick_data[:800]

    # # test_data = na_to_0_data[800:]
    # # test_answers = pick_data[800:]

    # # train_data = na_to_avg_data[:1000]
    # # train_answers = pick_data[:1000]

    # # test_data = na_to_avg_data[1000:]
    # # test_answers = pick_data[1000:]

    # print(len(no_na_player_data))

    linReg = calculateLinearRegression(train_data, train_answers)
    print(linReg.score(test_data, test_answers))
    print(str(linReg.predict([test_data[0]])) + " : " + str(test_answers[0]))

    #linReg = calculateLinearRegression(no_na_player_data, no_na_pick_data)
    #plotLinearRegression(linReg, no_na_player_data, no_na_pick_data, 1)

    # testX = np.array([[1, 1], [1.5, 2], [2, 6], [5, 3]])
    # testY = np.dot(testX, np.array([1])) + 3
    # testY = [2, 3, 4, 5]

    # testLinReg = calculateLinearRegression(testX, testY)
    # plotLinearRegression(testLinReg, testX, testY, 0)
