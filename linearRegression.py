import re
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Read all the data from the CSV file and put into an array
# Returns list of each player's data used for prediction, and player pick number
def read_csv(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        selected_cols = [4, 5, 6, 7, 8, 9, 10, 11, 13]
        pick_data_col = 12  # Column that pick data can be found in
        data = []

        player_data = []    # Holds return data for player stats
        pick_number = []    # Holds return for pick number

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

        return np.array(player_data), np.array(pick_number)


# Delete any data points that have "NA" for any of the physical tests
# Completed with the help of ChatGPT
def remove_na_players(player_data, pick_data) -> tuple[list[list[float]], list[int]]:
    filtered_player_data = [[float(val) for val in row] for row in player_data if "NA" not in row]
    filtered_pick_data = []

    for i in range(len(player_data)):
        row = player_data[i]
        if "NA" not in row:
            filtered_pick_data.append(pick_data[i])

    return filtered_player_data, filtered_pick_data


# Replace every instance of "NA" with the column average
# Completed with the help of ChatGPT
def replace_na_with_averages(no_na_data, data) -> list[list[float]]:
    col_avg = [sum(col) / len(col) for col in no_na_data]
    filtered_data = [[col_avg if val == "NA" else float(val) for val, col_avg in zip(row, col_avg)] for row in data]
    return filtered_data


def calculate_linear_regression(player_training_data: np.ndarray, player_draft_pick: np.ndarray) -> object:
    lin_reg = LinearRegression().fit(player_training_data, player_draft_pick)
    return lin_reg


# The problem with this function is that we are trying to make a straight line in two dimensions out of a straight line
    # in 8 dimensions. So, not very useful as a visual
# xAxisDataIndex : Index in array of parsed data that should be used as the x-axis variable when displaying linea
    # regression line
def plot_linear_regression(linear_regression: LinearRegression, player_training_data: list[list[float]],
                           player_draft_pick: list[int], x_axis_data_index: int):

    scatter_x_data: list[float] = []

    for data in player_training_data:
        scatter_x_data.append(data[x_axis_data_index])

    line_y_data: list[float] = []

    print(linear_regression.coef_)
    print(linear_regression.intercept_)

    # for i in scatterXData:
    #     lineYData.append(linearRegression.coef_[xAxisDataIndex] * i + linearRegression.intercept_)

    # for i in range(len(scatterXData)):
    #     lineYData.append(linearRegression.predict([playerTrainingData[i]]))

    line_y_data = linear_regression.predict(player_training_data)

    print(scatter_x_data)
    print(line_y_data)

    plt.scatter(scatter_x_data, player_draft_pick)
    plt.plot(scatter_x_data, line_y_data)
    plt.show()

    pass
    

# testX = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# testY = np.dot(testX, np.array([1, 2])) + 3

# linReg = calculateLinearRegression(testX, testY)
# plotLinearRegression(linReg, testX, testY, 0)


if __name__ == '__main__':
    p = Path(__file__).with_name('NFL.csv')

    # Contains all data points, don't use
    full_player_data, pick_data = read_csv(p.absolute())

    joined_data = list(zip(full_player_data, pick_data))
    random.shuffle(joined_data)
    full_player_data, pick_data = zip(*joined_data)

    # Contains only the data points with every column completed
    no_na_player_data, no_na_pick_data = remove_na_players(full_player_data, pick_data)

    # Changed all "NA" data points to be the column average
    na_to_avg_data = replace_na_with_averages(no_na_player_data, full_player_data)

    features = no_na_player_data
    target = no_na_pick_data

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    # linReg = calculate_linear_regression(train_data, train_answers)
    # print(linReg.score(test_data, test_answers))
    # print(str(linReg.predict([test_data[0]])) + " : " + str(test_answers[0]))

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_prediction = model.predict(x_test)
    linReg = calculate_linear_regression(x_train, y_train)
    r2_score = linReg.score(x_test, y_test)
    print("R-squared Score: ", r2_score)

    # linReg = calculateLinearRegression(no_na_player_data, no_na_pick_data)
    # plotLinearRegression(linReg, no_na_player_data, no_na_pick_data, 1)

    # testX = np.array([[1, 1], [1.5, 2], [2, 6], [5, 3]])
    # testY = np.dot(testX, np.array([1])) + 3
    # testY = [2, 3, 4, 5]

    # testLinReg = calculateLinearRegression(testX, testY)
    # plotLinearRegression(testLinReg, testX, testY, 0)
