import csv
import numpy as np
from pathlib import Path
from typing import List
from sklearn.linear_model import LinearRegression

def linearRegression():

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    linReg = LinearRegression().fit(X, y)
    print(y)
    print(linReg.predict([[1, 3], [2, 8]]))

    print(linReg.score([[1,3], [2,9]], [10, 21]))


# Read all the data from the CSV file and put into an array
def readCSV(file_path: Path) -> list[list]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        selected_cols = [4, 5, 6, 7, 8, 9, 10, 11]
        data = []

        for row in reader:
            row_vals = [row[col] for col in selected_cols]
            data.append(row_vals)
        return data


# Delete any data points that have "NA" for any of the physical tests
# Completed with the help of ChatGPT
def remove_na_players(data) -> list[list[float]]:
    filtered_data = [[float(val) for val in row] for row in data if "NA" not in row]
    return filtered_data


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


if __name__ == '__main__':
    p = Path(__file__).with_name('NFL.csv')

    # Contains all data points, don't use
    full_data = readCSV(p.absolute())

    # Contains only the data points with every column completed
    no_na_data = remove_na_players(full_data)
    print(no_na_data[:5])

    # Changed all "NA" data points to be 0
    na_to_0_data = replace_na_with_zero(full_data)
    print(na_to_0_data[:5])

    # Changed all "NA" data points to be the column average
    na_to_avg_data = replace_na_with_averages(no_na_data, full_data)
    print(na_to_avg_data[:5])

    linearRegression()
