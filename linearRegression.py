import csv
import numpy as np
from sklearn.linear_model import LinearRegression


def linearRegression():

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    
    linReg = LinearRegression().fit(X, y)
    print(y)
    print(linReg.predict([[1, 3], [2, 8]]))

    print(linReg.score([[1,3], [2,9]], [10, 21]))


linearRegression()

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