import re
import csv
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



# Read all the data from the CSV file and put into an array
# Returns list of each player's data used for prediction, and player pick number
def read_csv(file_path: Path, position_filter="ALL") -> tuple[np.ndarray, np.ndarray]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        selected_cols = [5, 6, 7, 8, 9, 10, 11]
        pick_data_col = 12  # Column that pick data can be found in
        data = []

        player_data = []    # Holds return data for player stats
        pick_number = []    # Holds return for pick number

        for row in reader:
            pick_str = row[pick_data_col]

            if pick_str == "NA":
                continue

            if position_filter != "ALL" and row[16] != position_filter:
                continue

            # https://stackoverflow.com/questions/1450897/remove-characters-except-digits-from-string-using-python
            # Gets rid of all non-digits in the string
            # Example: Arizona Cardinals / 1st / 31st pick / 2009
            pick_str = pick_str.split("/")[2]
            pick_number.append(int(re.sub('\D', '', pick_str)))

            row_vals = [row[col] for col in selected_cols]
            player_data.append(row_vals)

        return np.array(player_data), np.array(pick_number)


def get_unique_positions(file_path):
    unique_positions = set()

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Read the header row to get column names

        # Assuming the "Position" column is the 17th column (0-based index)
        position_column_index = 16

        for row in reader:
            if len(row) > position_column_index:
                position = row[position_column_index].strip()
                if position:
                    unique_positions.add(position)

    return list(unique_positions)


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


def calculate_linear_regression(player_training_data: list[list[float]], player_draft_pick: list[int]) -> object:
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    lin_regression = LinearRegression().fit(player_training_data, player_draft_pick)
    return lin_regression


def calculate_random_forest_regression(player_training_data: list[list[float]],
                                       player_draft_pick: list[int]) -> RandomForestRegressor:
    """
    Train a random forest regression model to predict draft pick numbers based on player training data.
    """

    # Scikit Random Forest Regressor Documentation:
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=8)

    # Fit the regressor to the training data
    rf_regressor.fit(player_training_data, player_draft_pick)

    return rf_regressor


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


def plot_mae_by_position(positions, maes):
    plt.figure(figsize=(10, 6))
    plt.bar(positions, maes, color='blue')
    plt.xlabel('Position')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error by Position')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def run_random_forest(data_path, position_filter="ALL"):
    player_data, player_pick_data = read_csv(data_path.absolute(), position_filter=position_filter)
    player_joined_data = list(zip(player_data, player_pick_data))
    random.shuffle(player_joined_data)
    shuffled_player_data, shuffled_pick_data = zip(*player_joined_data)

    minimum_sample_size = 50

    # Contains only the data points with every column completed
    no_na_player_data, no_na_pick_data = remove_na_players(shuffled_player_data, shuffled_pick_data)

    player_avg_data = replace_na_with_averages(no_na_player_data, shuffled_player_data)

    features = player_avg_data
    target = shuffled_pick_data

    if len(features) < minimum_sample_size:
        print("------------------------")
        print(f'Cannot run analysis on {position_filter}, not enough data points ({len(features)})')
        return

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    rf_reg = calculate_random_forest_regression(x_train, y_train)
    predictions = rf_reg.predict(x_test)
    rf_score = rf_reg.score(x_test, y_test)
    mae = mean_absolute_error(y_test, predictions)

    # Output results to console
    print("------------------------")
    print(f'Random Forest Results for: {position_filter}')
    print(f'Number of data points: {len(features)}')
    print(f'R-squared: {rf_score}')
    print("Mean Absolute Error:", mae)

    return position_filter, mae


def run_linear_regression(data_path) -> None:
    # Contains all data points, don't use
    full_player_data, pick_data = read_csv(data_path.absolute())

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

    # Running linear regression
    model = LinearRegression()
    model.fit(x_train, y_train)
    lin_y_prediction = model.predict(x_test)
    lin_reg = calculate_linear_regression(x_train, y_train)
    lin_reg_score = lin_reg.score(x_test, y_test)
    print("Linear Regression Results:")
    print("R-squared Score: ", lin_reg_score)


if __name__ == '__main__':
    p = Path(__file__).with_name('NFL.csv')

    run_linear_regression(p)

    # Run a random forest analysis on all positions at once
    # all_positions_result = run_random_forest(p)

    # # Run a random forest analysis on each unique position
    # position_list = get_unique_positions(p.absolute())
    # position_results = []
    # for position in position_list:
    #     position_result = run_random_forest(p, position_filter=position)
    #     if position_result is not None:
    #         position_results.append(position_result)

    # # Convert the result of run_random_forest to a tuple
    # all_positions_result = (all_positions_result[0], all_positions_result[1])

    # # Extract positions and MAEs for plotting
    # all_position, all_positions_mae = all_positions_result
    # unique_positions, unique_positions_maes = zip(*position_results)

    # # Plot MAE by Position
    # plot_mae_by_position([all_position] + list(unique_positions), [all_positions_mae] + list(unique_positions_maes))