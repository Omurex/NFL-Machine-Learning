import csv


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
