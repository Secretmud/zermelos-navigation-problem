import csv
import pathlib

def store_data():

f_name = f"{nsteps}_{beta}_{T_0}_{T_f}_{pen}_fids.csv"

data_dir = pathlib.Path("data")
data_dir.mkdir(exist_ok=True, parents=True)

file_path = data_dir / f_name
file_path.touch(exist_ok=True)

fidelities = np.array(fidelities)

data = {N: fidelities.tolist()}

with open(file_path, newline="") as csvfile:
    data_reader = csv.reader(csvfile, delimiter=',')
    for d in data_reader:
        key = int(d[0])
        if key not in data:
            data[key] = d[1]

with open(file_path, "w", newline="") as f:
    writer = csv.writer(f)
    for k, v in data.items():
        writer.writerow([k, v])
