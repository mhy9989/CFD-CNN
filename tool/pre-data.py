# -*- coding: utf-8 -*-
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

data_path = "../../cfd-data-1001"
nx = 1181
ny = 220
data_num = 1001
data_type_num = 4

def process_file(i):
    """Process a single data file."""
    flow_data_path = os.path.join(data_path, f"flowxy-{i:04d}.dat")
    data = np.loadtxt(flow_data_path, skiprows=2)
    data_processed = np.zeros((data_type_num, ny, nx))
    for j in range(data_type_num):
        data_processed[j] = data[:nx*ny, j+2].reshape(ny, nx)
    print(f"flowxy-{i:04d}.dat load successful.")
    return i, data_processed

def main():
    # Pre-load the mesh from the first file
    first_data_path = os.path.join(data_path, "flowxy-0001.dat")
    first_data = np.loadtxt(first_data_path, skiprows=2)
    mesh = np.zeros((2, ny, nx))
    mesh[0] = first_data[:nx*ny, 0].reshape(ny, nx)
    mesh[1] = first_data[:nx*ny, 1].reshape(ny, nx)

    data_all = np.zeros((data_num, data_type_num, ny, nx))
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, i) for i in range(1, 1 + data_num)]
        for future in as_completed(futures):
            i, data_processed = future.result()
            data_all[i-1] = data_processed


    np.save("./CFD_data.npy", data_all)
    np.save("./CFD_mesh.npy", mesh)

if __name__ == '__main__':
    main()