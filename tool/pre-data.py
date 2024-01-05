# -*- coding: utf-8 -*-
import numpy as np
import os

data_path = "../flow2d-t-1001"
nx = 1181
ny = 220
data_num = 1001
data_type_num = 4

def main():
    data_all = np.zeros((data_num,data_type_num, ny, nx))
    mesh = np.zeros((2, ny, nx))
    record_mesh = True
    # Read all data
    for i in range(1, 1+data_num):
        flow_data_path = os.path.join(data_path , f"flowxy-{i:04d}.dat")
        data = np.loadtxt(flow_data_path,skiprows=2)
        if record_mesh:
            mesh[0] = data[:nx*ny,0].reshape(ny, nx)
            mesh[1] = data[:nx*ny,1].reshape(ny, nx)
            record_mesh = False
        for j in range(data_type_num):
            data_all[i-1, j] = data[:nx*ny,j+2].reshape(ny, nx)
        print(f"flowxy-{i:04d}.dat load successful.")
    np.save("./CFD_data.npy",data_all)
    np.save("./CFD_mesh.npy",mesh)

if __name__ == '__main__':
    main()