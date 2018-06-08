import numpy as np


def compute_std_of_column(fileName, delimiter=','):
    mx = np.loadtxt(fileName, delimiter=delimiter, skiprows=1)
    print(np.std(mx[:,1]))


if __name__ == "__main__":
    GPS_X_File = "/Users/tdong/git/FCND-Estimation-CPP/config/log/Graph1.txt"
    Accel_X_File = "/Users/tdong/git/FCND-Estimation-CPP/config/log/Graph2.txt"
    compute_std_of_column(GPS_X_File)
    compute_std_of_column(Accel_X_File)



