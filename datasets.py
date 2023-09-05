import numpy as np
import pandas as pd
from scipy import stats


def generate_data(n, dataset):
    if dataset == 0:
        return generate_data_0(n)
    elif dataset == 1:
        return generate_data_1(n)
    elif dataset == 2:
        return generate_data_2(n)
    elif dataset == 3:
        return generate_data_3(n)
    elif dataset == 4:
        return generate_data_4(n)
    elif dataset == 5:
        return generate_data_5(n)
    elif dataset == 6:
        return generate_data_6(n)
    elif dataset == 7:
        return generate_data_7(n)
    elif dataset == 8:
        return generate_data_8(n)
    elif dataset == 9:
        return generate_data_9(n)
    elif dataset == 10:
        return generate_data_10(n)
    elif dataset == 11:
        return generate_data_11(n)
    elif dataset == 12:
        return generate_data_12(n)
    elif dataset == 13:
        return generate_data_13(n)
    elif dataset == 14:
        return generate_data_14(n)
    else:
        return generate_data_15(n)


# X<-T->Y
def generate_data_0(n):

    def extract_air_pressure(filename):
        colspecs = [(68, 75)]
        df = pd.read_fwf(filename, colspecs=colspecs, header=None,
                         names=["Pressure1"])
        pressure = df["Pressure1"].values[300:300 + n].astype(float)

        return pressure

    filename = "data/64060KABE201312.dat"
    filename2 = "data/64060KABI201312.dat"

    pressure = extract_air_pressure(filename)
    pressure2 = extract_air_pressure(filename2)

    time = np.arange(0, n)

    return (time[:n], pressure[:n], pressure2[:n])


# X<-T->Y
def generate_data_1(n):
    data_apple = pd.read_csv("data/AAPL.csv")
    data_caterpillar = pd.read_csv("data/CAT.csv")
    data_dowjones = pd.read_csv("data/DJ.csv")

    apple_prices = data_apple["Close"].values
    caterpillar_prices = data_caterpillar["Close"].values
    dowjones_prices = data_dowjones["Price"].values

    return (dowjones_prices[:n], apple_prices[:n],
            caterpillar_prices[:n])


# Y->X
def generate_data_2(n):
    data = pd.read_csv("data/hfi_cc_2019.csv")

    ef = pd.to_numeric(data["ef_score"], errors="coerce").values
    hf = pd.to_numeric(data["hf_score"], errors="coerce").values

    return (np.arange(0, n), ef[:n], hf[:n])


# X->Y
def generate_data_3(n):
    data = pd.read_csv("data/energy_dataset.csv")
    
    temperature = data["total load actual"].values
    feels_like = data["price actual"].values
    
    return (np.arange(0, n), temperature[:n], feels_like[:n])


# X<-T->Y
def generate_data_4(n):
    np.random.seed(seed=0)

    T = np.linspace(0.1, 1, n)
    Ny = np.random.uniform(-0.035, 0.035, size=len(T))
    Nx = np.random.uniform(-0.035, 0.035, size=len(T))

    X = np.sqrt(T) + Nx
    Y = np.square(T - 0.5) + Ny

    return (T, X, Y)


# Y->X
def generate_data_5(n):
    np.random.seed(seed=1)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.08, 0.08, size=len(T))
    Ny = np.random.uniform(-0.00015, 0.0, size=n)

    X = np.square(T - 0.3) + Nx
    Y = 0.7 * T + Ny

    return (T, X, Y)


# X<-T->Y
def generate_data_6(n):
    np.random.seed(seed=0)

    T = np.linspace(0.1, 1, n)
    Ny = np.random.uniform(-0.01, 0.02, size=len(T))
    Nx = np.random.uniform(-0.01, 0.015, size=len(T))

    X_true = np.log(T) * np.log(T)
    Y_true = 0.7 * np.square(T)

    X = np.log(T) * np.log(T) + Nx
    Y = 0.7 * np.square(T) + Ny

    return (T, X, Y)


# X->Y
def generate_data_7(n):
    np.random.seed(seed=1)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.015, 0.0, size=n)
    Ny = np.random.uniform(-0.08, 0.08, size=len(T))

    X = T + Nx
    Y = np.square(T - 0.5) + Ny

    return (T, X, Y)


# X->Y
def generate_data_8(n):
    np.random.seed(seed=1)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.0, 0.02, size=n)
    Ny = np.random.uniform(-0.06, 0.08, size=len(T))

    X = T + Nx
    Y = np.square(np.exp(T) - 0.3) + Ny

    return (T, X, Y)


# Y->X
def generate_data_9(n):
    np.random.seed(seed=0)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.09, 0.09, size=len(T))
    Ny = np.random.uniform(-0.01, 0.001, size=n)

    X = np.square(T) * np.log(T) * 2 + Nx
    Y = 0.9 * T + 0.01 + Ny

    return (T, X, Y)


# no CAN model
def generate_data_10(n):
    np.random.seed(seed=1)

    T = np.linspace(0.1, 1, n)
    Nx = 5 * T * np.random.uniform(-0.15, 0.15, size=n)
    Ny = 8 * T * np.random.uniform(-0.15, 0.15, size=n)

    X = 4 * (T + 1.2) ** 3 + 0.1 * T + Nx
    Y = 3.3 * (T - 0.5) ** 2 + 0.3 * T + Ny

    return (T, X, Y)


# X->Y
def generate_data_11(n):
    np.random.seed(seed=3)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.015, 0.0, size=n)
    Ny = np.random.uniform(-0.08, 0.08, size=len(T))

    X = T + Nx
    Y = np.sin(T) + Ny

    return (T, X, Y)


# X->Y

def generate_data_12(n):
    np.random.seed(seed=3)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.0, 0.001, size=n)
    Ny = np.random.uniform(-0.06, 0.08, size=len(T))

    X = 1.3 * T + Nx
    Y = np.log(T) + Ny

    return (T, X, Y)


# Y->X
def generate_data_13(n):
    np.random.seed(seed=3)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.08, 0.08, size=len(T))
    Ny = np.random.uniform(-0.00015, 0.0, size=n)

    X = np.sin(T) + Nx
    Y = T + Ny

    return (T, X, Y)


# Y->X
def generate_data_14(n):
    np.random.seed(seed=2)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.09, 0.09, size=len(T))
    Ny = np.random.uniform(-0.01, 0.001, size=n)

    X = np.exp(T) + Nx
    Y = 0.7 * T + Ny

    return (T, X, Y)


# X->Y
def generate_data_15(n):
    np.random.seed(seed=4)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.01, 0.0, size=n)
    Ny = np.random.uniform(0.02, 0.19, size=len(T))

    X = T + Nx
    Y = np.cos(T) * np.sin(T) + Ny

    return (T, X, Y)

