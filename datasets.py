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
    elif dataset == 15:
        return generate_data_15(n)
    elif dataset == 16:
        return generate_data_16(n)
    elif dataset == 17:
        return generate_data_17(n)
    else:
        return generate_data_0(n)


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
    column_names = [
        "Sex", "Length", "Diameter", "Height", 
        "Whole weight", "Shucked weight", 
        "Viscera weight", "Shell weight", "Rings"
    ]
    
    data = pd.read_csv('data/abalone.data', header=None, names=column_names)
    
    rings = data["Rings"].values
    shell_weight = data["Shell weight"].values
    
    return (np.arange(0, n), shell_weight[:n], rings[:n])

# X->Y
def generate_data_3(n):
    data = pd.read_csv("data/geyser.csv", delimiter=";")
    
    eruptions = data["eruptions"].values
    waiting = data["waiting"].values
    
    return (np.arange(0, n), eruptions[:n], waiting[:n])

# X->Y
def generate_data_4(n):
    data = pd.read_csv("data/energy_dataset.csv")
    
    load = data["total load actual"].values
    price = data["price actual"].values
    
    return (np.arange(0, n), load[:n], price[:n])

# X<-T->Y
def generate_data_5(n):
    np.random.seed(seed=0)

    T = np.linspace(0.1, 1, n)
    Ny = np.random.uniform(-0.01, 0.02, size=len(T))
    Nx = np.random.uniform(-0.01, 0.015, size=len(T))

    X = np.sin(T) * np.cos(T) + Nx
    Y = 0.7 * np.square(T) + np.sin(T) + Ny

    return (T, X, Y)
    
# X->Y
def generate_data_6(n):
    np.random.seed(seed=1)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.01, 0.0, size=n)
    Ny = np.random.uniform(-0.06, 0.06, size=len(T))

    X = T + Nx
    Y = np.square(np.sin(T) - 0.5) + Ny

    return (T, X, Y)

# X<-T->Y 
def generate_data_7(n):
    np.random.seed(seed=2)

    T = np.linspace(0.1, 1, n)
    Ny = np.random.normal(0, 0.03, size=len(T))
    Nx = np.random.normal(0, 0.03, size=len(T))

    X = np.sin(T) + np.sqrt(T) + Nx
    Y = np.cos(2*T) * np.square(T - 0.4) + Ny

    return (T, X, Y)

# X<-T->Y 
def generate_data_8(n):
    np.random.seed(seed=3)

    T = np.linspace(0.1, 1, n)
    Ny = np.random.uniform(-0.02, 0.02, size=len(T)) * 2
    Nx = np.random.uniform(-0.02, 0.02, size=len(T)) * 2

    X = np.sin(2*T) * np.cos(2*T) + Nx
    Y = 0.6 * np.square(T) + np.sin(3*T) + Ny

    return (T, X, Y)

# Y->X
def generate_data_9(n):
    np.random.seed(seed=4)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.08, 0.09, size=len(T))
    Ny = np.random.uniform(-0.01, 0.005, size=n)

    X = np.square(T) * np.log(2*T) * np.sin(T)+ Nx
    Y = 0.8 * T + Ny

    return (T, X, Y)
    
# X<-T->Y
def generate_data_10(n):
    np.random.seed(seed=2)

    T = abs(np.random.normal(0.5, 0.2, n))
    Nx = np.random.uniform(-0.01, 0.015, size=len(T))
    Ny = np.random.uniform(-0.01, 0.015, size=len(T))

    X = np.log(T) * np.sin(T) + Nx
    Y = np.sin(T) * np.cos(T)**2 + Ny

    return (T, X, Y)

# Y->X
def generate_data_11(n):
    np.random.seed(seed=2)

    T = np.random.uniform(0.1, 1, n)
    Nx = np.random.uniform(-0.05, 0.05, size=len(T))
    Ny = np.random.uniform(-0.01, 0.01, size=len(T))

    X = np.sin(T)**2 + Nx
    Y = T**3 + Ny

    return (T, X, Y)

# X->Y
def generate_data_12(n):
    np.random.seed(seed=2)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.01, 0.01, size=len(T))
    Ny = np.random.uniform(-0.05, 0.05, size=len(T))

    X = np.sqrt(T) + Nx
    Y = np.sin(T)**2 + Ny

    return (T, X, Y)
    
# X<-T->Y
def generate_data_13(n):
    np.random.seed(seed=5)
    
    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.02, 0.02, size=len(T))
    Ny = np.random.uniform(-0.02, 0.02, size=len(T))
    
    X = np.sin(2*T) * np.cos(2*T) + Nx
    Y = np.sin(T) * np.square(T - 0.3) + Ny
    
    return (T, X, Y)

# X->Y
def generate_data_14(n):
    np.random.seed(seed=6)
    
    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.005, 0.005, size=len(T))
    Ny = np.random.uniform(-0.05, 0.04, size=len(T))
    
    X = T + Nx
    Y = np.sin(T**2) + Ny
    
    return (T, X, Y)

# X<-T->Y
def generate_data_15(n):
    np.random.seed(seed=8)
    
    T = np.linspace(0.1, 1, n)
    
    Nx = np.random.uniform(-0.02, 0.02, size=len(T))
    Ny = np.random.uniform(-0.02, 0.02, size=len(T))
    
    X = np.sin(T) * np.cos(T) + Nx
    Y = np.cos(T) * np.square(T - 0.6) + Ny
    
    return (T, X, Y)

# X->Y
def generate_data_16(n):
    np.random.seed(seed=1)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-0.0005, 0.0, size=n)
    Ny = np.random.uniform(-0.005, 0.005, size=len(T))

    X = T + Nx
    Y = np.square(np.sin(T) - 0.5) + Ny

    return (T, X, Y)
    
# Y->X
def generate_data_17(n):
    np.random.seed(seed=4)

    T = np.linspace(0.1, 1, n)
    
    Nx = np.random.normal(0, 0.07, size=len(T))
    Ny = np.random.normal(0, 0.01, size=n)

    X = np.square(T) * np.log(2*T) * np.sin(T)+ Nx
    Y = 0.8 * T + Ny

    return (T, X, Y)

