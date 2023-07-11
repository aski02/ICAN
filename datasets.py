import numpy as np

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
	else:
		return generate_data_16(n)
	
# X<-T->Y
def generate_data_0(n):
	np.random.seed(seed=0)
	
	T = np.linspace(0.1, 1, n)
	Ny = np.random.uniform(-0.035, 0.035, size=len(T))
	Nx = np.random.uniform(-0.035, 0.035, size=len(T))
	
	X_true = np.sqrt(T)
	Y_true = np.square(T - 0.5)
	
	X = np.sqrt(T) + Nx
	Y = np.square(T - 0.5) + Ny
	
	return T, X, Y, X_true, Y_true

# X<-T->Y
def generate_data_1(n):
	np.random.seed(seed=0)
	
	T = np.linspace(0.1, 1, n)
	Ny = np.random.uniform(-0.01, 0.02, size=len(T))
	Nx = np.random.uniform(-0.01, 0.015, size=len(T))
	
	X_true = np.log(T) * np.log(T)
	Y_true = 0.7 * np.square(T)
	
	X = np.log(T) * np.log(T) + Nx
	Y = 0.7 * np.square(T) + Ny
	
	return T, X, Y, X_true, Y_true
    
# X->Y
def generate_data_2(n):
	np.random.seed(seed=1)
		
	T = np.linspace(0.1, 1, n)
	Nx = np.random.uniform(-0.015, 0.0, size=n)
	Ny = np.random.uniform(-0.08, 0.08, size=len(T))
	
	X_true = T
	Y_true = np.square(T - 0.5)
	
	X = T + Nx
	Y = np.square(T - 0.5) + Ny
	
	return T, X, Y, X_true, Y_true	
	
# X->Y
def generate_data_3(n):
	np.random.seed(seed=1)
		
	T = np.linspace(0.1, 1, n)
	Nx = np.random.uniform(-0.0, 0.02, size=n)
	Ny = np.random.uniform(-0.06, 0.08, size=len(T))
	
	X_true = T
	Y_true = np.square(np.exp(T) - 0.3)
	
	X = T + Nx
	Y = np.square(np.exp(T) - 0.3) + Ny
	
	return T, X, Y, X_true, Y_true
# Y->X
def generate_data_4(n):
	np.random.seed(seed=1)
	
	T = np.linspace(0.1, 1, n)
	Nx = np.random.uniform(-0.08, 0.08, size=len(T))
	Ny = np.random.uniform(-0.00015, 0.0, size=n)
	
	X_true = np.square(T - 0.3)
	Y_true = 0.7 * T
	
	X = np.square(T - 0.3) + Nx
	Y = 0.7 * T + Ny
	
	return T, X, Y, X_true, Y_true
	
# Y->X
def generate_data_5(n):
	np.random.seed(seed=0)
	
	T = np.linspace(0.1, 1, n)
	Nx = np.random.uniform(-0.09, 0.09, size=len(T))
	Ny = np.random.uniform(-0.01, 0.001, size=n)
	
	X_true = np.square(T) * np.log(T) * 2
	Y_true = 0.9 * T + 0.01
	
	X = np.square(T) * np.log(T) * 2 + Nx
	Y = 0.9 * T + 0.01 + Ny
	
	return T, X, Y, X_true, Y_true
	
# no CAN model
def generate_data_6(n):
    np.random.seed(seed=1)
    
    T = np.linspace(0.1, 1, n)
    Nx = 5 * T * np.random.uniform(-0.15, 0.15, size=n)
    Ny = 8 * T * np.random.uniform(-0.15, 0.15, size=n)
    
    X_true = 4 * (T+1.2)**3 + 0.1 * T
    Y_true = 3.3 * (T-0.5)**2 + 0.3 * T
    
    X = X_true + Nx
    Y = Y_true + Ny
    
    return T, X, Y, X_true, Y_true
	
# X<-T->Y
def generate_data_7(n):
	np.random.seed(seed=2)
	
	T = np.linspace(0.1, 1, n)
	Ny = np.random.uniform(-0.01, 0.02, size=len(T))
	Nx = np.random.uniform(-0.01, 0.02, size=len(T))
	
	X_true = np.sin(T)
	Y_true = np.cos(T - 0.5)
	
	X = np.sin(T) + Nx
	Y = np.cos(T - 0.5) + Ny
	
	return T, X, Y, X_true, Y_true

# X<-T->Y
def generate_data_8(n):
	np.random.seed(seed=2)
	
	T = np.linspace(0.1, 1, n)
	Ny = np.random.uniform(-0.01, 0.02, size=len(T))
	Nx = np.random.uniform(-0.01, 0.015, size=len(T))
	
	X_true = np.exp(T) 
	Y_true = np.log(T)
	
	X = np.exp(T) + Nx
	Y = np.log(T) + Ny
	
	return T, X, Y, X_true, Y_true
    
# X->Y
def generate_data_9(n):
	np.random.seed(seed=3)
		
	T = np.linspace(0.1, 1, n)
	Nx = np.random.uniform(-0.015, 0.0, size=n)
	Ny = np.random.uniform(-0.08, 0.08, size=len(T))
	
	X_true = T
	Y_true = np.sin(T)
	
	X = T + Nx
	Y = np.sin(T) + Ny
	
	return T, X, Y, X_true, Y_true	

# X->Y
def generate_data_10(n):
	np.random.seed(seed=3)
		
	T = np.linspace(0.1, 1, n)
	Nx = np.random.uniform(-0.0, 0.001, size=n)
	Ny = np.random.uniform(-0.06, 0.08, size=len(T))
	
	X_true = 1.3 * T
	Y_true = np.log(T)
	
	X = 1.3 * T + Nx
	Y = np.log(T) + Ny
	
	return T, X, Y, X_true, Y_true

# Y->X
def generate_data_11(n):
	np.random.seed(seed=3)
	
	T = np.linspace(0.1, 1, n)
	Nx = np.random.uniform(-0.08, 0.08, size=len(T))
	Ny = np.random.uniform(-0.00015, 0.0, size=n)
	
	X_true = np.sin(T)
	Y_true = T
	
	X = np.sin(T) + Nx
	Y = T + Ny
	
	return T, X, Y, X_true, Y_true
	
# Y->X
def generate_data_12(n):
	np.random.seed(seed=2)
	
	T = np.linspace(0.1, 1, n)
	Nx = np.random.uniform(-0.09, 0.09, size=len(T))
	Ny = np.random.uniform(-0.01, 0.001, size=n)
	
	X_true = np.exp(T)
	Y_true = 0.7 * T
	
	X = np.exp(T) + Nx
	Y = 0.7 * T + Ny
	
	return T, X, Y, X_true, Y_true
	
# no CAN model
def generate_data_13(n):
    np.random.seed(seed=3)
    
    T = np.linspace(0.1, 1, n)
    Nx = 7 * T * np.random.uniform(-0.25, 0.15, size=n)
    Ny = 4 * T * np.random.uniform(-0.15, 0.15, size=n)
    
    X_true = np.sin(T)
    Y_true = np.cos(T)
    
    X = X_true + Nx
    Y = Y_true + Ny
    
    return T, X, Y, X_true, Y_true

# X<-T->Y
def generate_data_14(n):
	np.random.seed(seed=3)
	
	T = np.linspace(0.1, 1, n)
	Ny = np.random.uniform(-0.01, 0.02, size=len(T))
	Nx = np.random.uniform(-0.01, 0.02, size=len(T))
	
	X_true = np.log(T)
	Y_true = np.sin(T - 0.5)
	
	X = np.log(T) + Nx
	Y = np.sin(T - 0.5) + Ny
	
	return T, X, Y, X_true, Y_true

# X->Y
def generate_data_15(n):
	np.random.seed(seed=4)
		
	T = np.linspace(0.1, 1, n)
	Nx = np.random.uniform(-0.015, 0.0, size=n)
	Ny = np.random.uniform(-0.08, 0.09, size=len(T))
	
	X_true = T
	Y_true = np.cos(T) * np.sin(T)
	
	X = T + Nx
	Y = np.cos(T) * np.sin(T) + Ny
	
	return T, X, Y, X_true, Y_true	

# Y->X
def generate_data_16(n):
	np.random.seed(seed=4)
	
	T = np.linspace(0.1, 1, n)
	Nx = np.random.uniform(-0.08, 0.08, size=len(T))
	Ny = np.random.uniform(-0.0003, -0.001, size=n)
	
	X_true = 5 * np.log(T)
	Y_true = T
	
	X = 5 * np.log(T) + Nx
	Y = T + Ny
	
	return T, X, Y, X_true, Y_true
