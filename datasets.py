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
	else:
		return generate_data_4(n)

# X<-T->Y
def generate_data_0(n):
	np.random.seed(seed=0)		# set seed for reproducibility
	
	def random_linear_combination(x, num_bumps):
		y = np.zeros_like(x)
    		
		for _ in range(num_bumps):
			# Randomly generate parameters for each Gaussian bump
			amplitude = np.random.uniform(0.5, 1.5)
			mean = np.random.uniform(-5,5)
			std = np.random.uniform(0.5, 1.5)

			# Evaluate the Gaussian bump function
			bump = amplitude * np.exp(-(x - mean)**2 / (2 * std**2))

			# Add the bump to the linear combination
			y += bump
		return y

	# Generate T values
	T = np.linspace(0.1, 1, n)

	# Generate noise terms Nx and Ny
	Nx = np.random.uniform(-0.035, 0.035, size=len(T))
	Ny = np.random.uniform(-0.035, 0.035, size=len(T))

	# Generate v(T) and u(T) as random linear combinations of Gaussian bumps
	num_bumps_v = 10
	num_bumps_u = 10
	v = random_linear_combination(T, num_bumps_v)
	u = random_linear_combination(T, num_bumps_u)

	# Generate X and Y values
	X = u + Nx
	Y = v + Ny
	
	return T, X, Y
	
# Y->X
def generate_data_1(n):
	np.random.seed(seed=1)		# set seed for reproducibility
	
	def random_linear_combination(x, num_bumps):
		y = np.zeros_like(x)
    		
		for _ in range(num_bumps):
			# Randomly generate parameters for each Gaussian bump
			amplitude = np.random.uniform(0.5, 1.5)
			mean = np.random.uniform(-5,5)
			std = np.random.uniform(0.5, 1.5)

			# Evaluate the Gaussian bump function
			bump = amplitude * np.exp(-(x - mean)**2 / (2 * std**2))

			# Add the bump to the linear combination
			y += bump
		return y

	# Generate T values
	T = np.linspace(0.1, 1, n)

	# Generate noise terms Nx and Ny
	Nx = np.random.uniform(-0.008, 0.008, size=n)
	Ny = np.random.uniform(-0.0015, 0.0, size=n)

	# Generate u(X) as random linear combinations of Gaussian bumps
	num_bumps_u = 5
	u = random_linear_combination(T, num_bumps_u)
	v = T

	# Generate X and Y values
	X = u + Nx
	Y = v + Ny
	
	return T, X, Y
	
# no CAN model (Nx, Ny dependent on T)
def generate_data_2(n):
	np.random.seed(seed=2)		# set seed for reproducibility
	
	def random_linear_combination(x, num_bumps):
		y = np.zeros_like(x)
    		
		for _ in range(num_bumps):
			# Randomly generate parameters for each Gaussian bump
			amplitude = np.random.uniform(0.5, 1.5)
			mean = np.random.uniform(-5,5)
			std = np.random.uniform(0.5, 1.5)

			# Evaluate the Gaussian bump function
			bump = amplitude * np.exp(-(x - mean)**2 / (2 * std**2))

			# Add the bump to the linear combination
			y += bump
		return y

	# Generate T values
	T = np.linspace(0.1, 1, n)

	# Generate noise terms Nx and Ny
	Nx = T * np.random.uniform(-0.035, 0.035, size=len(T))
	Ny = np.square(T) * np.random.uniform(-0.035, 0.035, size=len(T))

	# Generate v(T) and u(T) as random linear combinations of Gaussian bumps
	num_bumps_v = 10
	num_bumps_u = 10
	v = random_linear_combination(T, num_bumps_v)
	u = random_linear_combination(T, num_bumps_u)

	# Generate X and Y values
	X = v + Nx
	Y = u + Ny
	
	return T, X, Y
	
	
# X<-T->Y
def generate_data_3(n):
	np.random.seed(seed=0)
	
	T = np.linspace(0.1, 1, n)
	Ny = np.random.uniform(-0.035, 0.035, size=len(T))
	Nx = np.random.uniform(-0.035, 0.035, size=len(T))
	
	X = np.log(T) * T + Nx
	Y = np.square(T) + Ny
	
	return T, X, Y
	
# X->Y
def generate_data_4(n):
	np.random.seed(seed=1)
	
	T = np.linspace(0.1, 1, n)
	Nx = np.random.uniform(-0.0015, 0.0, size=n)
	Ny = np.random.uniform(-0.008, 0.008, size=len(T))
	
	X = T + Nx
	Y = np.log(T) * np.log(T) * T + Ny
	
	return T, X, Y
	
