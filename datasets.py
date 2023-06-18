import numpy as np

def generate_data(n, dataset):
	if dataset == 0:
		return generate_data_0(n)
	elif dataset == 1:
		return generate_data_1(n)
	else:
		return generate_data_2(n)

# X<-T->Y
def generate_data_0(n):
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
	Nx = np.random.uniform(-0.035, 0.035, size=len(T))
	Ny = np.random.uniform(-0.035, 0.035, size=len(T))

	# Generate v(T) and u(T) as random linear combinations of Gaussian bumps
	num_bumps_v = 10
	num_bumps_u = 10
	v = random_linear_combination(T, num_bumps_v)
	u = random_linear_combination(T, num_bumps_u)

	# Generate X and Y values
	X = v + Nx
	Y = u + Ny
	
	return T, X, Y
	
# X->Y
def generate_data_1(n):
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
	X = np.linspace(0.1, 1, n)

	# Generate noise terms Nx and Ny
	Nx = np.random.uniform(-0.035, 0.035, size=n)
	Ny = np.random.uniform(-0.035, 0.035, size=n)

	# Generate u(X) as random linear combinations of Gaussian bumps
	num_bumps_u = 10
	u = random_linear_combination(X, num_bumps_u)

	# Generate Y values
	Y = u + Ny
	
	return X, X, Y
	
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
	Ny = np.log(T) * np.random.uniform(-0.035, 0.035, size=len(T))

	# Generate v(T) and u(T) as random linear combinations of Gaussian bumps
	num_bumps_v = 10
	num_bumps_u = 10
	v = random_linear_combination(T, num_bumps_v)
	u = random_linear_combination(T, num_bumps_u)

	# Generate X and Y values
	X = v + Nx
	Y = u + Ny
	
	return T, X, Y
