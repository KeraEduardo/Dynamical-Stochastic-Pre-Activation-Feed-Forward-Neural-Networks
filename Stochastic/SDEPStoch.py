# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
import os

# %%
import os
print("Available CPUs: ", os.cpu_count())

# We save the run_dir and run_id
run_dir = os.environ.get("RUN_DIR")
run_id = int(os.environ.get("RUN_ID"))

#run_dir = "C:/Users/eduar/OneDrive/Desktop/Yani/Fake/Stoch"
#run_id = 1
print(f"Run ID: {run_id}")

# And we set the seed for the samples
#np.random.seed(run_id)

# %%
setup_dict = {}

#with open("setup.txt") as f:
with open("/home/eduardo.martinezgarc/projects/Yani/Fake/Stoch/Training/setup.txt") as f:
    for line in f:
        line = line.strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Use eval only if you trust the file
            setup_dict[key] = eval(value)

# %%
setup_dict

# %%
neurons_per_layer = setup_dict["neurons_per_layer"]
activations_per_layer = setup_dict["activations_per_layer"]
sigma_per_layer = setup_dict["sigma_per_layer"]
samples_per_layer = setup_dict["samples_per_layer"]
capacitors_per_layer = setup_dict["capacitors_per_layer"]
resistors_per_layer = setup_dict["resistors_per_layer"]
height_per_layer = setup_dict["height_per_layer"]
width_per_layer = setup_dict["width_per_layer"]
n_t = setup_dict["n_t"]

n_graph = setup_dict["n_graph"]
min_time = setup_dict["min_time"]
max_time = setup_dict["max_time"]
t_crit = max_time * 0.90
#n_jump = setup_dict["n_jump"]
points_per_layer = setup_dict["points_per_layer"]
#idx_data = setup_dict["idx_data"]
initial_sigma_per_layer = setup_dict["initial_sigma_per_layer"]

# We obtain the solution method
scale_y = setup_dict["scale_y"]
cross_validate = setup_dict["cross_validate"]
dataset = setup_dict["dataset"]

n_samples_dynamical = setup_dict["n_samples_dynamical"]

print(f"Capacitors per layers: {capacitors_per_layer}")

# %% [markdown]
#  ### For multiprocessing

# %%
multiprocessing = setup_dict["multiprocessing"]
#multiprocessing = False
USE_MPI = setup_dict["USE_MPI"]
#USE_MPI = False

if multiprocessing:
    if USE_MPI:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        #rank = sum(neurons_per_layer)
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        from multiprocessing import Pool
        rank = 0
        size = 1

# %% [markdown]
# ### Time vector

# %%
t_vec = np.linspace(min_time, max_time, n_t)
dt = np.abs(t_vec[1] - t_vec[0])
print("dt: ", dt)

# %% [markdown]
#    ### We create the indices for the graph

# %%
# Vectors for the l2 distances at two times
#l1tmax = np.max(t_vec)
#l1tmax = 5
t_graph = np.linspace(min_time, max_time, n_graph)
t1graph = np.linspace(min_time, max_time, n_graph)

# %% [markdown]
#   ## Now we define the activation functions

# %%
def g(x, activation = "sigmoid", beta = 1):
 
    if activation == "sigmoid":
        return np.exp(x)/(1+np.exp(x))
    
    elif activation == "linear":
        return x
    
    elif activation == "tanh":
        return np.tanh(x)

    elif activation == "softplus":
        #return 1/beta * np.log(1 + np.exp(beta*x))
        return x/beta + 1/beta * np.log(np.exp(-x) + np.exp(x*(beta-1)))

def gp(x, activation = "sigmoid", beta = 1):
    if activation == "sigmoid":
        return g(x, activation = "sigmoid")*(1- g(x, activation = "sigmoid"))
    elif activation == "linear":
        return 1
    elif activation == "tanh":
        #D[Tanh[x],x] = Sech[x]^2
        # Sech[x] = 1/Cosh[x]
        return (1/np.cosh[x])**2

    elif activation == "softplus":
        #return np.exp(beta*x) / (1 + np.exp(beta*x))
        # The sigmoid function ! 
        return 1 / (np.exp(-beta*x) + 1)

def F(x, activation = "sigmoid", beta = 1):
    if activation == "sigmoid":
        return np.log(1-x) + x*np.log(x/(1-x))
    
    elif activation == "linear":
        return 0.5 * x**2
    
    elif activation == "tanh":
        return np.log(np.cosh(x))

    elif activation == "softplus":
        print("F creo que no hay antiderivada")

def f(x, activation = "sigmoid", beta = 1, eps = 1e-3, high_val = 1e6):
    
    if activation == "sigmoid":
        return np.log(x/(1-x))
    
    elif activation == "linear":
        return x
    
    elif activation == "tanh":
        #return np.arctanh(x)
        return 0.5 * np.log((1+x)/(1-x))

    elif activation == "softplus":
        #return 1/beta * np.log(-1 + np.exp(beta*x))
        #return x/beta + 1/beta * np.log(np.exp(-x) + np.exp(x*(beta-1)))

        x = np.array(x)
        result = np.zeros_like(x)

        # Use approximation or big value near zero
        near_zero = np.abs(x) < eps
        not_near_zero = ~near_zero

        # Use regular formula where it's safe
        result[not_near_zero] = 1/beta * np.log(-1 + np.exp(beta * x[not_near_zero]))

        # Use approximation or capped value near zero
        # Option 1: very large negative value (or even `np.nan`)
        result[near_zero] = -high_val  # or whatever "really-high" value you want

        return result

def fp(x, activation = "sigmoid", beta = 1, eps = 1e-4, high_val = 1e6):
    x = np.array(x)
    if activation == "sigmoid":
        return 1/(x-x**2)
    
    elif activation == "linear":
        return np.ones_like(x)
    
    elif activation == "tanh":
        return 1/(1-x**2)
    
    elif activation == "softplus":
        #return np.exp(x*beta) / (-1 + np.exp(x*beta))
        #return 1/(-np.exp(-x*beta) + 1)
        denom = -np.exp(-beta * x) + 1
        close_to_zero = np.abs(denom) < eps
        result = np.where(close_to_zero, high_val, 1 / denom)
        return result

def fpp(x, activation = "sigmoid", beta = 1):
    if activation == "sigmoid":
        num = 2*x-1
        den = x-x**2
        return num/(den**2)
    
    elif activation == "linear":
        return np.zeros_like(x)
    
    elif activation == "tanh":
        return 2*x/(1-x**2)**2
    
    elif activation == "softplus":
        return beta/(2-2*np.cosh(x*beta))

def fppp(x, activation = "sigmoid", beta = 1):
    if activation == "sigmoid":
        return -2/(x-1)**3 + 2/x**3
    elif activation == "linear":
        return np.zeros_like(x)
    elif activation == "tanh":
        return 8*x**2 / (1-x**2)**3 + 2/(1-x**2)**2

    elif activation == "softplus":
        return 2* (beta**2) * np.sinh(beta*x) / (2- 2*np.cosh(x*beta))**2

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# %%
def flatten(x):
    x = tf.reshape(x, [-1])           # Flatten to 1D first
    return tf.reshape(x, [-1, 1])     # Then reshape to (x, 1)

# %%
weights = np.load("/home/eduardo.martinezgarc/projects/Yani/Fake/Stoch/Training/weights_per_layer.npz", allow_pickle = True)
#weights = np.load("weights_per_layer.npz", allow_pickle = True)
weights = [weights[f"{i}"] for i in range(len(weights.files))]
#print("Weights: ", weights)

biases = np.load("/home/eduardo.martinezgarc/projects/Yani/Fake/Stoch/Training/biases_mean_per_layer.npz", allow_pickle = True)
#biases = np.load("biases_mean_per_layer.npz", allow_pickle = True)
biases = [biases[f"{i}"] for i in range(len(biases.files))]

# And now we load the variances !
sigma_per_layer = np.load("/home/eduardo.martinezgarc/projects/Yani/Fake/Stoch/Training/biases_var_per_layer.npz", allow_pickle = True)
#sigma_per_layer = np.load("biases_var_per_layer.npz", allow_pickle = True)
sigma_per_layer = [sigma_per_layer[f"{i}"] for i in range(len(sigma_per_layer.files))]

# %% [markdown]
# ### Now we import the data vector
# %%
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split

UCI_FOLDERS = {"concrete": "concrete", "energy": "energy", "kin8nm": "kin8nm",
               "naval": "naval-propulsion-plant", "power": "power-plant",
               "protein": "protein-tertiary-structure", "wine": "wine-quality-red", "yacht": "yacht"}

name = dataset.lower()
if name == "housing":
    data = fetch_california_housing(data_home=sklearn_data_home)
    X, y = data.data, data.target
    # The targets are capped at 5, so we keep only y < housing_y_max
    housing_y_max = 4.99
    idxs = (y <= housing_y_max)
    X, y = X[idxs], y[idxs]
elif name == "diabetes":
    data = load_diabetes()
    X, y = data.data, data.target
elif name in UCI_FOLDERS:
    local_dir = os.path.join(uci_data_dir, UCI_FOLDERS[name], "data")
    raw = np.loadtxt(os.path.join(local_dir, "data.txt"))
    idx_f = np.atleast_1d(np.loadtxt(os.path.join(local_dir, "index_features.txt"))).astype(int)
    idx_t = int(np.atleast_1d(np.loadtxt(os.path.join(local_dir, "index_target.txt")))[0])
    X, y = raw[:, idx_f], raw[:, idx_t]
else:
    raise ValueError(f"Unknown dataset: {dataset}")
seed = 42
# %%
# Now we normalise
from sklearn.preprocessing import StandardScaler

if cross_validate:
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.10, random_state=seed)

    print("Development samples:", len(y_dev))
    print("Test samples:", len(y_test))
else:
    # We have X_train 81%, X_test 10% and X_validation 9%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state = seed)

    # Now we normalise the data (scalers are fitted on the training set only)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val   = scaler_X.transform(X_val)
    X_test  = scaler_X.transform(X_test)

    if scale_y:
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # For faster convergence using GPUs! 
    X_train = X_train.astype('float64')
    y_train = y_train.astype('float64')

    X_test = X_test.astype('float64')
    y_test = y_test.astype('float64')

    X_val = X_val.astype('float64')
    y_val = y_val.astype('float64')

    print("Training samples  :", len(y_train))
    print("Validation samples:", len(y_val))
    print("Test samples      :", len(y_test))

# %%
# Now we select the input vector
input_vector = X_test[run_id]
x_input = tf.convert_to_tensor(input_vector)
x_input = tf.expand_dims(x_input, axis = 0)
# %%
# And now we define the to original units function
def to_original_units(y_net):
    # network output -> original units of y (unchanged if scale_y = False)
    if scale_y:
        return scaler_y.inverse_transform(np.asarray(y_net).reshape(-1, 1)).ravel()
    else: 
        return y_net
# %%
activations_per_neuron = []

for layer_idx in range(len(neurons_per_layer)):
    # Now we save the results
    acts = [activations_per_layer[layer_idx]] * neurons_per_layer[layer_idx]
    activations_per_neuron.extend(acts)

# %% [markdown]
# ### Now we define the sigma vector per layer

# %%
sigma_vector_per_layer = []
for layer_idx in range(len(neurons_per_layer)):
    vector = [sigma_per_layer[layer_idx]] * neurons_per_layer[layer_idx]

    sigma_vector_per_layer.append(vector)

# %% [markdown]
#  ### We define the "effective" noise sigma

# %%
effective_sigma_per_layer = []
for layer_idx in range(len(neurons_per_layer)):
    sigma_layer = sigma_per_layer[layer_idx]
    capacitor_layer = capacitors_per_layer[layer_idx]
    width_layer = width_per_layer[layer_idx]
    rho_layer = resistors_per_layer[layer_idx]

    # Now we calculate the effective sigma
    #result = 2 * sigma_layer * capacitor_layer
    result = 2 * sigma_layer * capacitor_layer / rho_layer * width_layer**2
    #result = sigma_per_layer

    effective_sigma_per_layer.append(result)


# Nose amp
noise_amp_per_layer = [
    (np.sqrt(np.asarray(s, dtype=float)).reshape(-1, 1) / capacitors_per_layer[i]) * np.sqrt(dt)
    for i, s in enumerate(effective_sigma_per_layer)
    ]

# %%
def smart_euler_maruyama_step(z, x, w, b, act_layer, c, rho, width, height, sigma_layer, layer_idx, external_input):
    # We make the distinction for this case
    # BECAUSE we save ourselves an extra matrix multiplication that is
    # ALWAYS constant for the first dynamical layer
    if layer_idx == 0:
        syn_input = external_input
    else:
        syn_input = np.matmul(w, x) + b

    # Now we define the rhs
    #rhs_charge = -z/(c*rho) + syn_input/c
    noise_value = np.random.normal(loc = 0, scale = 1, size = z.shape)

    # Now we define the rhs
    z_nexttimestep = z + (-z/(c*rho) + syn_input/c) * dt + (np.sqrt(sigma_layer)/c) * np.sqrt(dt) * noise_value
    #z_nexttimestep = z + rhs_charge * dt + np.random.normal(loc = 0, scale = (np.sqrt(sigma_layer)/c) * np.sqrt(dt), size = z.shape)
  
    # Now we calculate both the a and the z
    #a_nexttimestep = g(z_nexttimestep, activation = act_layer)
    a_nexttimestep = height * g(z_nexttimestep/width, activation = act_layer)
    return z_nexttimestep, a_nexttimestep


def smart_euler_maruyama_step(z, x, w, b, act_layer, c, rho, width, height, sigma_layer, layer_idx, external_input, rng):
    if layer_idx == 0:
        syn_input = external_input
    else:
        syn_input = np.matmul(w, x) + b

    sigma_layer = np.asarray(sigma_layer).reshape(-1, 1)            # (n_neurons, 1), never (n_neurons,)
    #noise_value = np.random.normal(size=z.shape)
    noise_value = rng.standard_normal(z.shape)

    z_next = z + (-z/(c*rho) + syn_input/c) * dt + sigma_layer * noise_value
    assert z_next.shape == z.shape, f"layer {layer_idx}: {z_next.shape} vs {z.shape}"

    a_next = height * g(z_next / width, activation=act_layer)
    return z_next, a_next
# %%
# Now we define the step size for storing the elements
step = n_t // n_graph 

# %%
print(f"Architecture: {neurons_per_layer}, activations: {activations_per_layer}")
print(f"Time points: {n_t}, exponent = 2**{np.log2(n_t)}")
print(f"Number of samples: {n_samples_dynamical}, exponent = 2**{np.log2(n_samples_dynamical)}")
print(f"Noise added per layer: {sigma_per_layer}")
print(f"Capacitors per neuron: {capacitors_per_layer}")

# %% [markdown]
# ### We define both functions to run the whole code

# %%
def run_simulation_chunk(seed, n_samples_chunk):
    print(f"Process {os.getpid()} started")
    rng = np.random.default_rng()
    # For reproducibility:
    #rng = np.random.default_rng([run_id, seed])
    #print("Setting seed:" , np.random.seed(seed))
    # Now we define the 
    # External input
    external_input = np.matmul(weights[0], input_vector) + biases[0]
    external_input = np.tile(external_input[:, None], (1, n_samples_chunk))
    #print("External input shape: ", external_input.shape)

    # Same for the biases !
    biases_per_layer = []
    for layer_idx in range(len(neurons_per_layer)):
        vec = np.tile(biases[layer_idx][:, None], (1, n_samples_chunk))
        biases_per_layer.append(vec)

    # We do the initial conditions
    dynamical_outputs = []
    dynamical_output_linear = []
    # dynamical_outputs[layer_idx][sample_idx, neuron_idx]
    
    for layer_idx in range(len(neurons_per_layer)):
    #for layer_idx in [-1]:
        if (activations_per_layer[layer_idx] == "softplus"):
            mat = np.zeros((neurons_per_layer[layer_idx], n_samples_chunk)) + 70
        if (activations_per_layer[layer_idx] == "sigmoid"):
            mat = np.zeros((neurons_per_layer[layer_idx], n_samples_chunk)) + 0.5
        if (activations_per_layer[layer_idx] == "tanh"):
            mat = np.zeros((neurons_per_layer[layer_idx], n_samples_chunk))
        if (activations_per_layer[layer_idx] == "linear"):
            mat = np.zeros((neurons_per_layer[layer_idx], n_samples_chunk))
        
        dynamical_outputs.append(mat)
        # Now we save the linear case
        mat = np.zeros((neurons_per_layer[layer_idx], n_samples_chunk))
        dynamical_output_linear.append(mat)
    
    # Now we set up the elements for the samples 
    #samples_dynamical = []
    #for layer_idx in range(len(neurons_per_layer)):
    #    mat = np.zeros((neurons_per_layer[layer_idx], n_samples_chunk, n_graph)
    #    samples_dynamical.append(mat)

    samples_dynamical = np.zeros((neurons_per_layer[-1], n_samples_chunk, n_graph))

    # Now we evolve
    for t_step in range(0, n_t):
        for layer_idx in range(len(neurons_per_layer)):

            # We calculate the input from the previous layer
            if layer_idx == 0:
                x = input_vector
            else:
                x = dynamical_outputs[layer_idx - 1]

            # We load the current state
            a_state = dynamical_outputs[layer_idx]
            z_state = dynamical_output_linear[layer_idx]
            
            z_result, a_result = smart_euler_maruyama_step(z = z_state, x = x, w = weights[layer_idx], b = biases_per_layer[layer_idx], 
                                    act_layer = activations_per_layer[layer_idx], 
                                    c = capacitors_per_layer[layer_idx], rho = resistors_per_layer[layer_idx],
                                    height = height_per_layer[layer_idx], width = width_per_layer[layer_idx],
                                    sigma_layer = noise_amp_per_layer[layer_idx], layer_idx = layer_idx, 
                                    external_input= external_input, rng = rng)
            # We now save the result 
            dynamical_output_linear[layer_idx][:, :] = z_result
            dynamical_outputs[layer_idx][:, :] = a_result

            if (t_step) % step == 0:
                # We check which fraction we are currently located at
                k = (t_step + 1) // step
                
                #if (layer_idx == 0):
                if (layer_idx == len(neurons_per_layer) - 1):
                    
                    #print("LA TAREAAAAA")
                    #print(f"layer_idx = {layer_idx}")
                    #print(f"expected neurons = {neurons_per_layer[layer_idx]}")
                    #print(f"a_result.shape = {a_result.shape}")
                    #print(f"target.shape = {samples_dynamical[:, :, k].shape}")
                    #print(f"Step: {k} / {n_graph}")

                    # Now we save the results
                    samples_dynamical[:, :, k] = to_original_units(a_result)
    print(f"Process {os.getpid()} Finished")
    return dynamical_outputs, samples_dynamical

# %% [markdown]
# ### Now we define the function that parallelises the evolution

# %%
from multiprocessing import Pool

def run_parallel(nproc):

    # split samples
    n_samples_chunk = n_samples_dynamical // nproc

    # different seeds per process
    #seeds = np.random.randint(0, 1000000, size=nproc)
    seeds = np.arange(0, nproc)

    # prepare arguments
    args = [
        (int(seeds[i]), n_samples_chunk)
        for i in range(nproc)]
    #args = [n_samples_chunk] * nproc

    with Pool(processes=nproc) as pool:
        results = pool.starmap(run_simulation_chunk, args)
    
    # unzip results
    dyn_list = [r[0] for r in results]   # per process
    samp_list = [r[1] for r in results]
    
  
    return dyn_list, samp_list

# %%
# Now we run the code !
if __name__ == "__main__":
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print("Number of processes: ", nproc)
    #dynamical_outputs, samples_dynamical = run_parallel(nproc)
    dyn_list, samp_list = run_parallel(nproc)

# %%
# We now save the results !
#pdf_dict = {f"{i}": w for i, w in enumerate(dyn_list)}

# We do not need to save this last one as we only need the results for the last layer !
#file_path = os.path.join(run_dir, "dyn_list.npz")
#print("File path: ,", file_path)   
#np.savez(file_path, dyn_list = np.array(dyn_list, dtype = object))

file_path = os.path.join(run_dir, "samp_list.npz")
print("File path: ,", file_path)
np.savez(file_path, samp_list = np.array(samp_list, dtype = object))

# %% [markdown]
#  ### Now we save all of the samples

# %%
samples_dynamical = np.zeros((neurons_per_layer[-1], n_samples_dynamical, n_graph))

n_samples_chunk = n_samples_dynamical // nproc

for chunk_idx, chunk in enumerate(samp_list):
    start = int(chunk_idx * n_samples_chunk)
    end = int(start + n_samples_chunk)
    #samples_dynamical[layer_idx][:, start:end, :] = chunk[layer_idx]
    samples_dynamical[:, start:end, :] = chunk

#pdf_dict = {f"{i}": w for i, w in enumerate(samples_dynamical)}
#np.savez("samples_dynamical.npz", **pdf_dict)
file_path = os.path.join(run_dir, "samples_dynamical.npz")
print("File path: ,", file_path)
np.savez(file_path, samples_dynamical = np.array(samples_dynamical, dtype = object))

# %%
print("Ahora va la guardadera de cosas")
#raise Exception("Alto perrillo ! ")


