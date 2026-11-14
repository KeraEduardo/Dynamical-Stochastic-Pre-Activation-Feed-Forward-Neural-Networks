# %%
import numpy as np

#import latex
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

from scipy.sparse import diags, kron, eye, lil_matrix, csr_matrix, identity, csc_matrix

from scipy.sparse.linalg import spsolve, splu, bicgstab, spilu, LinearOperator, expm_multiply, gmres

import pyamg

#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif"
#})

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer
import tensorflow as tf 

# For distributed processes
from tensorflow.python.distribute import mirrored_strategy
import matplotlib.image as mpimg
import os

# %% [markdown]
#  ### This file sovles the Fokker-Planck Equation

# %%
setup_dict = {}

with open("setup.txt") as f:
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
n_t = setup_dict["n_t"]
#n_t = 2**14
n_graph = setup_dict["n_graph"]
min_time = setup_dict["min_time"]
max_time = setup_dict["max_time"]
#n_jump = setup_dict["n_jump"]
points_per_layer = setup_dict["points_per_layer"]
idx_data = setup_dict["idx_data"]
initial_sigma_per_layer = setup_dict["initial_sigma_per_layer"]

# We obtain the solution method
use_gpu = setup_dict["use_gpu"]
tensorise = setup_dict["tensorise"]
derivative_mode = setup_dict["derivative_mode"]
solution_method = setup_dict["solution_method"]
neuron_idxs = setup_dict["neuron_idxs"]

# For the graph
labels = setup_dict["labels"]
colors = setup_dict["colors"]
graphing_idxs = setup_dict["graphing_idxs"]

# For the times
l1tmax = setup_dict["l1tmax"]

# %% 
# Now we set the normalisation constants
if (sum(neurons_per_layer)==1):
    norm_const = 1
elif (sum(neurons_per_layer)==2):
    norm_const = 1e2
else:
    norm_const = 1e4

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
#    ### We check if we need large memory

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
t_graph = np.linspace(min_time, l1tmax, n_graph)
t1graph = np.linspace(min_time, l1tmax, n_graph)

# Now we obtain the selected indixes
max_idx1 = int(l1tmax / max_time * n_t)

idxs_layer1 = np.linspace(0, max_idx1, n_graph)

# We turn the indices into integer values
for elem_idx in range(len(idxs_layer1)):
    elem = idxs_layer1[elem_idx]
    idxs_layer1[elem_idx] = int(elem)

# %% [markdown]
# ### Now we load the weights

# %%
loaded = np.load("weights_per_layer.npz")

# Numeric keys
weights_per_layer = [loaded[f"{i}"] for i in range(len(loaded.files))]

# %% [markdown]
# ### Now we load the biases

# %%
loaded = np.load("biases_per_layer.npz")

# Numeric keys
biases_per_layer = [loaded[f"{i}"] for i in range(len(loaded.files))]

# %% [markdown]
# ### Now we load the samples

# %%
loaded = np.load("pdf_per_neuron.npz")

# Numeric keys
pdf_per_neuron = [loaded[f"{i}"] for i in range(len(loaded.files))]

# %% [markdown]
# ### Now we load the image

# %%
# Load your image
img = mpimg.imread('model_architecture1.jpeg')

# %% [markdown]
# ### Now we obtain the points per neuron

# %%
capacitors_per_neuron = []
resistors_per_neuron = []
sigma_per_neuron = []
points_per_neuron = []

for layer in range(len(neurons_per_layer)):
    mini_capac = [capacitors_per_layer[layer]] * neurons_per_layer[layer]
    mini_resis = [resistors_per_layer[layer]] * neurons_per_layer[layer]
    mini_sigma = [sigma_per_layer[layer]] * neurons_per_layer[layer]
    mini_points = [points_per_layer[layer]] * neurons_per_layer[layer]
    
    # Now we save it
    capacitors_per_neuron.extend(mini_capac)
    resistors_per_neuron.extend(mini_resis)
    sigma_per_neuron.extend(mini_sigma)
    points_per_neuron.extend(mini_points)



# %% [markdown]
# ### Now we obtain the biases per neuron

# %%
biases_per_neuron = []
for layer_idx in range(len(neurons_per_layer)):
    bias_layer = biases_per_layer[layer_idx]
    
    biases_per_neuron.extend(bias_layer)

# %% [markdown]
# ### Now we define the activation functions

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

# %%
f(0.5, activation= "tanh")

# %% [markdown]
# ### Tensorflow Versons

# %%
def g_tf(x, activation="sigmoid", beta=1.0):
    if activation == "sigmoid":
        return tf.nn.sigmoid(x)
    
    elif activation == "linear":
        return x
    
    elif activation == "tanh":
        return tf.nn.tanh(x)

    elif activation == "softplus":
        return x/beta + 1/beta * tf.math.log(tf.math.exp(-x) + tf.math.exp(x*(beta-1)))

def gp_tf(x, activation="sigmoid", beta=1.0):
    if activation == "sigmoid":
        sig = g_tf(x, activation="sigmoid")
        return sig * (1 - sig)
    
    elif activation == "linear":
        return tf.ones_like(x)
    
    elif activation == "tanh":
        return 1 / tf.math.cosh(x)**2

    elif activation == "softplus":
        return 1 / (tf.math.exp(-beta * x) + 1)

def F_tf(x, activation="sigmoid", beta=1.0):
    if activation == "sigmoid":
        return tf.math.log(1-x) + x*tf.math.log(x/(1-x))
    
    elif activation == "linear":
        return 0.5 * x**2
    
    elif activation == "tanh":
        return tf.math.log(tf.math.cosh(x))

    elif activation == "softplus":
        raise NotImplementedError("F for softplus activation not implemented")

def f_tf(x, activation="sigmoid", beta=1.0, eps=1e-3, high_val=1e6):
    if activation == "sigmoid":
        return tf.math.log(x/(1-x))
    
    elif activation == "linear":
        return x
    
    elif activation == "tanh":
        return tf.math.atanh(x)

    elif activation == "softplus":
        x = tf.convert_to_tensor(x)
        result = tf.zeros_like(x)
        
        near_zero = tf.abs(x) < eps
        not_near_zero = tf.logical_not(near_zero)
        
        result = tf.where(
            not_near_zero,
            1/beta * tf.math.log(-1 + tf.math.exp(beta * x)),
            -high_val * tf.ones_like(x)
        )
        return result

def fp_tf(x, activation="sigmoid", beta=1.0, eps=1e-4, high_val=1e6):
    x = tf.convert_to_tensor(x)
    if activation == "sigmoid":
        return 1/(x - x**2)
    
    elif activation == "linear":
        return tf.ones_like(x)
    
    elif activation == "tanh":
        return 1/(1 - x**2)
    
    elif activation == "softplus":
        denom = -tf.math.exp(-beta * x) + 1
        close_to_zero = tf.abs(denom) < eps
        return tf.where(close_to_zero, high_val * tf.ones_like(x), 1 / denom)

def fpp_tf(x, activation="sigmoid", beta=1.0):
    if activation == "sigmoid":
        num = 2*x - 1
        den = x - x**2
        return num/(den**2)
    
    elif activation == "linear":
        return tf.zeros_like(x)
    
    elif activation == "tanh":
        return 2*x/(1 - x**2)**2
    
    elif activation == "softplus":
        return beta/(2 - 2*tf.math.cosh(x*beta))

def fppp_tf(x, activation="sigmoid", beta=1.0):
    if activation == "sigmoid":
        return -2/(x-1)**3 + 2/x**3
    
    elif activation == "linear":
        return tf.zeros_like(x)
    
    elif activation == "tanh":
        return 8*x**2 / (1-x**2)**3 + 2/(1-x**2)**2

    elif activation == "softplus":
        return 2 * (beta**2) * tf.math.sinh(beta*x) / (2 - 2*tf.math.cosh(x*beta))**2

# %% [markdown]
#  ### Tf flatten and np flatten

# %%
def flatten(x):
    x = tf.reshape(x, [-1])           # Flatten to 1D first
    return tf.reshape(x, [-1, 1])     # Then reshape to (x, 1)

def flatten_np(x):
    return np.reshape(x, (-1))

# %% [markdown]
#  ### Now we choose the input to the neural network

# %%
import tensorflow_datasets as tfds
x_test, y_test = tfds.load("mnist", split=['test'], batch_size=-1, as_supervised=True)[0]
x_dynamical = x_test / 255

# We select the input data
input_vector = flatten_np( x_dynamical[idx_data].numpy() )

# %% [markdown]
#  ### Function that creates input vectors for the neurons

# %%
epsilon = 1e-6

def create_vector(activation, n_x, eps = epsilon, min_z = -5, max_z = 5):
    if activation == "sigmoid":
        x = np.linspace(eps, 1-eps, n_x, dtype = np.float64)
    elif activation == "linear":
        x = np.linspace(min_z, max_z, n_x, dtype = np.float64)
    elif activation == "tanh":
        x = np.linspace(-1 + eps, 1-eps, n_x, dtype = np.float64)
    elif activation == "relu":
        x = np.linspace(min_z, max_z, n_x, dtype = np.float64)
    elif activation == "softplus":
        x = np.linspace(min_z, max_z, n_x, dtype = np.float64)
    return x

# %%
vectors_per_layer = []
for layer_idx in range(len(neurons_per_layer)):
    result = create_vector(activation = activations_per_layer[layer_idx], n_x = points_per_layer[layer_idx] )
    vectors_per_layer.append( result )

# %% [markdown]
#  ### Vectors per neuron

# %%
vectors_per_neuron = []
for layer_idx in range(len(neurons_per_layer)):
    vector = vectors_per_layer[layer_idx]
    result = [vector] * neurons_per_layer[layer_idx]
    vectors_per_neuron.extend(result)

# %% [markdown]
#  ### Now we create the effective noise per neuron

# %%
effective_sigma_per_neuron = [] 
for neuron_idx in range(sum(neurons_per_layer)):
    sigma_neuron = sigma_per_neuron[neuron_idx]
    capacitor_neuron = capacitors_per_neuron[neuron_idx]
    prod = 2 * sigma_neuron * np.abs(capacitor_neuron)
    
    # If we have noise
    effective_sigma_per_neuron.append(prod)

    # If we do not have any noise
    #effective_sigma_per_neuron.append(0)

# %% [markdown]
#  ### Adjacency matrix

# %%
def create_adjacency_matrix_ff(layer_sizes, weights):
    total_neurons = np.sum(layer_sizes)
    adj_matrix = np.zeros((total_neurons, total_neurons), dtype = np.float64)

    # We iterate over the layers
    start_row = layer_sizes[0]
    start_col = 0
    for layer_idx in range(len(layer_sizes) - 1):
        from_row = start_row
        to_row = from_row + weights[layer_idx].shape[0]

        from_col = start_col
        to_col = from_col + weights[layer_idx].shape[1]

        # Now we update the matrix
        adj_matrix[from_row : to_row, from_col : to_col] = weights[layer_idx]

        start_row = to_row
        start_col = to_col
    
    return adj_matrix

adj_matrix = create_adjacency_matrix_ff(neurons_per_layer, weights_per_layer[1:])
adj_matrix = adj_matrix * 100
print("Adjacency matrix: ", adj_matrix)

# %% [markdown]
#  ### Now we create the mesh network

# %%
mesh_network = np.meshgrid(*vectors_per_neuron)

# %% [markdown]
#  ### Now we create the synaptic input from other neurons

# %%
input_from_neurons = []

for i in range(sum(neurons_per_layer)):
    row = adj_matrix[i, :]
    suma = 0
    for j in range(sum(neurons_per_layer)):
        # We iterate over the weights
        weight_j = row[j]
        if weight_j != 0.0:
            # We sum the element
            suma = suma + mesh_network[j] * weight_j

            # We only perform the sums whenever the weight is different!
            # Because then the multiplication may be expensive
        else:
            print("Skipping this element in the input_from_neurons ! ")
    
    # Now we save the result
    # tf tensor 
    #suma = tf.cast( suma, dtype = tf.float32)
    input_from_neurons.append( suma )

# %% [markdown]
#  ### Now we define the external input from other neurons

# %%
external_input_per_neuron = np.zeros((sum(neurons_per_layer), ))

# We save the original external input
external_input_per_neuron[0: neurons_per_layer[0]] = weights_per_layer[0] @ input_vector

# Now we add the bias
external_input_per_neuron = external_input_per_neuron + biases_per_neuron

# %% [markdown]
#  ### Now we create the initial joint pdf for the neurons in the neural network

# %%
initial_sigma_per_neuron = []
activations_per_neuron = []

for layer_idx in range(len(neurons_per_layer)):
    sigma_neuron = [initial_sigma_per_layer[layer_idx]] * neurons_per_layer[layer_idx]

    # Now we save the results
    initial_sigma_per_neuron.extend(sigma_neuron)

    acts = [activations_per_layer[layer_idx]] * neurons_per_layer[layer_idx]
    activations_per_neuron.extend(acts)

# %% [markdown]
#  ### For the individual neurons

# %%
initial_pdfs = []
for neuron_idx in range(sum(neurons_per_layer)):
    print(f"Neuron idx: {neuron_idx}")
    factor = 1/(np.sqrt(2*np.pi) * initial_sigma_per_neuron[neuron_idx])

    vec = mesh_network[neuron_idx]
    initial_mesh_pdf = np.exp(-0.5 * ( (f(vec, activation = activations_per_neuron[neuron_idx]) - 0) / initial_sigma_per_neuron[neuron_idx]) ** 2)
    initial_mesh_pdf = initial_mesh_pdf * np.abs(fp(vec, activation = activations_per_neuron[neuron_idx]))
    
    # Now we normalise
    initial_mesh_pdf = initial_mesh_pdf / np.sum(initial_mesh_pdf)

    # We save the result
    initial_pdfs.append(initial_mesh_pdf)

# %% [markdown]
#  ### Now we define the differentials per neuron

# %%
differentials_per_neuron = []
for layer_idx in range(len(neurons_per_layer)):
    vec = vectors_per_layer[layer_idx]
    dx = np.abs( vec[1] - vec[0] )

    result = [dx] * neurons_per_layer[layer_idx]
    # Now we save 
    differentials_per_neuron.extend(result)

# %% [markdown]
#  ### For the neural network (since they are all independent of each other)

# %%
P = 1
for neuron_idx in range(sum(neurons_per_layer)):
    pdf = initial_pdfs[neuron_idx]

    # We multiply 
    # Initial PDFs are independent of each other
    P = P * pdf

# Now we normalise
Q = np.sum(P) * np.prod(differentials_per_neuron)
P = P / Q

# %% [markdown]
#  ### Numpy numerical derivative

# %%
def spat_der(P, da, axis = 0):
    return np.gradient(P, da, axis=axis)

def spat_sec_der(P, da, axis = 0):
    first = np.gradient(P, da, axis=axis)
    second = np.gradient(first, da, axis=axis)
    return second

# %% [markdown]
#  ### Now we introduce the marginalise function

# %%
def marginalize(P, axis_to_keep):
    """
    Marginalize a joint probability distribution over all axes except one.

    Parameters:
    - P: np.ndarray
        A multidimensional array representing the joint probability distribution.
    - axis_to_keep: int
        The axis to keep (e.g., 0 to keep x in P(x, y, z)).

    Returns:
    - np.ndarray: The marginalized distribution over the selected axis.
    """
    axes = tuple(i for i in range(P.ndim) if i != axis_to_keep)
    P = np.sum(P, axis=axes)

    # Normalise
    P = P / np.sum( P )
    return P

# %% [markdown]
#  ### Now we introduce the dynamical PDFs per neuron

# %%
dynamical_pdf_per_neuron = []

for neuron_idx in range(sum(neurons_per_layer)):
    matr = np.zeros((n_graph, len(vectors_per_neuron[neuron_idx])))
    dynamical_pdf_per_neuron.append(matr)

reversed_list = list( reversed(range(sum(neurons_per_layer))) )

# %% [markdown]
#  ### We obtain the coefficients

# %%
drift1, drift2 = [], []
diffussion1, diffussion2, diffussion3 = [], [], []

for neuron_idx in range(sum(neurons_per_layer)):
        # Constants
        c = capacitors_per_neuron[neuron_idx]
        rho = resistors_per_neuron[neuron_idx]

        neuron_syn_input = input_from_neurons[neuron_idx]
        external_input = external_input_per_neuron[neuron_idx]
        # Now we save the total synaptit input
        total_syn_input = neuron_syn_input + external_input

        a_vec = mesh_network[neuron_idx]
        act = activations_per_neuron[neuron_idx]
        #noise = sigma_per_neuron[neuron_idx]
        noise = effective_sigma_per_neuron[neuron_idx]
        da = differentials_per_neuron[neuron_idx]

        f_neuron = f(a_vec, activation = act)
        fp_neuron = fp(a_vec, activation = act)
        fpp_neuron = fpp(a_vec, activation = act)
        fppp_neuron = fppp(a_vec, activation = act)


        ##### DRIFT #####
        # We compute the drift coefficients
        term11 = -1/(c*rho)
        term12 = -total_syn_input/c * fpp_neuron / fp_neuron **2
        term13 = f_neuron * fpp_neuron / (c*rho * fp_neuron**2)
        term14 = 3/2 * (noise/c**2) * (fpp_neuron ** 2) / (fp_neuron**4) 
        term15 = -noise/(2*c**2) * fppp_neuron / (fp_neuron**3)

        term21 = total_syn_input / (c* fp_neuron )
        term22 = -f_neuron / (c*rho* fp_neuron )
        term23 = -noise/(2*c**2) * fpp_neuron / (fp_neuron**3)

        drift1_neuron = -(term11 + term12 + term13 + term14 + term15)
        drift2_neuron = -(term21 + term22 + term23)

        ##### DIFFUSSION #####
        term1_diff = 6*(noise) * fpp_neuron**2 / (c**2 * fp_neuron**4)
        term2_diff = -2*(noise)/(c**2 *fp_neuron**3) * fppp_neuron
            
        diffussion1_neuron = term1_diff + term2_diff
        diffussion2_neuron = -4*(noise)/(c**2 * fp_neuron**3) * fpp_neuron
        diffussion3_neuron = (noise)/(c**2 * fp_neuron**2)

        # Now we save the coefficients
        ## We save the drift coefficients
        drift1.append(drift1_neuron)
        drift2.append(drift2_neuron)

        ## We save the diffussion coefficients
        diffussion1.append(diffussion1_neuron * 1/2)
        diffussion2.append(diffussion2_neuron * 1/2)
        diffussion3.append(diffussion3_neuron * 1/2)

# %% [markdown]
# # Now we define the solution methods

# %%
# %%
# This function works for pickling ! 

def neuron_contribution(args):
            # We unpack everything
            neuron_idx, P, drift1_tf, drift2_tf, diff1_tf, diff2_tf, diff3_tf, differentials_per_neuron, reversed_list = args

            axis = reversed_list[neuron_idx]
            da = differentials_per_neuron[neuron_idx]

            sd1 = spat_der(P, da, axis=axis)
            sd2 = spat_sec_der(P, da, axis=axis)

            return (
                drift1_tf[neuron_idx] * P
                + drift2_tf[neuron_idx] * sd1
                + diff1_tf[neuron_idx] * P
                + diff2_tf[neuron_idx] * sd1
                + diff3_tf[neuron_idx] * sd2)

# %% [markdown]
# ## General method

# %%
def num_solve(P, dt, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf, use_gpu = use_gpu, tensorise = False,
        multiprocessing = multiprocessing, USE_MPI = USE_MPI,
        method = "RK4"):
    
    if multiprocessing:

        # We define the MPIP version
        def rhs_mpi(P, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf, differentials_per_neuron, reversed_list):
            #comm = MPI.COMM_WORLD
            #rank = comm.Get_rank()
            #size = comm.Get_size()

            # We check that the number of neurons is the same as the number of tasks!
            if size != sum(neurons_per_layer):
                if rank == 0:
                    raise RuntimeError(
                        f"MPI ranks ({size}) must equal number of neurons ({sum(neurons_per_layer)})")
                MPI.Finalize()
                return

            # Each rank handles one neuron
            neuron_idx = rank
            axis = reversed_list[neuron_idx]
            da = differentials_per_neuron[neuron_idx]

            # Compute derivatives only for this neuron
            sd1 = spat_der(P, da, axis=axis)
            sd2 = spat_sec_der(P, da, axis=axis)

            # Local contribution (shape 128^4)
            local_total = (
                drift1_tf[neuron_idx] * P
                + drift2_tf[neuron_idx] * sd1
                + diffussion1_tf[neuron_idx] * P
                + diffussion2_tf[neuron_idx] * sd1
                + diffussion3_tf[neuron_idx] * sd2 )

            # Sum contributions across ranks
            total = np.zeros_like(local_total)
            comm.Allreduce(local_total, total, op=MPI.SUM)

            return total
        
        # WE define the local version i.e. LAPTOP
        # Now we define a function that uses POOL !!!
        def rhs_local(P, drift1_tf, drift2_tf, diff1_tf, diff2_tf, diff3_tf, differentials_per_neuron, reversed_list, 
                      n_workers = sum(neurons_per_layer)):

            args = [
                (i, P, drift1_tf, drift2_tf, diff1_tf, diff2_tf, diff3_tf,
                differentials_per_neuron, reversed_list)
                for i in range(sum(neurons_per_layer)) ]

            with Pool(processes=n_workers) as pool:
                results = pool.map(neuron_contribution, args)

            return np.sum(results, axis=0)
        
        # Now FINALLY we create the rhs function
        def rhs(P, drift1_tf, drift2_tf, diff1_tf, diff2_tf, diff3_tf, 
                differentials_per_neuron = differentials_per_neuron, reversed_list = reversed_list,
                USE_MPI=USE_MPI, n_workers=sum(neurons_per_layer)):

            if USE_MPI:
                return rhs_mpi(P, drift1_tf, drift2_tf,
                            diff1_tf, diff2_tf, diff3_tf,
                            differentials_per_neuron, reversed_list)
            else:
                return rhs_local(P, drift1_tf, drift2_tf,
                                diff1_tf, diff2_tf, diff3_tf,
                                differentials_per_neuron, reversed_list,
                                n_workers = n_workers)


    else:
        def rhs(P, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf):
            P_expanded = [P]*sum(neurons_per_layer)
            P_stacked = np.stack(P_expanded, axis = 0)

            spatders1 = []
            spatders2 = []
            for neuron_idx in range(sum(neurons_per_layer)):
                differential = differentials_per_neuron[neuron_idx]
                axis = reversed_list[neuron_idx]

                # Now we compute the derivatives
                sd1 = spat_der(P, da = differential, axis = axis)
                sd2 = spat_sec_der(P, da = differential, axis = axis)

                spatders1.append(sd1)
                spatders2.append(sd2)

            spatders1 = np.stack(spatders1, axis = 0)  # Shape: [num_neurons, ...]
            spatders2 = np.stack(spatders2, axis = 0)  # Shape: [num_neurons, ...]

            # Vectorised computation of drift and diffusion terms
            drift_terms = drift1_tf * P_stacked + drift2_tf * spatders1
            diffusion_terms = (diffussion1_tf * P_stacked + 
                                diffussion2_tf * spatders1 + 
                                diffussion3_tf * spatders2)

            # Now we obtain the sum
            total = np.sum(drift_terms + diffusion_terms, axis=0)
            return total


    #if (use_gpu and tensorise):
    #    dt = tf.constant(dt, dtype = tf.float32)

    if method == "RK4":
        # Now we compute the rhs    
        k1 = dt * rhs(P, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf)
        k2 = dt * rhs(P + 0.5 * k1, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf)
        k3 = dt * rhs(P + 0.5 * k2, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf)
        k4 = dt * rhs(P + k3, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf)

        return P + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    elif method == "Butcher":
        k1 = dt * rhs(P, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf)
        k2 = dt * rhs(P + 1/4 * k1, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf)
        k3 = dt * rhs(P + 1/8 * k1 + 1/8 * k2, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf)
        k4 = dt * rhs(P - 1/2 * k2 + k3, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf)
        k5 = dt * rhs(P + 3/16 * k1 + 9/16 * k4, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf)
        k6 = dt * rhs(P - 3/7 * k1 + 2/7 * k2 + 12/7 * k3 - 12/7 * k4 + 8/7 * k5, drift1_tf, drift2_tf, diffussion1_tf, diffussion2_tf, diffussion3_tf)

        return P + (7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)/90

# %% [markdown]
# ## CRANK-NICOLSON

# %% [markdown]
# ### NEW METHOD OF DEFINING THE L OPERATOR

# %%
def gradient_matrix_1d(n, dx):

    D = lil_matrix((n, n))

    # forward difference
    D[0,0] = -1/dx
    D[0,1] = 1/dx

    # central differences
    for i in range(1, n-1):
        D[i, i-1] = -1/(2*dx)
        D[i, i+1] = 1/(2*dx)

    # backward difference
    D[n-1, n-2] = -1/dx
    D[n-1, n-1] = 1/dx

    return D.tocsr()

def gradient2_matrix_1d(n, dx):

    D = gradient_matrix_1d(n, dx)

    return D @ D

# %%
def gradient_operator(axis, grid_shape, dx):

    ops = []

    for i,n in enumerate(grid_shape):

        if i == axis:
            ops.append(gradient_matrix_1d(n, dx))
        else:
            ops.append(identity(n, format="csr"))

    G = ops[0]

    for op in ops[1:]:
        G = kron(G, op, format="csr")

    return G

# %%
def gradient2_operator(axis, grid_shape, dx):

    G = gradient_operator(axis, grid_shape, dx)

    return G @ G

# %%
def tensor_to_diag(T):

    return diags(T.flatten(), format="csr")

# %%
### And a function that writes the diagonal from a tensor A
def diagonal_from_tensor(A):

    return diags(A.flatten(), format="csr")

# %%
def build_rhs_matrix(
        grid_shape = points_per_neuron,
        differentials_per_neuron = differentials_per_neuron,
        drift1_tf = drift1,
        drift2_tf = drift2,
        diff1_tf = diffussion1,
        diff2_tf = diffussion2,
        diff3_tf = diffussion3):

    N = np.prod(grid_shape)

    L = csr_matrix((N,N))

    num_neurons = np.sum(neurons_per_layer)

    for neuron_idx in range(num_neurons):

        axis = reversed_list[neuron_idx]
        dx = differentials_per_neuron[neuron_idx]

        D = gradient_operator(axis, grid_shape, dx)
        D2 = D @ D

        a = drift1_tf[neuron_idx] + diff1_tf[neuron_idx]
        b = drift2_tf[neuron_idx] + diff2_tf[neuron_idx]
        c = diff3_tf[neuron_idx]

        A = tensor_to_diag(a)
        B = tensor_to_diag(b)
        C = tensor_to_diag(c)

        L += A + B @ D + C @ D2

    return L

# %% [markdown]
# ### Now we include the function that we are using FOR TESTING !

# %%
### And we define the function rhs just to see
def RHS_Func(P, drift1_tf = drift1, drift2_tf = drift2, 
             diffussion1_tf = diffussion1, diffussion2_tf = diffussion2, diffussion3_tf = diffussion3):
            P_expanded = [P]*sum(neurons_per_layer)
            P_stacked = np.stack(P_expanded, axis = 0)

            spatders1 = []
            spatders2 = []
            for neuron_idx in range(sum(neurons_per_layer)):
                differential = differentials_per_neuron[neuron_idx]
                axis = reversed_list[neuron_idx]

                # Now we compute the derivatives
                sd1 = spat_der(P, da = differential, axis = axis)
                sd2 = spat_sec_der(P, da = differential, axis = axis)

                spatders1.append(sd1)
                spatders2.append(sd2)

            spatders1 = np.stack(spatders1, axis = 0)  # Shape: [num_neurons, ...]
            spatders2 = np.stack(spatders2, axis = 0)  # Shape: [num_neurons, ...]

            # Vectorised computation of drift and diffusion terms
            drift_terms = drift1_tf * P_stacked + drift2_tf * spatders1
            diffusion_terms = (diffussion1_tf * P_stacked + 
                                diffussion2_tf * spatders1 + 
                                diffussion3_tf * spatders2)

            # Now we obtain the sum
            total = np.sum(drift_terms + diffusion_terms, axis=0)
            return total

# %%
# We measure how long it takes us to obtain the LOperator

import time
t1 = time.time()
LOperator = build_rhs_matrix()
t2 = time.time()
print("Elapsed time: ", t2-t1)

P_vec = P.flatten()

rhs_matrix = LOperator @ P_vec

rhs_tensor = rhs_matrix.reshape(P.shape)

error = np.max(np.abs(rhs_tensor - RHS_Func(P)))

print("Error between matrix definition and function:", error)

# %% 
# Now we check the eigenvalues of our matrix
from scipy.sparse.linalg import eigs
#largest_eigenvalues = eigs(LOperator, k = 6, which = "LM", return_eigenvectors = False)
#smallest_eigenvalues = eigs(LOperator, k = 6, which = "SM", return_eigenvectors = False)
#smallest_eigenvalues = eigs(LOperator, k = 6, sigma = -np.max(capacitors_per_layer), return_eigenvectors = False)
print("Capacitors per layer: ", capacitors_per_layer)
#print("Largest eigenvalues: ", largest_eigenvalues)
#print("Smallest eigenvalues: ", smallest_eigenvalues)

# %% [markdown]
# ### We confirm that this operator indeed satisfies this

# %%
print(np.max(np.abs(LOperator.sum(axis=0))))

# %%
print("Sum of the diagonal LOperator: ", LOperator.diagonal().sum() )

# %% [markdown]
# ### Let us fix the L operator through a QUICK FIX

# %%
# Quick fix for the operator L to conserve probability
L = LOperator.tolil()

for j in range(L.shape[1]):
    col_sum = L[:, j].sum()
    L[j, j] -= col_sum

L = L.tocsr()

print("Sum of rows for fixed operator: ",np.max(np.abs(L.sum(axis=0))))

# %%
print("Maximum value of the fixed L operator: ", np.max(L.data))
print("Minimum value of the fixed L operator: ", np.min(L.data))

# %%
L.diagonal().sum()

# %% [markdown]
# ### Now we replace with the new operator

# %%
# Now we once again test the rhs with the fixed operator
rhs_matrix_fixed = L @ P_vec
rhs_tensor_fixed = rhs_matrix_fixed.reshape(P.shape)
errorFixed = np.max(np.abs(rhs_tensor_fixed - RHS_Func(P)))
print("Error between Fixed matrix definition and function:", error)
#LOperator = L

# %% [markdown]
# ### Now our new operator conserves probability! Which works great !

# %% [markdown]
# ### Now we define the A and B matrices to solve

# %%
# A and B matrix
Id_Matrix = identity(LOperator.shape[0], format = "csr")
A_matrix = Id_Matrix - 0.5 * dt * LOperator
B_matrix = Id_Matrix + 0.5 * dt * LOperator

# We factorise the A_matrix
if (sum(neurons_per_layer) <=2):
    t1 = time.time()
    lu_matrix = splu(A_matrix.tocsc())
    t2 = time.time()
    print("Time taken to do the LU decomposition of the A matrix: ", t2-t1)
else:
    t1 = time.time()
    lu_matrix = None
    ilu = spilu(A_matrix.tocsc(), drop_tol = 1e-4, fill_factor = 10)
    M = LinearOperator(A_matrix.shape, ilu.solve)
    t2 = time.time()
    print("Time taken to do the numerical LU decomposition: ", t2-t1)

# Now we do the whole case GLOBALLY
#t1 = time.time()
#lu_matrix = None
#ilu = spilu(A_matrix.tocsc(), drop_tol = 1e-8, fill_factor = 30)
#M = LinearOperator(A_matrix.shape, ilu.solve)
#t2 = time.time()
#print("Time taken to do the numerical LU decomposition: ", t2-t1)

# %% [markdown]
# ### Now we define the Crank Nicolson Step

# %%

# Using PYAMG!!!
A_matrix = A_matrix.tocsr()
ml = pyamg.ruge_stuben_solver(A_matrix)
def crank_nicolson_step(P, A_matrix, B_matrix, lu):

    Pold = P.ravel()
    rhs = B_matrix @ Pold
    if lu == None:
        #Pnew, info = bicgstab(A_matrix, rhs, M = M, rtol = 1e-8, atol = 0)
        #if info != 0:
        #  print("Warning: solver did not converge. Info =", info)
        Pnew = ml.solve(rhs, tol=1e-8)
    else:
        Pnew = lu.solve(rhs)
    Pnew = Pnew.reshape(P.shape)
    #print("Pnew shape: ", Pnew.shape)
    return Pnew

# %% [markdown]
# ## Now we define the Krylov solver step:

# %%
def krylov_step(P, L = LOperator):
    Pvec = P.flatten()
    result = expm_multiply(L*dt, Pvec)
    return result.reshape(P.shape)

# %% [markdown]
# ## Now we define the step for the RADAU solver

# %%
def IRK_step(P, L = LOperator, dt = dt, A=None, b=None):
    """
    Perform a single implicit Runge-Kutta step for dP/dt = L P.

    Parameters
    ----------
    L : NxN sparse matrix
        Fokker-Planck operator.
    P0 : array, shape (N,)
        Current PDF (1D array).
    dt : float
        Time step.
    A : array, shape (s,s), optional
        Butcher matrix of the IRK method (s x s). Defaults to 3-stage Radau IIA (5th order).
    b : array, shape (s,), optional
        Weights for stage combination. Defaults to 3-stage Radau IIA (5th order).

    Returns
    -------
    P1 : array, shape (N,)
        PDF after a single time step.
    """
    
    # Flatten P to 1D for Kronecker system
    P_flat = P.ravel()
    N = P_flat.size

    # Default 3-stage Radau IIA
    if A is None or b is None:
        b = np.array([ (16 - np.sqrt(6))/36, (16 + np.sqrt(6))/36, 1/9 ])
        A = np.array([
            [ (88 - 7*np.sqrt(6))/360, (296 - 169*np.sqrt(6))/1800, (-2 + 3*np.sqrt(6))/225 ],
            [ (296 + 169*np.sqrt(6))/1800, (88 + 7*np.sqrt(6))/360, (-2 - 3*np.sqrt(6))/225 ],
            [ (16 - np.sqrt(6))/36, (16 + np.sqrt(6))/36, 1/9 ]
        ])
    
    s = len(b)
    
    # Identity matrices
    I_N = eye(N, format='csc')
    I_s = eye(s, format='csc')
    
    # Build Kronecker block system
    big_mat = kron(I_s, I_N) - dt * kron(A, L)
    #rhs = np.tile(P_flat, s)
    rhs = np.tile(L @ P_flat, s)

    # Solve for stages
    K_vec = spsolve(big_mat, rhs)
    K_stages = K_vec.reshape(s, N)

    # Combine stages
    #P_new_flat = P_flat + np.sum(b[:, None] * K_stages, axis=0)
    P_new_flat = P_flat + dt * np.sum(b[:, None] * K_stages, axis=0)
    #P_new_flat /= P_new_flat.sum()  # normalize

    # Reshape back to original multi-dimensional shape
    P_new = P_new_flat.reshape(P.shape)
    return P_new

# %% [markdown]
# ### And an iterative case for large networks

# %%
def IRK_step_iterative(P0, L = LOperator, dt = dt, A=None, b=None, solver='gmres', tol=1e-8, maxiter=None):
    """
    Perform a single implicit Runge-Kutta step for dP/dt = L P using iterative sparse solvers.

    Parameters
    ----------
    L : NxN sparse matrix
        Fokker-Planck operator.
    P0 : array, shape (N,)
        Current PDF (1D array).
    dt : float
        Time step.
    A : array, optional
        Butcher matrix of the IRK method (s x s). Defaults to 3-stage Radau IIA (5th order).
    b : array, optional
        Weights (s,). Defaults to 3-stage Radau IIA (5th order).
    solver : str
        'gmres' or 'bicgstab'.
    tol : float
        Tolerance for iterative solver.
    maxiter : int or None
        Maximum iterations for iterative solver.

    Returns
    -------
    P1 : array
        PDF after a single time step.
    """
    N = P0.size
    P = P0.ravel().copy()
    
    if not isinstance(L, csc_matrix):
        L = csc_matrix(L)
    
    # Default 3-stage Radau IIA (5th order)
    if A is None or b is None:
        b = np.array([ (16 - np.sqrt(6))/36, (16 + np.sqrt(6))/36, 1/9 ])
        A = np.array([
            [ (88 - 7*np.sqrt(6))/360, (296 - 169*np.sqrt(6))/1800, (-2 + 3*np.sqrt(6))/225 ],
            [ (296 + 169*np.sqrt(6))/1800, (88 + 7*np.sqrt(6))/360, (-2 - 3*np.sqrt(6))/225 ],
            [ (16 - np.sqrt(6))/36, (16 + np.sqrt(6))/36, 1/9 ]
        ])
    
    s = len(b)
    
    # Linear operator for Kronecker system
    def matvec(K_vec):
        K_stages = K_vec.reshape(s, N)
        out = np.zeros_like(K_stages)
        for i in range(s):
            sum_term = np.zeros(N)
            for j in range(s):
                sum_term += dt * A[i,j] * L.dot(K_stages[j])
            out[i] = K_stages[i] - sum_term
        return out.ravel()
    
    big_op = LinearOperator((s*N, s*N), matvec=matvec, dtype=np.float64)
    
    # Solve the linear system for this single step
    rhs = np.tile(L @ P, s)
    
    if solver.lower() == 'gmres':
        #K_vec, info = gmres(big_op, rhs, tol=tol, maxiter=maxiter)
        K_vec, info = gmres(big_op, rhs, rtol=tol, atol=0.0, maxiter=maxiter)
    elif solver.lower() == 'bicgstab':
        K_vec, info = bicgstab(big_op, rhs, tol=tol, maxiter=maxiter)
    else:
        raise ValueError("solver must be 'gmres' or 'bicgstab'")
    
    if info != 0:
        print(f"Warning: Iterative solver did not converge (info={info})")
    
    K_stages = K_vec.reshape(s, N)
    
    P1 = P + dt * np.sum(b[:, None] * K_stages, axis=0)
    #P1 /= P1.sum()  # normalize PDF
    
    return P1.reshape(P0.shape)

# %% [markdown]
# ## Backward Differentiation (BDF) Method

# %%
# We create a function that solves the BDF on the rhs
def implicit_solve(alpha0, rhs, L, dt, solver='gmres', tol=1e-8, maxiter=None):

    rhs = rhs.ravel()
    N = rhs.size

    def matvec(x):
        return alpha0*x - dt*(L @ x)

    Aop = LinearOperator((N, N), matvec=matvec)

    if solver.lower() == 'gmres':
        x, info = gmres(Aop, rhs, rtol=tol, atol=0.0, maxiter=maxiter)
    else:
        x, info = bicgstab(Aop, rhs, rtol=tol, atol=0.0, maxiter=maxiter)

    if info != 0:
        print("Warning: iterative solver did not converge")

    return x.reshape(rhs.shape)

# %% [markdown]
# ### And now the BDF methods

# %%
def BDF2_step(Pn, Pnm1, L, dt, **kwargs):
    # Flatten arrays to 1D
    Pn   = Pn.ravel()
    Pnm1 = Pnm1.ravel()
    Pnm2 = Pnm2.ravel()
    Pnm3 = Pnm3.ravel()
    Pnm4 = Pnm4.ravel()
    
    rhs = 2*Pn - 0.5*Pnm1
    alpha0 = 3/2

    return implicit_solve(alpha0, rhs, L, dt, **kwargs)

def BDF3_step(Pn, Pnm1, Pnm2, L, dt, **kwargs):
    # Flatten arrays to 1D
    Pn   = Pn.ravel()
    Pnm1 = Pnm1.ravel()
    Pnm2 = Pnm2.ravel()
    Pnm3 = Pnm3.ravel()
    Pnm4 = Pnm4.ravel()

    rhs = 3*Pn -1.5*Pnm1 +(1/3)*Pnm2
    alpha0 = 11/6

    return implicit_solve(alpha0, rhs, L, dt, **kwargs)

def BDF4_step(Pn, Pnm1, Pnm2, Pnm3, L, dt, **kwargs):
    # Flatten arrays to 1D
    Pn   = Pn.ravel()
    Pnm1 = Pnm1.ravel()
    Pnm2 = Pnm2.ravel()
    Pnm3 = Pnm3.ravel()
    Pnm4 = Pnm4.ravel()

    rhs = 4*Pn - 3*Pnm1 + (4/3)*Pnm2 - (1/4)*Pnm3
    alpha0 = 25/12

    return implicit_solve(alpha0, rhs, L, dt, **kwargs)


def BDF5_step(Pn, Pnm1, Pnm2, Pnm3, Pnm4, L, dt, **kwargs):
    # Flatten arrays to 1D
    Pn   = Pn.ravel()
    Pnm1 = Pnm1.ravel()
    Pnm2 = Pnm2.ravel()
    Pnm3 = Pnm3.ravel()
    Pnm4 = Pnm4.ravel()
    
    rhs = 5*Pn-5*Pnm1+(10/3)*Pnm2-(5/4)*Pnm3+(1/5)*Pnm4
    alpha0 = 137/60

    return implicit_solve(alpha0, rhs, L, dt, **kwargs)

# %%
# Now we write a solver for RK4
def RK4_matrix(P, L):
    P_flat = P.ravel().copy()
    def rhs(P = P_flat, L = L):
        # Now in this version we just multiply on the right hand side
        rhs_result = L @ P_flat
        return rhs_result
        
    # Now we compute the rhs    
    k1 = dt * rhs(P_flat)
    k2 = dt * rhs(P_flat + 0.5 * k1)
    k3 = dt * rhs(P_flat + 0.5 * k2)
    k4 = dt * rhs(P_flat + k3)
    P_final = P_flat + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return P_final.reshape(P.shape)

# %% [markdown]
# ## Now we write the PDF for the WHOLE network

# %%
gbn_pdf = 1
for neuron_idx in range(sum(neurons_per_layer)):
    const_norm = 1/(np.sqrt(2*np.pi*sigma_per_neuron[neuron_idx])) * np.abs(fp(mesh_network[neuron_idx], activation = activations_per_neuron[neuron_idx]))
    diff = f(mesh_network[neuron_idx], activation = activations_per_neuron[neuron_idx]) - (external_input_per_neuron[neuron_idx] + input_from_neurons[neuron_idx])
    diff = diff **2 / (2*sigma_per_neuron[neuron_idx])

    pdf_idx = const_norm * np.exp(-diff)
    #pdf_idx = pdf_idx / np.sum(pdf_idx)
    
    # We now multiply the whole thing
    gbn_pdf = gbn_pdf * pdf_idx

# Now we normalise the whole thing
Q_gbn_pdf = np.sum(gbn_pdf) * np.prod(differentials_per_neuron)
gbn_pdf = gbn_pdf / Q_gbn_pdf
gbn_pdf = gbn_pdf * norm_const

# %%
# Now we write our equivalent "GBN PDF for our system"
gbn_equiv_pdf = 1
for neuron_idx in range(sum(neurons_per_layer)):
    const_norm = np.sqrt( capacitors_per_neuron[neuron_idx]/(np.pi*resistors_per_neuron[neuron_idx]*effective_sigma_per_neuron[neuron_idx]) )
    const_norm = const_norm * np.abs(fp(mesh_network[neuron_idx], activation = activations_per_neuron[neuron_idx]))
    diff = (f(mesh_network[neuron_idx], activation = activations_per_neuron[neuron_idx]) 
            - resistors_per_neuron[neuron_idx]*(external_input_per_neuron[neuron_idx] + input_from_neurons[neuron_idx]))
    diff = (capacitors_per_neuron[neuron_idx]/(resistors_per_neuron[neuron_idx]*effective_sigma_per_neuron[neuron_idx])) * (diff **2)

    pdf_idx = const_norm * np.exp(-diff)
    #pdf_idx = pdf_idx / np.sum(pdf_idx)
    
    # We now multiply the whole thing
    gbn_equiv_pdf = gbn_equiv_pdf * pdf_idx

# Now we normalise the whole thing
Q_gbn_equiv_pdf = np.sum(gbn_equiv_pdf) * np.prod(differentials_per_neuron)
gbn_equiv_pdf = gbn_equiv_pdf / Q_gbn_equiv_pdf
gbn_equiv_pdf = gbn_equiv_pdf * norm_const

# %% Now we compare the differences in PDFs between our equivalent formulation and the correct one

diff_between_pdfs = np.max(gbn_pdf - gbn_equiv_pdf)
print("Maximum difference between the PDFs", diff_between_pdfs)

# And now we test the L operator acting on the maximum
P_pdf_flat = gbn_pdf.flatten()
P_pdf_equiv_flat = gbn_equiv_pdf.flatten()

# And now we test on both sides
rhs_gbn = LOperator @ P_pdf_flat
rhs_equiv_gbn = LOperator @ P_pdf_equiv_flat

print("*********************************************************************************")
print("Capacitors per layer: ", capacitors_per_layer)
print("LOperator acting on GBN's PDF maximum: ", np.max(rhs_gbn))
print("LOperator acting on GBN's EQUIV PDF maximum: ", np.max(rhs_equiv_gbn))

# And now we test on the function RHS !!!
print("Function RHS acting on GBN PDF: ", np.max(RHS_Func(gbn_pdf)) )
print("Function RHS acting on GBNs equiv PDF: ", np.max(RHS_Func(gbn_equiv_pdf)) )
print("*********************************************************************************")
# %%
# And now we set our model to be the "target PDF"
# We can always comment this part out
gbn_pdf = gbn_equiv_pdf

# %%
### Vector to save the whole KL div 
total_kl_div = np.zeros(shape = (n_graph, ))

# %% [markdown]
#  ### Now we create the KL divergence function

# %%
def safe_kl_divergence(P, Q, dimensions = True ,epsilon=1e-20):
    """
    Compute the KL divergence D_KL(P || Q) between two discrete distributions P and Q.
    If Q has zeros where P is positive, a small epsilon is added to Q to avoid explosion.
    
    Parameters:
        P (np.ndarray): The first probability vector (must sum to 1).
        Q (np.ndarray): The second probability vector (must sum to 1).
        epsilon (float): Small constant to avoid division by zero.
    
    Returns:
        float: KL divergence value.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Now we reshape
    P = P.reshape(Q.shape)
    
    # Add small epsilon to Q to avoid division by zero
    Q_safe = Q + epsilon
    P_safe = P  + epsilon
    
    # Renormalise
    if dimensions:
        ZQsafe = np.sum(Q_safe) * np.prod(differentials_per_neuron)
        ZPsafe = np.sum(P_safe) * np.prod(differentials_per_neuron)
    else:
        ZQsafe = np.sum(Q_safe) * differentials_per_neuron[dimensions]
        ZPsafe = np.sum(P_safe) * differentials_per_neuron[dimensions]
    
    Q_safe = Q_safe / ZQsafe
    P_safe = P_safe / ZPsafe
    
    # Only compute over entries where P > 0
    #mask = P > 0
    
    #kl_div = np.sum(P[mask] * np.log(P[mask] / Q_safe[mask]) )
    kl_div = np.sum(P_safe * np.log(P_safe / Q_safe) )
    
    return kl_div

# %%
### Now we plot if we have two neurons
if (sum(neurons_per_layer)==2):
    plt.contourf(mesh_network[0], mesh_network[1], P)
    plt.title("Initial PDF for two neurons")

# %% [markdown]
#  # Now we evolve

# %%
number_jumps = 0

min_t = 0 # Initial time_step
if solution_method.lower() == "bdf":
    # We first get the four steps !
    # Initial PDF
    P0 = P
    P1 = crank_nicolson_step(P = P0, A_matrix = A_matrix, B_matrix = B_matrix, lu = lu_matrix)
    P2 = crank_nicolson_step(P = P1, A_matrix = A_matrix, B_matrix = B_matrix, lu = lu_matrix)
    P3 = crank_nicolson_step(P = P2, A_matrix = A_matrix, B_matrix = B_matrix, lu = lu_matrix)
    P = crank_nicolson_step(P = P3, A_matrix = A_matrix, B_matrix = B_matrix, lu = lu_matrix)
    min_t = 4

    # We save the initial points ! 
    # We compute the general KL Div
    total_kl_div_timestamp = safe_kl_divergence(P = P0, Q = gbn_pdf, dimensions = True)
    total_kl_div[number_jumps] = total_kl_div_timestamp
    print("Total KL divergence at the beginning: ", total_kl_div_timestamp)

    for neuron_idx in range(sum(neurons_per_layer)):
        marg_pdf = marginalize(P, axis_to_keep = reversed_list[neuron_idx])

        # And now we normalise this PDF !
        Q_marg_pdf = np.sum(marg_pdf) * differentials_per_neuron[neuron_idx]
        marg_pdf = marg_pdf / Q_marg_pdf

        # We save the result
        dynamical_pdf_per_neuron[neuron_idx][number_jumps, :] = marg_pdf

    number_jumps = 1


# %%
# And NOW we iterate over the time steps

t1 = time.time()
for t_step in range(min_t, n_t+1):
    #print(f"Time step: {t_step}")

    if ((sum(neurons_per_layer) <= 3)):
        if (solution_method.lower() == "krylov"):
             result = krylov_step(P = P)
        elif (solution_method.lower() == "cn"):
            result = crank_nicolson_step(P = P, A_matrix = A_matrix, B_matrix = B_matrix, lu = lu_matrix)
        elif (solution_method.lower() == "irk"):
            #result = IRK_step(P = P)
            result = IRK_step_iterative(P0 = P)
        elif (solution_method.lower() == "bdf"):
            result = BDF5_step(P, P3, P2, P1, P0, L = LOperator, dt = dt)
            # And we nos update the Ps
            P0, P1, P2, P3, P = P1, P2, P3, P, result
        elif (solution_method.lower() == "rk4"):
            result = RK4_matrix(P = P, L = LOperator)
    else:
        result = num_solve(P, dt = dt, drift1_tf = drift1, drift2_tf = drift2, 
                             diffussion1_tf = diffussion1, diffussion2_tf = diffussion2, diffussion3_tf = diffussion3, 
                             use_gpu = use_gpu, tensorise = tensorise,
                             method= solution_method)

    P = result
    # We eliminate negatives
    P = np.where(P < 0, 0, P)

    #result = np.abs(result)
    #result = np.maximum(result, 0)

    # We normalise
    Z = np.sum(P) * np.prod(differentials_per_neuron)
    P = P / Z
    # Now we include the normalisation constant
    P = P * norm_const
    #print("P shape: ", P.shape)

    if (t_step in idxs_layer1):
        print("Number of jumps: ", number_jumps)
        print("Norm of P without normalising: ", np.sum(P)*np.prod(differentials_per_neuron))
        # Now we save the whole "actual pdf"
        
        # We compute the general KL Div
        total_kl_div_timestamp = safe_kl_divergence(P = P, Q = gbn_pdf, dimensions = True)
        total_kl_div[number_jumps] = total_kl_div_timestamp
        print("Total KL divergence time_stamp: ", total_kl_div_timestamp)

        for neuron_idx in range(sum(neurons_per_layer)):
            P = P.reshape(gbn_pdf.shape)
            marg_pdf = marginalize(P, axis_to_keep = reversed_list[neuron_idx])

            # And now we normalise this PDF !
            Q_marg_pdf = np.sum(marg_pdf) * differentials_per_neuron[neuron_idx]
            marg_pdf = marg_pdf / Q_marg_pdf

            # We save the result
            dynamical_pdf_per_neuron[neuron_idx][number_jumps, :] = marg_pdf
        number_jumps = number_jumps + 1

t2 = time.time()
print("Time elapsed evolving: ", t2-t1)
# %%
128 * 128 * 128

# %%
2048 * 2048

# %%
marg_pdf

# %% [markdown]
#  ### Now we visualise

# %%
if (sum(neurons_per_layer)==2):
    plt.contourf(mesh_network[0], mesh_network[1], P)
    plt.title("Final")
    plt.savefig("FinalNumericalP.pdf", format = "pdf", bbox_inches='tight')

# %%
if (sum(neurons_per_layer)==2):
    plt.contourf(mesh_network[0], mesh_network[1], gbn_pdf)
    plt.title("GBN's PDF")
    plt.savefig("FinalGBNP.pdf", format = "pdf", bbox_inches='tight')

# %% [markdown]
#  ### We write the closed-form pdf PER Neuron

# %%
analytical_pdf_per_neuron = []
for neuron_idx in range(sum(neurons_per_layer)):
    const_norm = 1/(np.sqrt(2*np.pi*sigma_per_neuron[neuron_idx])) * np.abs(fp(vectors_per_neuron[neuron_idx], activation = activations_per_neuron[neuron_idx]))
    diff = f(vectors_per_neuron[neuron_idx], activation = activations_per_neuron[neuron_idx]) - (external_input_per_neuron[neuron_idx] + input_from_neurons[neuron_idx])
    diff = diff **2 / (2*sigma_per_neuron[neuron_idx])

    pdf_idx = const_norm * np.exp(-diff)
    pdf_idx = pdf_idx / np.sum(pdf_idx)
    # Now we save the result
    analytical_pdf_per_neuron.append(pdf_idx)


# %%
analytical_pdf_per_neuron = []
# Now we just marginalise over the joint pdf
for neuron_idx in range(sum(neurons_per_layer)):
    pdf_idx = marginalize(P = gbn_pdf, axis_to_keep = reversed_list[neuron_idx])

    # Now we normalise
    Q_pdf_idx = np.sum(pdf_idx) * differentials_per_neuron[neuron_idx]
    pdf_idx = pdf_idx / Q_pdf_idx
    
    # Now we save the result
    analytical_pdf_per_neuron.append(pdf_idx)

# %% [markdown]
#  ### We create a vector for KL divergences

# %%
kl_divergences = np.zeros((sum(neurons_per_layer), n_graph))

# %% [markdown]
# ### Now we compute the KL divergences

# %%
for neuron_idx in range(sum(neurons_per_layer)):
    # The true probability distribution
    Pneuron = analytical_pdf_per_neuron[neuron_idx]
    for step_time in range(n_graph):
        # The model probability distribution
        Qneuron = dynamical_pdf_per_neuron[neuron_idx][step_time,:]

        div = safe_kl_divergence(P = Pneuron, Q = Qneuron, dimensions = neuron_idx)

        # Now we save the result 
        kl_divergences[neuron_idx, step_time] = div

# %% [markdown]
# ### Now we graph

# %%
cumsum = np.cumsum(neurons_per_layer)
neuron_idxs = [0]
neuron_idxs.extend(cumsum[0:-1])

# %%
neurons_per_layer == [1, 1]

# %% [markdown]
#   ### We obtain the final pdfs

# %%
final_pdf_per_neuron = []

for neuron_idx in range(sum(neurons_per_layer)):
    pdf = marginalize(P, axis_to_keep = reversed_list[neuron_idx])

    Q_pdf = np.sum(pdf) * differentials_per_neuron[neuron_idx]
    pdf = pdf / Q_pdf
    final_pdf_per_neuron.append(pdf)

#reversed_list = list( reversed(range(sum(neurons_per_layer))) )

# %%
size_reference = 100

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (12,8))
fig.suptitle(f"Comparison between a stochastic artificial neuron and a DSP neuron", fontsize= 22, x = 0.5)

axes[0, 0].imshow(np.reshape(input_vector, (28, 28)), 
                    cmap = "gray"
                    )
axes[0, 0].set_title("Input Image", fontsize = 16)
axes[0, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)



axes[1, 0].set_title("Architecture", fontsize = 16)
axes[1, 0].imshow(img)
axes[1, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


axes[0, 1].set_title(f"PDFs, nt = {n_t}, capacitors = {capacitors_per_layer}, points per space = {points_per_layer}", fontsize = 16)
for idx in graphing_idxs:
    #axes[0, 1].plot(vectors_per_neuron[idx], pdf_per_neuron[idx], label = f"Neuron {labels[neuron_idx]}'s samples", color = colors[idx], linestyle = "dotted")
    axes[0, 1].plot(vectors_per_neuron[idx], analytical_pdf_per_neuron[idx] , 
                    label = f"Neuron {labels[neuron_idx]}'s analytical PDF", color = colors[idx], linestyle = "dashed")
    axes[0, 1].plot(vectors_per_neuron[idx], final_pdf_per_neuron[idx] , label = f"DSP neuron {labels[neuron_idx]}'s PDF", color = colors[idx],
                    linestyle = "solid")
axes[0, 1].set_xlabel("Output Voltage [V]")
axes[0, 1].set_ylabel("Probability")
axes[0, 1].grid()
axes[0, 1].legend()


axes[1, 1].set_title("KL divergence between the PDFs of a stochastic", fontsize = 16)
for neuron_idx in range(len(neuron_idxs)):
    sc = axes[1, 1].scatter(t1graph, kl_divergences[neuron_idx, :], label = f"Neuron {labels[neuron_idx]}", color = colors[neuron_idx], s = 25)
axes[1, 1].legend()
axes[1, 1].grid()
axes[1, 1].set_xlabel("Time [s.]")
axes[1, 1].set_ylabel("KL Divergence")
axes[1, 1].set_yscale('log')


width = 0.6  # Width of outer bars
inner_width = width * 0.6  # Inner bars narrower

#plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout(pad=1, w_pad=0.3, h_pad=0.8)
fig.savefig("ComparisonFP.pdf", format = "pdf", bbox_inches='tight')

# %% [markdown]
#  ### Now we compute the potential

# %% [markdown]
#  c = capacitors_per_neuron[0]
# 
#  rho = resistors_per_neuron[0]
# 
#  sigma = effective_sigma_per_neuron[0]
# 
#  #sigma = sigma_per_neuron[0]
# 
#  upsilon = input_from_neurons[0] + external_input_per_neuron[0]
# 
#  act = activations_per_neuron[0]
# 
# 
# 
#  # Now we compute
# 
#  phi = -c*rho/sigma*(upsilon**2) + c*rho/sigma*(f(vectors_per_neuron[0], activation = act)-rho*upsilon)**2
# 
#  phi = phi + np.log(sigma/(2*c**2)*1/np.abs(fp(vectors_per_neuron[0], activation = act))**2 ) + np.log( np.abs(fp(vectors_per_neuron[0], activation = act)) )
# 
# 
# 
#  # Now we obtain the pdf
# 
#  pdf_from_phi = np.exp(-phi)
# 
#  # We normalise
# 
#  pdf_from_phi = pdf_from_phi / np.sum(pdf_from_phi)

# %% [markdown]
#  ### Now we save the graphs separately

# %%
fig, ax = plt.subplots()

for idx in graphing_idxs:
    #plt.plot(vectors_per_neuron[idx], pdf_per_neuron[idx], label = f"Artificial neuron's samples", color = colors[idx], linestyle = "dotted")
    ax.plot(vectors_per_neuron[idx], analytical_pdf_per_neuron[idx] , 
                    label = f"Artificial neuron's analytical PDF", color = colors[idx], linestyle = "solid")
    ax.plot(vectors_per_neuron[idx], final_pdf_per_neuron[idx] , label = f"DSP neuron's PDF", color = colors[idx],
                    linestyle = "dotted")
    
#ax.plot(vectors_per_neuron[0], pdf_from_phi, label = "Potential distribution", color = "red", linestyle = "dashdot")


#ax.plot(x, y)

# --- 1. Force scientific notation on the y-axis ---
formatter = ScalarFormatter(useMathText=True)
# use 10^-3 scale
formatter.set_powerlimits((-3, -3))
ax.yaxis.set_major_formatter(formatter)

# --- 2. Control number of y-axis divisions (ticks) ---
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# Optional: show the power of ten (×10⁻³) as a separate label
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))


# Now we add the labels
ax.set_xlabel("Output Voltage [V]", fontsize = 12)
ax.set_ylabel("Probability", fontsize = 12)
ax.grid()
ax.legend(fontsize = 12)
plt.tight_layout(pad=1, w_pad=0.3, h_pad=0.8)
plt.savefig("PDF2.pdf", format = "pdf", bbox_inches='tight')
plt.show()

# %% [markdown]
#  ### And now we graph the KL divergence

# %% [markdown]
#  ### We do a linear regression

# %%
from scipy import stats
stats.linregress(t1graph, y = np.log(kl_divergences[0,:]))

# %%
from scipy import stats
slopes = []
intercepts = []
for neuron_idx in range(sum(neurons_per_layer)):
    regress = stats.linregress(t1graph, y = np.log(kl_divergences[neuron_idx, :]))
    a, b = regress.slope, regress.intercept
    #a, b = np.polyfit(t1graph, y, 1)
    # We save the results
    slopes.append(a)
    intercepts.append(b)

print("Slope (a):", a)
print("Intercept (b):", b)

# %%
fig, ax = plt.subplots()

for neuron_idx in range(len(neuron_idxs)):
    ax.scatter(t1graph, kl_divergences[neuron_idx, :], 
               #label = f"Neuron {labels[neuron_idx]}",
               label = "Numerical evolution",
               color = colors[neuron_idx], s = 10)
    # We now include the linear regresor
    ax.plot(t1graph, np.exp(intercepts[neuron_idx]) * np.exp(t1graph*slopes[neuron_idx]), 
             label = f"Linear regression, slope = {slopes[neuron_idx]:.2e}", 
             color = colors[neuron_idx])


#ax.plot(x, y)

# --- 1. Force scientific notation on the y-axis ---
formatter = ScalarFormatter(useMathText=True)
# use 10^-3 scale
formatter.set_powerlimits((-3, -3))
ax.yaxis.set_major_formatter(formatter)

# --- 2. Control number of y-axis divisions (ticks) ---
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# Optional: show the power of ten (×10⁻³) as a separate label
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))


# Now we add the labels
ax.set_xlabel(r"Time [s.]", fontsize = 12)
ax.set_ylabel("DKL", fontsize = 12)
ax.set_yscale('log')
ax.grid()
ax.legend(fontsize = 12)
plt.tight_layout(pad=1, w_pad=0.3, h_pad=0.8)
plt.savefig("KLDiv2.pdf", format = "pdf", bbox_inches='tight')
plt.show()

# %% [markdown]
#  ### Now we confirm the potential

# %%
c = capacitors_per_neuron[0]
rho = resistors_per_neuron[0]
sigma = effective_sigma_per_neuron[0]
#sigma = sigma_per_neuron[0]
upsilon = input_from_neurons[0] + external_input_per_neuron[0]
act = activations_per_neuron[0]

# Now we compute 
phi = -c*rho/sigma*(upsilon**2) + c*rho/sigma*(f(vectors_per_neuron[0], activation = act)-rho*upsilon)**2
phi = phi + np.log(sigma/(2*c**2)*1/np.abs(fp(vectors_per_neuron[0], activation = act))**2 ) + np.log( np.abs(fp(vectors_per_neuron[0], activation = act)) )

# Now we obtain the pdf
pdf_from_phi = np.exp(-phi)
# We normalise
pdf_from_phi = pdf_from_phi / np.sum(pdf_from_phi)

# %% [markdown]
#  ### We save the KL divergence and the pdfs and so on

# %%
# We save the vectors per neuron
pdf_dict = {f"{i}": w for i, w in enumerate(vectors_per_neuron)}
np.savez("vectors_per_neuron.npz", **pdf_dict)

# %%
# Now we save the pdf per neuron ("analytical pdf per neuron")
pdf_dict = {f"{i}": w for i, w in enumerate(analytical_pdf_per_neuron)}
np.savez("analytical_pdf_per_neuron.npz", **pdf_dict)

# %%
# Now we save the final PDFs per neuron from the numerical case
pdf_dict = {f"{i}": w for i, w in enumerate(final_pdf_per_neuron)}
np.savez("final_pdf_per_neuron.npz", **pdf_dict)

# %%
pdf_dict = {f"{i}": w for i, w in enumerate(pdf_from_phi)}
np.savez("pdf_from_phi.npz", **pdf_dict)

# %% [markdown]
#  ### Now we save the result of the KL divergence

# %%
pdf_dict = {f"{i}": w for i, w in enumerate(t1graph)}
np.savez("t1graph.npz", **pdf_dict)

# %%
### And the KL divergences themsPelves
pdf_dict = {f"{i}": w for i, w in enumerate(kl_divergences)}
np.savez("kl_divergences.npz", **pdf_dict)

# %% [markdown]
#  ## Now we do the KL divergence for all neurons

# %% [markdown]
#  ### Final regression at the end

# %%
total_kl_div

# %%
from scipy import stats
slope = []
intercept = []
regress = stats.linregress(t1graph, y = np.log(total_kl_div))
a, b = regress.slope, regress.intercept

slope = a 
intercept = b

# %% [markdown]
#  ### Now we graph

# %%
plt.plot(figsize = (12,9))
plt.title("TOTAL KL Div", fontsize = 16)
plt.scatter(t1graph, total_kl_div, label = f"Total KL Div", color = "black", s = 10)
# We now include the linear regresor
plt.plot(t1graph, np.exp(intercept) * np.exp(t1graph*slope), 
             label = f"Linear regression, slope = {slope:.2e}", 
             color = "black")
plt.legend()
plt.grid()
plt.xlabel("Time [s.]", fontsize = 12)
plt.ylabel("$DKL", fontsize = 12)
plt.yscale('log')
plt.tight_layout(pad=1, w_pad=0.3, h_pad=0.8)
plt.savefig("KLDivTotal.pdf", format = "pdf", bbox_inches='tight')

# %% [markdown]
#  ### Now we save all of the data

# %%
pdf_dict = {f"{i}": w for i, w in enumerate(total_kl_div)}
np.savez("total_kl_div.npz", **pdf_dict)

# %% [markdown]
#  ### Now we save the regressions

# %%
# Individual KL div
pdf_dict = {f"{i}": w for i, w in enumerate(slopes)}
np.savez("slopes_kl_div_per_neuron.npz", **pdf_dict)

pdf_dict = {f"{i}": w for i, w in enumerate(intercepts)}
np.savez("intercepts_kl_div_per_neuron.npz", **pdf_dict)

# %%
pdf_dict = {f"{i}": w for i, w in enumerate([slope])}
np.savez("slope_kl_div_total.npz", **pdf_dict)

pdf_dict = {f"{i}": w for i, w in enumerate([intercept])}
np.savez("intercept_kl_div_total.npz", **pdf_dict)

# %%
P

# %%
gbn_pdf


