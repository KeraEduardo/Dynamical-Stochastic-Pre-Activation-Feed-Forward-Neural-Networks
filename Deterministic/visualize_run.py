# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
import os

# %%
import os
print("Available CPUs: ",os.cpu_count())

# We save the run_dir and run_id
run_dir = os.environ.get("RUN_DIR")
run_id = int(os.environ.get("RUN_ID"))

#run_dir = "C:/Users/eduar/OneDrive/Desktop/Yani/Fake/Regression"
#run_id = 1

# %%
setup_dict = {}

with open("/home/eduardo.martinezgarc/projects/Yani/Fake/Regression/Training/setup.txt") as f:
#with open("setup.txt") as f:
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
t_crit = max_time * 0.90
#n_jump = setup_dict["n_jump"]
points_per_layer = setup_dict["points_per_layer"]
#idx_data = setup_dict["idx_data"]
initial_sigma_per_layer = setup_dict["initial_sigma_per_layer"]

scale_y = setup_dict["scale_y"]
cross_validate = setup_dict["cross_validate"]
dataset = setup_dict["dataset"]

# We obtain the solution method
#use_gpu = setup_dict["use_gpu"]
#tensorise = setup_dict["tensorise"]
#derivative_mode = setup_dict["derivative_mode"]
#solution_method = setup_dict["solution_method"]
#neuron_idxs = setup_dict["neuron_idxs"]

# For the graph
#labels = setup_dict["labels"]
#colors = setup_dict["colors"]
#graphing_idxs = setup_dict["graphing_idxs"]

# For the times

n_samples_dynamical = setup_dict["n_samples_dynamical"]

# %% [markdown]
#  ### Now we import the data vector

# %%
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split

if dataset.lower() == "housing":
    data = fetch_california_housing()
elif dataset.lower() == "diabetes":
    data = load_diabetes()
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)   # NEW: also seeds Python's `random` -> more reproducible runs

X = data.data
y = data.target

# We only select the ones less than 5 IF USING CALIFORNIA HOUSING DATASET
if dataset.lower() == "housing":
    idxs = (y <= 4.99)
    X = X[idxs]
    y = y[idxs]
    
# %%
# Now we split into train and test
from sklearn.preprocessing import StandardScaler
#cross_validate = True
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

    scale_y = False
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
y_real = y_test[run_id]

data = np.load("/home/eduardo.martinezgarc/projects/Yani/Fake/Regression/Training/mlp_test_pred.npz", allow_pickle=True)
#data = np.load("mlp_test_pred.npz", allow_pickle=True)
mlp_test_pred = data["mlp_test_pred"]

y_pred = mlp_test_pred[run_id]

# %% [markdown]
#  ### For multiprocessing

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

# %%
def flatten(x):
    x = tf.reshape(x, [-1])           # Flatten to 1D first
    return tf.reshape(x, [-1, 1])     # Then reshape to (x, 1)

# %% [markdown]
# ### Now we load the probabilistic ANN results

# %%
data = np.load("/home/eduardo.martinezgarc/projects/Yani/Fake/Regression/Training/prob_ann_results.npz", allow_pickle=True)
#data = np.load("prob_ann_results.npz", allow_pickle=True)
prob_ann_results = data[str(run_id)]
prob_ann_results

# %% [markdown]
# ### Now we create the vectors for the neural network

# %%
data = np.load("samples_dynamical.npz", allow_pickle = True)
samples_dynamical = data["samples_dynamical"]

# %%
absolute_min = min(samples_dynamical.min(), y_pred, y_real)
absolute_max = max(samples_dynamical.min(), y_pred, y_real)

# %%
epsilon = 1e-16
delta_x = 5


def create_vector(activation, n_x, eps = epsilon, min_z = absolute_min - delta_x, max_z = absolute_max + delta_x):
    if activation == "sigmoid":
        #x = np.linspace(eps, 1-eps, n_x)
        x = np.linspace(0, 1, n_x)
    elif activation == "linear":
        x = np.linspace(min_z, max_z, n_x)
    elif activation == "tanh":
        x = np.linspace(-1 + eps, 1 - eps, n_x)
    elif activation == "relu":
        x = np.linspace(0, max_z, n_x)
    elif activation == "softplus":
        x = np.linspace(0, max_z, n_x)
    return x

# %% [markdown]
#   ### Now we define the functions we willbe using to plot

# %%
import scipy
from scipy.special import rel_entr

def safe_kl_divergence(P, Q):
    relative_entropy = scipy.special.rel_entr(P, Q)

    return np.sum(relative_entropy)

# %%
def safe_kl_divergence(P, Q, dx_vector, epsilon=1e-20):
    """
    Compute the KL divergence D_KL(P || Q) between two discrete distributions P and Q.
    If Q has zeros where P is positive, a small epsilon is added to Q to avoid explosion.
    
    Returns:
        float: KL divergence value.
    """
    
    # Add small epsilon to Q to avoid division by zero
    Q_safe = Q + epsilon
    P_safe = P  + epsilon
    
    #kl_div = np.sum(P[mask] * np.log(P[mask] / Q_safe[mask]) )
    #kl_div = np.sum(P_safe * np.log(P_safe / Q_safe) )
    
    integrand = P_safe * np.log(P_safe / Q_safe) 
    # Now we integrate

    return np.trapz(integrand, dx_vector)

# %%
1/(1e-20)

# %%
def l2_norm(x,y):
    return np.linalg.norm(x-y)

# %%
# Probability of X falling within an interval

def interval_probability(x, p, a, b):
    """
    Probability that X falls in the interval [a,b].
    """

    x = np.asarray(x)
    p = np.asarray(p)

    #p = p / p.sum()

    mask = (x >= a) & (x <= b)

    #return np.sum(p[mask])
    return np.trapz(p[mask], x[mask])

# %% [markdown]
#   ### Now we create the vectors per layer

# %%
vectors_per_layer = []
for layer_idx in range(len(neurons_per_layer)):
    result = create_vector(activation = activations_per_layer[layer_idx], n_x = points_per_layer[layer_idx] )
    vectors_per_layer.append( result )

activations_per_neuron = []
for layer_idx in range(len(neurons_per_layer)):
    # Now we save the results
    acts = [activations_per_layer[layer_idx]] * neurons_per_layer[layer_idx]
    activations_per_neuron.extend(acts)

# %%
dx_last_layer = np.abs(vectors_per_layer[-1][1] - vectors_per_layer[-1][0])

# %% [markdown]
#    ### Now we create a vector that stores the means of the samples and the L2 distances

# %%
means_dynamical = np.zeros((neurons_per_layer[-1], n_graph))
l2_dsp_ann = np.zeros((neurons_per_layer[-1], n_graph))
l2_dsp_data = np.zeros((neurons_per_layer[-1], n_graph))
l2_dsp_prob_ann = np.zeros((neurons_per_layer[-1], n_graph))

# %% [markdown]
#   ### Now we compute the pdfs per each neuron

# %%
def compute_pdf(samples, activation, number_points, min_z = absolute_min - delta_x, max_z = absolute_max + delta_x):
    if activation == "sigmoid":
        range_value = (0, 1)
    elif activation == "relu":
        range_value = (min_z, max_z)
    elif activation == "tanh":
        range_value = (-1, 1)
    elif activation == "softplus":
        range_value = (min_z, max_z)
    elif activation == "linear":
        range_value = (min_z, max_z)
    
    pdf_values, _ = np.histogram(samples, bins = number_points, range = range_value)

    return pdf_values

# %%
def new_compute_pdf(samples, vector):

    vector = np.array(vector)
    vector = np.append(vector, 1e16)
    pdf_values, _ = np.histogram(samples, bins = vector)

    return pdf_values

# %%
def new_compute_pdf(samples, vector):

    vector = np.asarray(vector)
    # We add a very long value
    vector = np.append(vector, 1e16)

    counts, edges = np.histogram(samples, bins=vector)

    dx = np.diff(edges)
    pdf = counts / (len(samples) * dx)

    return pdf

# %% [markdown]
#  ### Dynamical PDF

# %%
pdfs_dynamical = np.zeros((neurons_per_layer[-1], n_graph, points_per_layer[-1]))

for neuron_idx in range(neurons_per_layer[-1]):
    for time_idx in range(len(t_graph)):
            
            samples = samples_dynamical[neuron_idx, :, time_idx]
            
            # Now we obtain the sampless
            #pdf = compute_pdf(samples = samples, vec = layer_vector)
            #pdf = compute_pdf(samples = samples, activation = activations_per_layer[-1], number_points = points_per_layer[-1])
            pdf = new_compute_pdf(samples = samples, vector = vectors_per_layer[-1])

            # We save the PDF
            pdfs_dynamical[neuron_idx, time_idx, :] = pdf        

# %%
points_per_layer[-1]

# %% [markdown]
#  ### Now we fill the data that we are gonna be graphing

# %% [markdown]
#    ### The mean over time

# %%
for neuron_idx in range(neurons_per_layer[-1]):
    for time_idx in range(n_graph):

        mean = np.mean(samples_dynamical[neuron_idx, :, time_idx])

        # Now we do fill the vector
        means_dynamical[neuron_idx, time_idx] = mean

# %% [markdown]
#   ### And we also load the results we obtained manually

# %%
data = np.load("/home/eduardo.martinezgarc/projects/Yani/Fake/Regression/Training/manual_outputs.npz", allow_pickle=True)
#data = np.load("manual_outputs.npz", allow_pickle=True)
manual_test_pred = data[str(len(neurons_per_layer)-1)][0]

# %%
for neuron_idx in range(neurons_per_layer[-1]):
    for time_idx in range(n_graph):

        mse_ann = l2_norm(x = means_dynamical[neuron_idx, time_idx], y = y_pred)
        mse_data = l2_norm(x = means_dynamical[neuron_idx, time_idx], y = y_real)
        mse_prob_ann = l2_norm(x = means_dynamical[neuron_idx, time_idx], y = prob_ann_results.mean())

        # Now we fill the data
        l2_dsp_ann[neuron_idx, time_idx] = mse_ann 
        l2_dsp_data[neuron_idx, time_idx] = mse_data
        l2_dsp_prob_ann[neuron_idx, time_idx] = mse_prob_ann

# %% [markdown]
#    ### Now we calculate the probability of success

# %%
prob_real_succ = np.zeros((neurons_per_layer[-1], n_graph))
prob_ann_succ = np.zeros((neurons_per_layer[-1], n_graph))

# Now we fill it up !
for neuron_idx in range(neurons_per_layer[-1]):
    for time_idx in range(n_graph):

        # Now we obtain the probability of our result falling within either value!
        prob_ann_succ_neur = interval_probability(x = vectors_per_layer[-1], p = pdfs_dynamical[neuron_idx, time_idx, :],
                                                  a = y_pred - dx_last_layer, b = y_pred + dx_last_layer)

        prob_real_succ_neur = interval_probability(x = vectors_per_layer[-1], p = pdfs_dynamical[neuron_idx, time_idx, :],
                                                          a = y_real - dx_last_layer, b = y_real + dx_last_layer)

        # Now we save the results
        prob_real_succ[neuron_idx, time_idx] = prob_real_succ_neur
        prob_ann_succ[neuron_idx, time_idx] = prob_ann_succ_neur

# %% [markdown]
#   ### Now we calculate the prob ANN pdf

# %%
#prob_ann_pdf = compute_pdf(samples = prob_ann_results, activation = activations_per_layer[-1], number_points = points_per_layer[-1])
prob_ann_pdf = new_compute_pdf(samples = prob_ann_results, vector = vectors_per_layer[-1])

# %%
max_value_pdfs = max(pdfs_dynamical[0, -1, :].max(), prob_ann_pdf.max())

# %% [markdown]
#  ### Now we complute the kl divergence

# %%
kl_div_over_time = []
for time_idx in range(n_graph):
    kl_div_timestap = safe_kl_divergence(P = prob_ann_pdf, Q = pdfs_dynamical[0, time_idx, :], dx_vector = vectors_per_layer[-1], epsilon=1e-20)
    kl_div_over_time.append( kl_div_timestap )

kl_div_over_time = np.array(kl_div_over_time)

# %% [markdown]
#   ### Now we graph

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (12,8))

time = t_vec[-1]
time = str(time)[0:6]
fig.suptitle(f"Comparison between digital and dynamical neural networks at time {time}s. with {n_samples_dynamical} dynamical samples", fontsize=16, x = 0.5)


axes[0, 0].set_title("Mean over time")
axes[0, 0].plot(t1graph, means_dynamical[0,:], label = f"DSP, mean = {means_dynamical[0, -1]:.4f}", color = "black")
axes[0, 0].plot(np.array([t1graph[0],t1graph[-1]]), np.array([y_pred, y_pred]), label = f"ANN prediction, mean = {y_pred:.4f}", color = "blue")
#axes[0, 0].plot(np.array([t1graph[0],t1graph[-1]]), np.array([prob_ann_results.mean(), prob_ann_results.mean()]), label = f"Probabilistic ANN prediction, mean = {y_pred:.4f}", color = "green")
axes[0, 0].plot(np.array([t1graph[0],t1graph[-1]]), np.array([y_real, y_real]), label = f"Data Sample, mean = {y_real:.4f}", color = "red")
axes[0, 0].grid()
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Mean")
axes[0, 0].legend()


axes[1, 1].set_title("Obtained probability")
axes[1, 1].plot(vectors_per_layer[-1], pdfs_dynamical[0, -1, :], label = "DSP", color = "black")
#axes[1, 1].plot(vectors_per_layer[-1], prob_ann_pdf, label = "Probabilistic ANN", color = "green")
axes[1, 1].plot(np.array([y_pred, y_pred]), np.array([0, max_value_pdfs]), label = "ANN prediction", color = "blue")
axes[1, 1].plot(np.array([y_real, y_real]), np.array([0, max_value_pdfs]), label = "Data Sample", color = "red")
axes[1, 1].grid()
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("Probability")
axes[1, 1].legend()


#axes[1, 0].set_title("Probability of success over time")
#axes[1, 0].plot(t1graph, prob_real_succ[0, :], label = "Data Sample", color = "red")
#axes[1, 0].plot(t1graph, prob_ann_succ[0, :], label = "ANN prediction", color = "blue")
#axes[1, 0].grid()
#axes[1, 0].set_xlabel("Time")
#axes[1, 0].set_ylabel("Probability")
#axes[1, 0].legend()

axes[1, 0].set_title("KL Divergence over time")
axes[1, 0].plot(t1graph, kl_div_over_time, color = "black")
axes[1, 0].grid()
axes[1, 0].set_xlabel("Time")
axes[1, 0].set_ylabel("KL Div")
#axes[1, 0].legend()


axes[0, 1].set_title("L2 Distance of the DSP mean and prediction over time")
axes[0, 1].plot(t1graph, l2_dsp_data[0, :], label = "Data Sample", color = "red")
axes[0, 1].plot(t1graph, l2_dsp_ann[0, :], label = "ANN prediction", color = "blue")
#axes[0, 1].plot(t1graph, l2_dsp_prob_ann[0, :], label = "Prob ANN mean", color = "green")
axes[0, 1].grid()
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("MSE")
axes[0, 1].legend()


# Now we save the figure ! 

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout(pad=1, w_pad=0.3, h_pad=0.8)

# Now we save the figure

import os
file_path = os.path.join(run_dir, "SDE1.pdf")

fig.savefig(file_path, format = "pdf", bbox_inches='tight')

# %% [markdown]
#    ### Now we save the results!

# %%
print(f"Mean of our model predictions at time {t1graph[-1]}: {means_dynamical[0, -1]}")
print(f"ANN prediction {mlp_test_pred[run_id]}")

# %%
file_path = os.path.join(run_dir, "means_dynamical.npz")
print("File path: ,", file_path)
np.savez(file_path, means_dynamical = np.array(means_dynamical, dtype = object))

# %%
file_path = os.path.join(run_dir, "prob_real_succ.npz")
print("File path: ,", file_path)
np.savez(file_path, prob_real_succ = np.array(prob_real_succ, dtype = object))

# %%
file_path = os.path.join(run_dir, "prob_ann_succ.npz")
print("File path: ,", file_path)
np.savez(file_path, prob_ann_succ = np.array(prob_ann_succ, dtype = object))

# %%
file_path = os.path.join(run_dir, "kl_div_over_time.npz")
print("File path: ,", file_path)
np.savez(file_path, kl_div_over_time = np.array(kl_div_over_time, dtype = object))

# %%
print(f"Mean of our dynamical results at the last time: {np.mean(samples_dynamical[0, :, -1])}")
print(f"Variance of our dynamical results at the last time: {np.var(samples_dynamical[0, :, -1])}")


