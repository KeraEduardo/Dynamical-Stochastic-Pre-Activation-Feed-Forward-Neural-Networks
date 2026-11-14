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
# ### Now we import the data vector

# %%
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split

data = load_diabetes()

X = data.data
y = data.target

# Now we split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.10,
    # To reproduce results !
    random_state=42)

# %%
#y_real = y_test[run_id]

data = np.load("/home/eduardo.martinezgarc/projects/Yani/Fake/Regression/Training/mlp_test_pred.npz", allow_pickle=True)
#data = np.load("mlp_test_pred.npz", allow_pickle=True)
mlp_test_pred = data["mlp_test_pred"]

#y_pred = mlp_test_pred[run_id]

# %% [markdown]
# ### For multiprocessing

# %% [markdown]
#    ### Time vector

# %%
t_vec = np.linspace(min_time, max_time, n_t)
dt = np.abs(t_vec[1] - t_vec[0])
print("dt: ", dt)

# %% [markdown]
#   ### We create the indices for the graph

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
# # Now we READ the dataset of all probabilistic ANNs

# %%
data = np.load("/home/eduardo.martinezgarc/projects/Yani/Fake/Regression/Training/prob_ann_results.npz", allow_pickle=True)
#data = np.load("prob_ann_results.npz", allow_pickle=True)
prob_ann_results = []
prob_ann_means = []


for datum_idx in range(len(X_test)):
    data_idx = data[str(datum_idx)]
    prob_ann_results.append( data_idx )
    prob_ann_means.append( np.mean(data_idx) )

# %% [markdown]
#  # Now we READ the dataset for each of the samples

# %%
total_samples_dynamical = []
main_path = "/home/eduardo.martinezgarc/projects/Yani/Fake/Regression/"
for run_idx in range(len(X_test)):
    file_path = os.path.join(main_path, str(run_idx) + "/samples_dynamical.npz")
    data_idx = np.load(file_path, allow_pickle = True)
    data_idx = data_idx["samples_dynamical"]
    total_samples_dynamical.append(data_idx)

print(f"Length of included samples dynamical: {len(total_samples_dynamical)}")

# Now we save the results
file_path = os.path.join(run_dir, "total_samples_dynamical.npz")
print("File path: ,", file_path)
np.savez(file_path, total_samples_dynamical = np.array(total_samples_dynamical, dtype = object))

# %%
# Now we obtain the minimum value of all of them
absolute_min = 10000
absolute_max = 0
for run_idx in range(len(total_samples_dynamical)):
    absolute_min = min(absolute_min, total_samples_dynamical[run_idx].min(), y_test.min(), mlp_test_pred.min())
    absolute_max = max(absolute_max, total_samples_dynamical[run_idx].max(), y_test.max(), mlp_test_pred.max())

print(f"Absolute min obtained: {absolute_min}")
print(f"Absolute max obtained: {absolute_max}")

# %% [markdown]
#    ### Now we create the vectors for the neural network

# %%
epsilon = 1e-16
delta_x = 10


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
#  ### Now we define the functions we willbe using to plot

# %%
from scipy.special import rel_entr
def safe_kl_divergence(P, Q):
    return rel_entr(x = P, y = Q)

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
    
    print(f"x shape: {x.shape}")
    print(f"p shape: {p.shape}")

    #p = p / p.sum()

    mask = (x >= a) & (x <= b)

    #return np.sum(p[mask])
    return np.trapz(p[mask], x[mask])

# %% [markdown]
#  ### Now we create the vectors per layer

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
#   ### Now we create a vector that stores the means of the samples and the L2 distances

# %%
means_dynamical = np.zeros((len(X_test), neurons_per_layer[-1], n_graph))
l2_dsp_ann = np.zeros((len(X_test), neurons_per_layer[-1], n_graph))
l2_dsp_data = np.zeros((len(X_test), neurons_per_layer[-1], n_graph))

# %% [markdown]
#  ### Now we compute the pdfs per each neuron

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

# %% [markdown]
# ### Dynamical PDF

# %%
pdfs_dynamical = np.zeros((len(X_test), neurons_per_layer[-1], n_graph, points_per_layer[-1]))

for sample_idx in range(len(total_samples_dynamical)):
    for neuron_idx in range(neurons_per_layer[-1]):
        for time_idx in range(len(t_graph)):

                samples = total_samples_dynamical[sample_idx]
                samples = samples[neuron_idx, :, time_idx]
                
                # Now we obtain the sampless
                #pdf = compute_pdf(samples = samples, vec = layer_vector)
                pdf = compute_pdf(samples = samples, activation = activations_per_layer[-1], number_points = points_per_layer[-1])

                # We save the PDF
                pdfs_dynamical[sample_idx, neuron_idx, time_idx, :] = pdf        

# %% [markdown]
#   ## Now we fill the data that we are gonna be graphing

# %% [markdown]
#   ### The mean over time

# %%
for sample_idx in range(len(total_samples_dynamical)):
    for neuron_idx in range(neurons_per_layer[-1]):
        for time_idx in range(n_graph):

            mean = np.mean(total_samples_dynamical[sample_idx][neuron_idx, :, time_idx])

            # Now we do fill the vector
            means_dynamical[sample_idx, neuron_idx, time_idx] = mean

# %% [markdown]
#  ### And we also load the results we obtained manually

# %%
data = np.load("/home/eduardo.martinezgarc/projects/Yani/Fake/Regression/Training/manual_outputs.npz", allow_pickle=True)
#data = np.load("manual_outputs.npz", allow_pickle=True)
manual_test_pred = data[str(len(neurons_per_layer)-1)]

# %%
for sample_idx in range(len(total_samples_dynamical)):
    for neuron_idx in range(neurons_per_layer[-1]):
        for time_idx in range(n_graph):

            mse_ann = l2_norm(x = means_dynamical[sample_idx, neuron_idx, time_idx], y = mlp_test_pred[sample_idx])
            mse_data = l2_norm(x = means_dynamical[sample_idx, neuron_idx, time_idx], y = y_test[sample_idx])

            # Now we fill the data
            l2_dsp_ann[sample_idx, neuron_idx, time_idx] = mse_ann 
            l2_dsp_data[sample_idx, neuron_idx, time_idx] = mse_data

# %%
# Now we save the results for the L2 distances
file_path = os.path.join(run_dir, "l2_dsp_ann.npz")
print("File path: ,", file_path)
np.savez(file_path, l2_dsp_ann = np.array(l2_dsp_ann, dtype = object))

file_path = os.path.join(run_dir, "l2_dsp_data.npz")
print("File path: ,", file_path)
np.savez(file_path, l2_dsp_data = np.array(l2_dsp_data, dtype = object))

# %% [markdown]
#   ### Now we calculate the probability of success

# %%
prob_real_succ = np.zeros((len(total_samples_dynamical), neurons_per_layer[-1], n_graph))
prob_ann_succ = np.zeros((len(total_samples_dynamical), neurons_per_layer[-1], n_graph))

# Now we fill it up !
for sample_idx in range(len(total_samples_dynamical)):
    for neuron_idx in range(neurons_per_layer[-1]):
        for time_idx in range(n_graph):
            
            #print(f"Length of vector: {len(vectors_per_layer[-1])}")
            #print(f"Length of pdf: {len(pdfs_dynamical[sample_idx, neuron_idx, time_idx, :])}")
            # Now we obtain the probability of our result falling within either value!
            prob_ann_succ_neur = interval_probability(x = vectors_per_layer[-1], p = pdfs_dynamical[sample_idx, neuron_idx, time_idx, :],
                                                    a = mlp_test_pred[sample_idx] - dx_last_layer, b = mlp_test_pred[sample_idx] + dx_last_layer)

            prob_real_succ_neur = interval_probability(x = vectors_per_layer[-1], p = pdfs_dynamical[sample_idx, neuron_idx, time_idx, :],
                                                            a = y_test[sample_idx] - dx_last_layer, b = y_test[sample_idx] + dx_last_layer)

            # Now we save the results
            prob_real_succ[sample_idx, neuron_idx, time_idx] = prob_real_succ_neur
            prob_ann_succ[sample_idx, neuron_idx, time_idx] = prob_ann_succ_neur

# %% [markdown]
#  ### Now we read the results from the probabilistic ANN

# %%
#data_prob_ann = np.load("prob_ann_results.npz", allow_pickle=True)
data_prob_ann = np.load("/home/eduardo.martinezgarc/projects/Yani/Fake/Regression/Training/prob_ann_results.npz", allow_pickle=True)

means_prob_ann = []
for datum_idx in range(len(X_test)):
    means_prob_ann.append( data_prob_ann[str(datum_idx)].mean() )

# %% [markdown]
#  ### Now we save the results !

# %%
file_path = os.path.join(run_dir, "means_dynamical_dataset.npz")
print("File path: ,", file_path)
np.savez(file_path, means_dynamical = np.array(means_dynamical, dtype = object))

# %%
file_path = os.path.join(run_dir, "prob_real_succ_dataset.npz")
print("File path: ,", file_path)
np.savez(file_path, prob_real_succ = np.array(prob_real_succ, dtype = object))

# %%
file_path = os.path.join(run_dir, "prob_ann_succ_dataset.npz")
print("File path: ,", file_path)
np.savez(file_path, prob_ann_succ = np.array(prob_ann_succ, dtype = object))

# %%
file_path = os.path.join(run_dir, "means_prob_ann.npz")
print("File path: ,", file_path)
np.savez(file_path, means_prob_ann = np.array(means_prob_ann, dtype = object))

# %% [markdown]
#  ### Now we graph

# %%
# Time goes from 0 to n_graph - 1
#time_idx = 63
time_idx = n_graph - 1
time = t1graph[time_idx]
time = str(time)[0:6]

means_dynamical[sample_idx, neuron_idx, time_idx]
plt.figure(figsize = (12, 8))
plt.scatter(np.arange(len(total_samples_dynamical)), y_test, label = f"Real Values", color = "red")
plt.scatter(np.arange(len(total_samples_dynamical)), mlp_test_pred, label = f"FFN prediction", color = "blue")
plt.scatter(np.arange(len(total_samples_dynamical)), means_prob_ann, label = f"Probabilistic FFN Means", color = "green")
plt.scatter(np.arange(len(total_samples_dynamical)), means_dynamical[:, 0, time_idx], label = "DSP FFN prediction", color = "black")
plt.title(f"Comparison between FFNa nd DSP FFNs at time {time}s. with {n_samples_dynamical} dynamical samples", fontsize = 16)
plt.legend()
plt.grid()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout(pad=1, w_pad=0.3, h_pad=0.8)

# Now we save the figure
plt.savefig("results.pdf", format = "pdf", bbox_inches='tight')

# %%
# Now we do a movie
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(total_samples_dynamical))

def update(time_idx):
    ax.clear()

    time = t1graph[time_idx]
    time = str(time)[:6]

    ax.scatter(x, y_test, label="Real Values", color="red")
    ax.scatter(x, mlp_test_pred, label="FFN prediction", color="blue")
    ax.scatter(x, means_dynamical[:, 0, time_idx], label="DSP FFN prediction", color="black")
    ax.scatter(x, means_prob_ann, label = f"Probabilistic FFN prediction", color = "green")

    ax.set_title(f"Comparison between FFN and DSP FFNs at time {time}s with {n_samples_dynamical} dynamical samples and {sigma_per_layer} noise variances", fontsize=16)

    ax.legend()
    ax.grid(True)

# Create animation
anim = FuncAnimation(fig, update, frames=n_graph,
    interval=100,  # milliseconds per frame
    repeat=True)

anim.save("results.mp4", writer="ffmpeg", fps=10)
# Or save as GIF
# anim.save("results.gif", writer="pillow", fps=10)

plt.show()


