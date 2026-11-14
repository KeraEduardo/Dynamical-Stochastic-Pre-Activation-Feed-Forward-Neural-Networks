# %%
import numpy as np

#import latex
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

#plt.rcParams.update({
#    "text.usetex": False,
#    "mathtext.fontset": "cm",  # Computer Modern style
#    "font.family": "serif"
#})


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer
import tensorflow as tf 

# For distributed processes
from tensorflow.python.distribute import mirrored_strategy
import matplotlib.image as mpimg
import os

# %%
import os
print("Available CPUs: ",os.cpu_count())

# We save the run_dir and run_id
#run_dir = os.environ.get("RUN_DIR")
#run_id = int(os.environ.get("RUN_ID"))

#run_dir = "C:/Users/eduar/OneDrive/Desktop/Yani/Fake/Regression"
#run_id = 1

# %%
setup_dict = {}

#with open("/home/eduardo.martinezgarc/projects/Yani/Fake/Regression/Training/setup.txt") as f:
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
#  ### Now we import the data vector

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


print(f"Dataset keys: {data.keys()}")
print(f"Dataset features: {data.feature_names}")
print(f"Data description: {data.DESCR}")

# %%
#y_real = y_test[run_id]

#data = np.load("/home/eduardo.martinezgarc/projects/Yani/Fake/Regression/Training/mlp_test_pred.npz", allow_pickle=True)
data = np.load("mlp_test_pred.npz", allow_pickle=True)
mlp_test_pred = data["mlp_test_pred"]

#y_pred = mlp_test_pred[run_id]

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
# ### Now we load the results for the means

# %%
mlp_test_pred

# %%
y_test

# %%
data = np.load("means_dynamical_dataset.npz", allow_pickle=True)
means_dynamical = data["means_dynamical"]

# %%
len(means_dynamical)

# %%
time_idx = n_graph - 1
time = t1graph[time_idx]
time = str(time)[0:6]

#means_dynamical[sample_idx, neuron_idx, time_idx]
fig, ax = plt.subplots(figsize = (12,8))

#plt.figure(figsize = (12, 8))
ax.scatter(np.arange(len(means_dynamical)), y_test, label = f"Dataset", color = "black", s = 25)
ax.scatter(np.arange(len(means_dynamical)), mlp_test_pred, label = f"FFN", color = "black", marker = ",", alpha = 0.2, s = 60)
ax.scatter(np.arange(len(means_dynamical)), means_dynamical[:, 0, time_idx], label = "DSP FFN", color = "black", marker = "*", alpha = 0.9, s = 50)
ax.set_title(f"Solution outputs for a DSP FFN at time {time} [s]. and its corresponding FFN", fontsize = 24)
ax.legend(fontsize = 20)
ax.grid()
ax.set_xlabel("Patient Index", size = 20)
ax.set_ylabel("Target Value", size = 20)

# We control the number of divisions !
ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
ax.xaxis.set_major_locator(MaxNLocator(nbins=3))

# We change the size of the ticks !
ax.tick_params(axis='x', which='major', labelsize=20)
ax.tick_params(axis='y', which='major', labelsize=20)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout(pad=1, w_pad=0.3, h_pad=0.8)

# Now we save the figure
plt.savefig("DeterministicDSPFFNResults.pdf", format = "pdf", bbox_inches='tight')

# %% [markdown]
# ### Now we create the video

# %%
# Now we do a movie
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(means_dynamical))

def update(time_idx):
    ax.clear()

    time = t1graph[time_idx]
    time = str(time)[:6]

    ax.scatter(np.arange(len(means_dynamical)), y_test, label = f"Dataset", color = "black", s = 25)
    ax.scatter(np.arange(len(means_dynamical)), mlp_test_pred, label = f"FFN", color = "black", marker = ",", alpha = 0.2, s = 60)
    ax.scatter(np.arange(len(means_dynamical)), means_dynamical[:, 0, time_idx], label = "DSP FFN", color = "black", marker = "*", alpha = 0.9, s = 50)
    #ax.scatter(x, means_dynamical[:, 0, time_idx], label="DSP FFN prediction", color="black")

    ax.set_title(f"Solution outputs for a DSP FFN at time {time} [s]. and its corresponding FFN", fontsize = 24)

    # For the legend
    ax.legend(fontsize = 20)
    ax.grid(True)

    ax.set_xlabel("Patient Index", size = 20)
    ax.set_ylabel("Target Value", size = 20)

    # We control the number of divisions !
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    # We change the size of the ticks !
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20)

# Create animation
anim = FuncAnimation(fig, update, frames=n_graph,
    interval=100,  # milliseconds per frame
    repeat=True)

#anim.save("DeterministicResults.mp4", writer="ffmpeg", fps=10)
# Or save as GIF
anim.save("DeterministicResults.gif", writer="pillow", fps=10)

plt.show()

# %% [markdown]
# ### Now we load the L2 norms

# %%
l2_data = np.load("l2_dsp_ann.npz", allow_pickle=True)
l2_dsp_ann = l2_data["l2_dsp_ann"]

# %% [markdown]
# ### Now we do a linear regressin

# %%
from scipy import stats

l2_sum = np.sum(l2_dsp_ann[:, 0, :], axis = 0)
l2_sum = np.array(l2_sum, dtype=float)
l2_log = np.log(l2_sum)

regress = stats.linregress(t1graph[1:], l2_log[1:])
slope, intercept = regress.slope, regress.intercept

# %%
fig, ax = plt.subplots(figsize = (12,8))
ax.scatter(t1graph, l2_log, color = "black", alpha = 0.5)
ax.plot(t1graph[1:], np.array(t1graph[1:])*slope + intercept, color = "black", linewidth = 1)
ax.set_title(f"L2D between the DSP FFN's and its corresponding FFN's solution outputs", fontsize = 24)


ax.grid()
ax.set_xlabel("Time [s]", size = 20)
ax.set_ylabel(r"ln$($L2D$)$", size = 20)

# We control the number of divisions !
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

# We change the size of the ticks !
ax.tick_params(axis='x', which='major', labelsize=20)
ax.tick_params(axis='y', which='major', labelsize=20)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout(pad=1, w_pad=0.3, h_pad=0.8)

# Now we save the figure
plt.savefig("L2Deterministic.pdf", format = "pdf", bbox_inches='tight')

# %%



