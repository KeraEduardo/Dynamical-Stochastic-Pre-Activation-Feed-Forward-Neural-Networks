# %%
import numpy as np
import matplotlib.pyplot as plts
import tensorflow as tf

# %% [markdown]
# ## All we do in this notebook is generate the weights

# %% [markdown]
# ### We first read the setup

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
n_t = setup_dict["n_t"]
min_time = setup_dict["min_time"]
max_time = setup_dict["max_time"]
#n_jump = setup_dict["n_jump"]
idx_data = setup_dict["idx_data"]

# %% [markdown]
# ### We load the input image

# %%
def flatten(x):
    x = tf.reshape(x, [-1])           # Flatten to 1D first
    return tf.reshape(x, [-1, 1])     # Then reshape to (x, 1)

def flatten_np(x):
    return np.reshape(x, (-1))

# %%
import tensorflow_datasets as tfds
x_test, y_test = tfds.load("mnist", split=['test'], batch_size=-1, as_supervised=True)[0]
x_dynamical = x_test / 255

# We select the input data
#idx_data = 1
input_vector = flatten_np( x_dynamical[idx_data].numpy() )

# %% [markdown]
# ### Now we generate the weights

# %%
weights_per_layer = []

# We add the first case
weight = 0.01 * np.random.randn(neurons_per_layer[0], input_vector.shape[0])
weights_per_layer.append(weight)

for layer_idx in range(len(neurons_per_layer)-1):
    dim1 = neurons_per_layer[layer_idx]
    dim2 = neurons_per_layer[layer_idx + 1]
    weight = 0.1*np.random.randn(dim2, dim1)
    #weight = 0 * np.random.randn(dim2, dim1)
    weights_per_layer.append(weight)


biases_per_layer = []
for layer_idx in range(len(neurons_per_layer)):
    dim = neurons_per_layer[layer_idx]
    bias = np.random.rand(dim)
    biases_per_layer.append(bias)

# %% [markdown]
# ### Now we save the results

# %%
weights_dict = {f"{i}": w for i, w in enumerate(weights_per_layer)}
biases_dict = {f"{i}": w for i, w in enumerate(biases_per_layer)}

# %%
np.savez("weights_per_layer.npz", **weights_dict)
np.savez("biases_per_layer.npz", **biases_dict)

# %% [markdown]
# np.save("weights_per_layer.npy", np.array(weights_per_layer, dtype=object))
# np.save("biases_per_layer.npy", biases_per_layer, dtype=object)
# 

# %% [markdown]
# np.savetxt("weights_per_layer.txt", weights_per_layer)
# np.savetxt("biases_per_layer.txt", biases_per_layer)

# %% [markdown]
# np.savez("variationalparameters.npz", 
#          weights = weights_per_layer, 
#          biases = biases_per_layer)

# %% [markdown]
# np.array(weights_per_layer, dtype=object)

# %%



