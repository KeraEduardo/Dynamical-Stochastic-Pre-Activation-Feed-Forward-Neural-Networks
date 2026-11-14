# %%
import numpy as np
import matplotlib.pyplot as plts
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer

# %% [markdown]
# ## Here we generate the samples from a neural network

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

solve_using_ivp = setup_dict["solve_using_ivp"]

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


x_input = tf.convert_to_tensor(input_vector)
x_input = tf.expand_dims(x_input, axis = 0)

# %% [markdown]
# ### We now create the neural network model

# %%
class NoisyDense(Layer):
    def __init__(self, units, activation=None, noise_std=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.noise_std = noise_std

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=(input_shape[-1], self.units),
                                      initializer="glorot_uniform",
                                      trainable=False)
        self.bias = self.add_weight("bias",
                                    shape=(self.units,),
                                    initializer="zeros",
                                    trainable=False)

    def call(self, inputs, training=False):
        z = tf.matmul(inputs, self.kernel) + self.bias
        if training and self.noise_std > 0:
            noise = tf.random.normal(tf.shape(z), stddev=self.noise_std)
            z = z + noise
        return self.activation(z)



# %%
models_list = []
outputs_list = []

inputs = Input(shape= (input_vector.shape))
for layer_idx in range(0, len(neurons_per_layer)):
    #print("layer idx: ", layer_idx)
    if layer_idx == 0:
        x = NoisyDense(neurons_per_layer[layer_idx], 
                       noise_std = sigma_per_layer[0], 
                       activation = activations_per_layer[0], 
                       name = "NoisyDense_0")(inputs)
        model_layer = Model(inputs = inputs, outputs = x)
    else: 
        x = NoisyDense(neurons_per_layer[layer_idx], 
                       noise_std = sigma_per_layer[layer_idx], 
                       activation = activations_per_layer[layer_idx], 
                       name = f"NoisyDense_{layer_idx}")(x)
        layer_name = f"NoisyDense_{layer_idx - 1}"
        model_layer = Model(inputs = model_layer.get_layer(layer_name).output, outputs = x)
    models_list.append(model_layer)

# %% [markdown]
#   ### Now we set the weights

# %%
loaded = np.load("weights_per_layer.npz")

# Numeric keys
weights_per_layer = [loaded[f"{i}"] for i in range(len(loaded.files))]


# %%
loaded = np.load("biases_per_layer.npz")

# Numeric keys
biases_per_layer = [loaded[f"{i}"] for i in range(len(loaded.files))]

# %%
for layer_idx in range(0, len(neurons_per_layer) ):
    model_idx = models_list[layer_idx]

    dense_layers = [layer for layer in model_idx.layers if isinstance(layer, NoisyDense)]
    for i, layer in enumerate(dense_layers):
        layer.set_weights([np.transpose(weights_per_layer[layer_idx]), biases_per_layer[layer_idx]])



# %% [markdown]
#   ### Now we perform inference

# %% [markdown]
#   ### Now we obtain the outputs for the tensorflow model

# %%
cumprod = np.cumprod(samples_per_layer)
cumprod



# %%
sample_digital_results_per_layer = []

for layer_idx in range(len(neurons_per_layer)):
    mat = np.zeros((neurons_per_layer[layer_idx], cumprod[layer_idx]))
    sample_digital_results_per_layer.append( mat )

# And we select it as
#sample_digital_results_per_layer[layer][neuron, sample]



# %% [markdown]
#   ### We first add the first layer

# %%
outputs_list.append(input_vector)

for layer_idx in range(0, len(neurons_per_layer)):
    #print(f"============================= Layer ============================= {layer_idx}")
    model = models_list[layer_idx]

    if layer_idx == 0: 
        for sample_idx in range(cumprod[0]):

            # We obtain the prediction
            prediction = models_list[0](x_input, training = True).numpy()[0]
            
            # Now we save the result
            sample_digital_results_per_layer[layer_idx][:, sample_idx] = prediction
    else:
        counter = 0
        #for prev_sample_idx in range(samples_per_layer[layer_idx -1]):
        for prev_sample_idx in range(cumprod[layer_idx -1]):
            # We go through previous result
            input_sample = sample_digital_results_per_layer[layer_idx -1][:, prev_sample_idx]
            
            # We convert this input_sample into a tensor
            input_sample = tf.convert_to_tensor(input_sample)
            input_sample = tf.expand_dims(input_sample, axis = 0)
            
            
            for curr_sample_idx in range(samples_per_layer[layer_idx]):
                #print("Counter: ", counter)
                
                # Now we sample over this input sample!
                prediction = models_list[layer_idx](input_sample, training = True).numpy()[0]

                # Now we save the result
                sample_digital_results_per_layer[layer_idx][:, counter] = prediction
                # We increase the counter once
                counter = counter + 1



# %% [markdown]
# ### Now we compute the vectors per layer

# %%
epsilon = 1e-6

def create_vector(activation, n_x, eps = epsilon, min_z = -5, max_z = 5):
    if activation == "sigmoid":
        x = np.linspace(eps, 1-eps, n_x, dtype = np.float32)
    elif activation == "linear":
        x = np.linspace(min_z, max_z, n_x, dtype = np.float32)
    elif activation == "tanh":
        x = np.linspace(-1 + eps, 1-eps, n_x, dtype = np.float32)
    elif activation == "relu":
        x = np.linspace(min_z, max_z, n_x, dtype = np.float32)
    elif activation == "softplus":
        x = np.linspace(min_z, max_z, n_x, dtype = np.float32)
    return x

# %%
vectors_per_layer = []
for layer_idx in range(len(neurons_per_layer)):
    result = create_vector(activation = activations_per_layer[layer_idx], n_x = points_per_layer[layer_idx] )
    vectors_per_layer.append( result )

# %% [markdown]
#   ### Now we create a function that computes probability distributions

# %%
def compute_pdf(samples, vec):
    # Convert inputs to numpy arrays if they aren't already
    samples = np.asarray(samples)
    vec = np.asarray(vec)
    
    
    # Compute histogram (counts in each bin)
    counts, bin_edges = np.histogram(samples, bins=vec)
    
    # Normalize the counts to get probability density
    pdf_values = counts / (np.sum(counts) * np.diff(bin_edges))
    #pdf_values = counts / np.sum(counts)

    # We normalise the pdf values
    pdf_values = pdf_values / ( np.sum(pdf_values) )
    
    return pdf_values



# %% [markdown]
# ### Now we obtain the probability samples per neuron

# %%
digital_pdf_per_neuron = []



# %%
for layer_idx in range(len(neurons_per_layer)):
    vector = vectors_per_layer[layer_idx]
    vector = np.concatenate((vector, [1e8]), axis = 0) #Always works for sigmoid, tanh and softplus

    for neuron_idx in range(neurons_per_layer[layer_idx]):
        
        # We obtain the samples
        samples = sample_digital_results_per_layer[layer_idx][neuron_idx, :]

        pdf = compute_pdf(samples, vector)

        digital_pdf_per_neuron.append(pdf)



# %% [markdown]
# ### Plot just to check

# %%
import matplotlib.pyplot as plt
plt.plot(vectors_per_layer[0], digital_pdf_per_neuron[0])

# %%
digital_pdf_per_neuron[0][0]

# %% [markdown]
# ### Now we save the samples

# %%
32768*2*2

# %%
pdf_dict = {f"{i}": w for i, w in enumerate(digital_pdf_per_neuron)}
np.savez("pdf_per_neuron.npz", **pdf_dict)



