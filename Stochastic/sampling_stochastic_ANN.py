# %%
import os

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
# Use 64 BITS !
tf.keras.backend.set_floatx("float64")
from tensorflow.keras import layers, models
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# %% [markdown]
#   ### We now read the setup file

# %%
setup_dict = {}
with open("setup.txt") as f:
    for line in f:
        line = line.split("#", 1)[0].strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            # Use eval only if you trust the file
            setup_dict[key.strip()] = eval(value.strip())

neurons_per_layer = setup_dict["neurons_per_layer"]
activations_per_layer = setup_dict["activations_per_layer"]
n_samples_ann = setup_dict["n_samples_ann"]
dataset = setup_dict.get("dataset", "diabetes")
scale_y = setup_dict.get("scale_y", True)
seed = setup_dict.get("seed", 42)
uci_data_dir = setup_dict.get("uci_data_dir", "UCI_Datasets")
sklearn_data_home = setup_dict.get("sklearn_data_home", None)

np.random.seed(seed)
tf.random.set_seed(seed)

print(f"Architecture: {neurons_per_layer}")
print(f"Activations: {activations_per_layer}")
print(f"Samples per test point: {n_samples_ann}")

# %% [markdown]
#   ### Load the data

# %%
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

# Now we split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.10,
    # To reproduce results !
    random_state=seed)

print("Number of training data points: ", len(y_train))
print("Number of testing data points: ", len(y_test))

# %%
X_train_81, X_val_9, y_train_81, y_val_9 = train_test_split(
    X_train, y_train,
    test_size=0.10,
    random_state=seed)
print("Points used to fit the scalers (as in training):", len(y_train_81))

# Inputs: always standardised in training
scaler = StandardScaler().fit(X_train_81)
X_train = scaler.transform(X_train).astype("float64")
X_test = scaler.transform(X_test).astype("float64")
X_scaled = scaler.transform(X).astype("float64")

# Outputs: standardised in training only if scale_y = True
if scale_y:
    scaler_y = StandardScaler().fit(y_train_81.reshape(-1, 1))
    y_mean, y_scale = scaler_y.mean_[0], scaler_y.scale_[0]
else:
    y_mean, y_scale = 0.0, 1.0
print(f"scale_y = {scale_y}: y_mean = {y_mean:.6f}, y_scale = {y_scale:.6f}")

#def to_original_units(y_net):
#    # network output -> original units of y (identity if scale_y = False)
#    return y_net * y_scale + y_mean

def to_original_units(y_net):
    # network output -> original units of y (unchanged if scale_y = False)
    if scale_y:
        return scaler_y.inverse_transform(np.asarray(y_net).reshape(-1, 1)).ravel()
    return y_net

# %% [markdown]
#   ### Load the trained parameters

# %%
weights = np.load("weights_per_layer.npz")
weights = [weights[str(i)] for i in range(len(weights.files))]                  # (out, in)
bias_means = np.load("biases_mean_per_layer.npz")
bias_means = [bias_means[str(i)] for i in range(len(bias_means.files))]
bias_vars = np.load("biases_var_per_layer.npz")
bias_stds = [np.sqrt(bias_vars[str(i)]) for i in range(len(bias_vars.files))]   # variance -> std

assert [w.shape[0] for w in weights] == list(neurons_per_layer), \
    "neurons_per_layer in setup.txt does not match the saved weights"
for i, s in enumerate(bias_stds):
    print(f"Layer {i}: bias std mean {s.mean():.4f}, min {s.min():.4f}, max {s.max():.4f}")

# %% [markdown]
# ### The stochastic layer: one bias mean and one bias std per neuron

# %%
class StochasticBiasDense(layers.Layer):
    # z = x W + mu_b + sigma_b * eps with fresh eps ~ N(0, 1) for every row, then the activation.
    # The noise is always on (this layer is only used for sampling).

    def __init__(self, kernel, bias_mean, bias_std, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel_value = tf.constant(kernel, dtype=tf.float64)      # (in, out)
        self.bias_mean = tf.constant(bias_mean, dtype=tf.float64)      # (out,)
        self.bias_std = tf.constant(bias_std, dtype=tf.float64)        # (out,)
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        z = tf.matmul(inputs, self.kernel_value) + self.bias_mean
        z = z + self.bias_std * tf.random.normal(tf.shape(z), dtype=tf.float64)
        return self.activation(z)

# %% [markdown]
# ### And now we build the stochastic model and the deterministic model (biases at their means)

# %%
def build_prob_model():
    model = models.Sequential(name="StochasticBiasRegression")
    model.add(tf.keras.Input(shape=(X_train.shape[1],), dtype=tf.float64))
    for i in range(len(neurons_per_layer)):
        model.add(StochasticBiasDense(weights[i].T, bias_means[i], bias_stds[i],
                                      activation=activations_per_layer[i]))
    return model


def build_det_model():
    model = models.Sequential(name="Regression")
    model.add(tf.keras.Input(shape=(X_train.shape[1],), dtype=tf.float64))
    for i in range(len(neurons_per_layer)):
        dense = layers.Dense(neurons_per_layer[i], activation=activations_per_layer[i])
        model.add(dense)
        dense.set_weights([np.asarray(weights[i].T, dtype=np.float64),
                           np.asarray(bias_means[i], dtype=np.float64)])
    return model


prob_model = build_prob_model()
det_model = build_det_model()

# %% [markdown]
#  ### Deterministic predictions (biases at their means), in original units

# %%
mlp_tot_pred = to_original_units(det_model.predict(X_scaled, verbose=0).flatten())
np.savez("mlp_tot_pred.npz", mlp_tot_pred=mlp_tot_pred)

mlp_test_pred = to_original_units(det_model.predict(X_test, verbose=0).flatten())
np.savez("mlp_test_pred.npz", mlp_test_pred=mlp_test_pred)

# %% [markdown]
# ### Now we do inference on the probabilistic model

# %%
prob_ann_results = []
for datum_idx in range(len(X_test)):
    datum_vector = np.repeat(X_test[datum_idx].reshape(1, -1), n_samples_ann, axis=0)
    results = prob_model.predict(datum_vector, verbose=0).flatten()
    prob_ann_results.append(to_original_units(results))
    
# %%
prob_ann_dict = {}
for datum_idx in range(len(X_test)):
    prob_ann_dict[str(datum_idx)] = prob_ann_results[datum_idx]
    
np.savez("prob_ann_results.npz", **prob_ann_dict)

# %%
samples = np.stack(prob_ann_results, axis=1)       # (n_samples_ann, n_test)
print(f"Mean of the samples vs deterministic prediction, largest difference: "
      f"{np.max(np.abs(samples.mean(axis=0) - mlp_test_pred)):.4f}")
print(f"Std of the samples over the test set: mean {samples.std(axis=0, ddof=1).mean():.4f}, "
      f"min {samples.std(axis=0, ddof=1).min():.4f}, max {samples.std(axis=0, ddof=1).max():.4f}")