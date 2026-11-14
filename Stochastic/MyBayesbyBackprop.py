# %%
import numpy as np
#import latex
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import matplotlib.image as mpimg


#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif"
#})

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",  # Computer Modern style
    "font.family": "serif"
})

# SKLEARN
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression   # NEW: baseline
from sklearn.metrics import r2_score

# TENSORFLOW
import tensorflow as tf 
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# For distributed processes
from tensorflow.python.distribute import mirrored_strategy

import math
import os

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# %% [markdown]
# ### Now we read the data

# %%
setup_dict = {}

with open("setup.txt") as f:
    for line in f:
        line = line.strip()
        # CHANGED: skip comment lines so setup.txt can be documented
        if line.startswith("#"):
            continue
        if line and "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Use eval only if you trust the file
            setup_dict[key] = eval(value)
            
neurons_per_layer = setup_dict["neurons_per_layer"]
activations_per_layer = setup_dict["activations_per_layer"]
neuron_idxs = setup_dict.get("neuron_idxs", [-1])
#idx_data = setup_dict["idx_data"]

# For training
batch_size = setup_dict["batch_size"]
lr = setup_dict["lr"]
epochs = setup_dict["epochs"]
patience = setup_dict["patience"]
scale_y = setup_dict["scale_y"]
cross_validate = setup_dict["cross_validate"]
dataset = setup_dict["dataset"]

n_samples_ann = setup_dict["n_samples_ann"]

# %% [markdown]
# ### Now we load the dataset

# %% [markdown]
# ### We also add the option of doing the case of the dataset in the paper

# %%
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split

UCI_FOLDERS = {"concrete": "concrete", "energy": "energy", "kin8nm": "kin8nm",
               "naval": "naval-propulsion-plant", "power": "power-plant",
               "protein": "protein-tertiary-structure", "wine": "wine-quality-red", "yacht": "yacht"}

name = dataset.lower()
if name == "housing":
    data = fetch_california_housing()
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
elif name == "paper":
    # distinct x values
    n_inputs = 15
    # noisy observations per x       
    n_repeats = 5
    X_grid = np.linspace(0, 0.5, n_inputs)

    # each x appears n_repeats times
    X = np.repeat(X_grid, n_repeats).reshape(-1, 1)
    eps = np.random.normal(loc=0.0, scale=0.02, size=X.shape[0])
    # The same eps is used inside the sines and added at the end, as in the paper's Eq. (Section 5.2)
    y = X[:, 0] + 0.3*np.sin(2*np.pi*(X[:, 0] + eps)) + 0.3*np.sin(4*np.pi*(X[:, 0] + eps)) + eps

    n_training_samples = X.shape[0]
    plt.plot(X, y)
else:
    raise ValueError(f"Unknown dataset: {dataset}")

# %% [markdown]
# ### Now we normalise

# %% [markdown]
# ### And we do the special case of test data

# %%
if dataset == "paper":

    n_test = 35
    # points in [-0.2, 0]
    n_left = 10
    # points in [0.5, 1.2]
    n_right = n_test - n_left

    X_left = np.linspace(-0.2, 0.0, n_left)
    X_right = np.linspace(0.5, 1.2, n_right)
    X_test_flat = np.concatenate([X_left, X_right])
    X_test = X_test_flat.reshape(-1, 1)

    eps_test = np.random.normal(loc=0.0, scale=0.02, size=n_test)
    # Same eps inside the sines and added at the end, as in the training data
    y_test = (X_test_flat
            + 0.3*np.sin(2*np.pi*(X_test_flat + eps_test))
            + 0.3*np.sin(4*np.pi*(X_test_flat + eps_test))
            + eps_test)

    y_noiseless = (X_test_flat
            + 0.3*np.sin(2*np.pi*(X_test_flat))
            + 0.3*np.sin(4*np.pi*(X_test_flat)))

    plt.scatter(X_test_flat, y_test, s=12, label="Test points", color = "red")
    plt.scatter(X, y, s=12, label="Training points", color = "black", marker = "+")
    plt.legend(); plt.grid()

    X_train = X.copy()
    y_train = y.copy()
else:
    # We have X_train 81%, X_test 10% and X_validation 9%    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = seed)
    y_test_original = y_test.astype("float64")

# %% [markdown]
# ### Now we scale

# %%
from sklearn.preprocessing import StandardScaler

# Now we normalise the data (scalers are fitted on the training set only)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test  = scaler_X.transform(X_test)

if scale_y:
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()


X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)

print("Training samples  :", len(y_train))
print("Test samples      :", len(y_test))

# %% [markdown]
# ### And the inverse value

# %%
def to_original_units(y_net):
    # network output -> original units of y (unchanged if scale_y = False)
    if scale_y:
        return scaler_y.inverse_transform(np.asarray(y_net).reshape(-1, 1)).ravel()
    else: 
        return y_net

# %% [markdown]
# # Now we choose the dataset
# if dataset.lower() == "housing":
#     data = fetch_california_housing()
# elif dataset.lower() == "diabetes":
#     data = load_diabetes()
# seed = 42
# np.random.seed(seed)
# tf.random.set_seed(seed)
# tf.keras.utils.set_random_seed(seed)
# 
# X = data.data
# y = data.target
# 
# # We only select the ones less than 5 IF USING CALIFORNIA HOUSING DATASET
# if dataset.lower() == "housing":
#     idxs = (y <= 4.99)
#     X = X[idxs]
#     y = y[idxs]

# %%
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=seed, reshuffle_each_iteration=True)
            .batch(batch_size))
#test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# %% [markdown]
# ### Now we define the constants we will be using

# %%
# Globals, as in the PyTorch snippet (placeholder values)
pi = 0.5
# STDS of the mixture components
sigma1 = math.exp(0.0)
sigma2 = math.exp(-6.0)
# Number of samples used in the monte carlo step !
samples = 2**9
#samples = 120
num_batches = int(train_ds.cardinality().numpy())
#num_batches = len(X_train) // batch_size

# %% [markdown]
# ### Now we create all the functions we will be using 

# %%
# Logarithm of a 
def normal_log_pdf(x, mu, sigma):
    #x = float(x)
    #mu = float(mu)
    #sigma = float(sigma)
    
    return -0.5 * math.log(2.0 * math.pi) - tf.math.log(sigma) - 0.5 * tf.square((x - mu) / sigma)

# %% [markdown]
# ### Now we define the class of Gaussian

# %%
class Gaussian(tf.Module):
    #  We initialise
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho

    # Now we define the sigma = ln(1+exp(rho))
    @property
    def sigma(self):
        return tf.math.log1p(tf.exp(self.rho))

    # Now we return the reparametrisation trick
    # x = mu + sigma * epsilon
    def sample(self):
        epsilon = tf.random.normal(tf.shape(self.mu))
        return self.mu + self.sigma * epsilon

    # Now we define the log prob
    # P(X) = p(x1)*p(x2)*p(x3)...
    # ln(P(X)) = log(P(x1)) + log(P(x2)) + ...
    def log_prob(self, input):
        return tf.reduce_sum(normal_log_pdf(input, self.mu, self.sigma))

# %% [markdown]
# ### Now we define the initialisation of the weights

# %%
# This defines the prior pdf for the variational parameters
class ScaleMixtureGaussian(tf.Module):
    # We initialise
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    # We have pi*N(x|0, s1) + (1-pi)*N(x|0, s2)
    #p(W) = \prod_{j}(pi N(x_j, 0, s1) + (1-pi)*N(x_j, 0, s2))
    #log(p(W)) = sum_{j}(log(pi N(x_j, 0, s1) + (1-pi)*N(x_j, 0, s2)))
    def log_prob(self, input):
        # log(pi*N(x|0,s1) + (1-pi)*N(x|0,s2)), via logsumexp to avoid underflow
        log_p1 = tf.math.log(self.pi) + normal_log_pdf(input, 0.0, self.sigma1)
        log_p2 = tf.math.log(1.0 - self.pi) + normal_log_pdf(input, 0.0, self.sigma2)

        # Now we use reduce reduce_logsumexp, which computes log(a+b) written as logsumexp(loga, logb)
        return tf.reduce_sum(tf.reduce_logsumexp(tf.stack([log_p1, log_p2], axis=0), axis=0))

# %% [markdown]
# ### Now the function for the bayesian weights

# %%
class BayesianLinear(tf.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # TF kernels are [in, out]; PyTorch's are [out, in]
        
        ## These are the values for the variational posteriors.
        # The following values are TRAINABLE variational values
        # WEIGHTS first
        self.weight_mu = tf.Variable(tf.random.uniform([in_features, out_features], -0.2, 0.2), name="weight_mu")
        self.weight_rho = tf.Variable(tf.random.uniform([in_features, out_features], -5.0, -4.0), name="weight_rho")
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # BIASES then
        self.bias_mu = tf.Variable(tf.random.uniform([out_features], -0.2, 0.2), name="bias_mu")
        self.bias_rho = tf.Variable(tf.random.uniform([out_features], -5.0, -4.0), name="bias_rho")
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # Now we define the priors for the weights and baises
        self.weight_prior = ScaleMixtureGaussian(pi, sigma1, sigma2)
        self.bias_prior = ScaleMixtureGaussian(pi, sigma1, sigma2)

    def forward(self, input, sample=False, training=False):
        """Returns (output, log_prior, log_variational_posterior); log terms are 0 unless training."""
        if (training or sample):
            # We obtain a weight and bias sample
            weight = self.weight.sample()
            bias = self.bias.sample()
        # If not, we then obtain the deterministic value
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        # If training we obtain the logarithmic priors
        if training:
            log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            log_post = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            log_prior = 0.0
            log_post = 0.0
        return tf.matmul(input, weight) + bias, log_prior, log_post

# %% [markdown]
# ### And the version where only the bias has stochasticity

# %% [markdown]
# For the bias-only version: delete weight_mu, weight_rho, weight, and weight_prior from BayesianLinear. Set weight = self.weight_mu in forward, and drop the weight terms from log_prior and log_post.

# %%
class BayesianLinearOnlyBias(tf.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Deterministic weights: trainable point estimate, no posterior, no prior, no KL term
        self.weight = tf.Variable(tf.random.uniform([in_features, out_features], -0.2, 0.2), name="weight")
        
        ## These are the values for the variational posteriors.
        # The following values are TRAINABLE variational value
        # BIASES
        self.bias_mu = tf.Variable(tf.random.uniform([out_features], -0.2, 0.2), name="bias_mu")
        self.bias_rho = tf.Variable(tf.random.uniform([out_features], -5.0, -4.0), name="bias_rho")
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # Now we define the priors for the biases ONLY
        self.bias_prior = ScaleMixtureGaussian(pi, sigma1, sigma2)

    def forward(self, input, sample=False, training=False):
        """Returns (output, log_prior, log_variational_posterior); log terms are 0 unless training."""
        # Deterministic weight
        weight = self.weight
        if (training or sample):
            # We obtain a weight and bias sample
            bias = self.bias.sample()
        # If not, we then obtain the deterministic value
        else:
            bias = self.bias.mu
        # If training we obtain the logarithmic priors
        if training:
            log_prior = self.bias_prior.log_prob(bias)
            log_post =  self.bias.log_prob(bias)
        else:
            log_prior = 0.0
            log_post = 0.0
        return tf.matmul(input, weight) + bias, log_prior, log_post

# %% [markdown]
# ### Now we define the neural network!

# %%
activations_tf_dict = {
    "relu": tf.nn.relu,
    "tanh": tf.tanh,
    "sigmoid": tf.sigmoid,
    "none": lambda z: z,
    "linear": lambda z: z,
    "softplus": tf.nn.softplus,}

# %%
class BayesianNetwork(tf.Module):
    def __init__(self, input_dim, neurons_per_layer, activations_per_layer,
                 layer_cls=BayesianLinearOnlyBias, obs_sigma=1.0):
        # layer_cls is layer Class, we can set either stochastic weights+biases or only biases
        super().__init__()

        # We initialise
        self.input_dim = input_dim
        self.output_dim = neurons_per_layer[-1]
        # Observation noise std for the Gaussianl ikelihood
        self.obs_sigma = obs_sigma 
        self.activations = [activations_tf_dict[a] for a in activations_per_layer]

        # layer_cls lets swapping in BayesianLinear (weights + biases) or BayesianLinearOnlyBias
        self.layers = []
        in_dim = input_dim
        for n in neurons_per_layer:
            self.layers.append(layer_cls(in_dim, n))
            in_dim = n

    def forward(self, x, sample=False, training=False):
        h = tf.reshape(x, [-1, self.input_dim])
        log_prior = 0.0
        log_post = 0.0
        for layer, act in zip(self.layers, self.activations):
            h, lp, lq = layer.forward(h, sample, training)
            h = act(h)
            log_prior += lp
            log_post += lq
        return h, log_prior, log_post

    def sample_elbo(self, input, target, samples = samples, num_batches= num_batches):
        target = tf.reshape(target, [-1, self.output_dim])
        log_priors, log_posts, log_liks = [], [], []
        for _ in range(samples):
            out, lp, lq = self.forward(input, sample=True, training=True)
            # Gaussian likelihood: log p(y|x,w) = sum_i log N(y_i; f_w(x_i), obs_sigma^2)
            log_liks.append(tf.reduce_sum(normal_log_pdf(target, out, self.obs_sigma)))
            log_priors.append(lp)
            log_posts.append(lq)
        log_prior = tf.reduce_mean(tf.stack(log_priors))
        log_variational_posterior = tf.reduce_mean(tf.stack(log_posts))
        
        # E_q[log p(D|w)]
        expected_log_lik = tf.reduce_mean(tf.stack(log_liks)) 
        # Minibatch free energy: (KL estimate)/M - E_q[log p(D|w)]
        return (log_variational_posterior - log_prior) / num_batches - expected_log_lik

    #def predict(self, x, n_samples=TEST_SAMPLES):
    #    outs = tf.stack([self.forward(x, sample=True)[0] for _ in range(n_samples)])
    #    return tf.reduce_mean(outs, axis=0), tf.math.reduce_std(outs, axis=0)

# %% [markdown]
# ### Now we define the neural network

# %%
BNN = BayesianNetwork(input_dim = X.shape[1], neurons_per_layer = neurons_per_layer, activations_per_layer = activations_per_layer)
optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

# %% [markdown]
# ### Now we define the training function

# %%
@tf.function
def train_step(data, target):
    with tf.GradientTape() as tape:
        loss = BNN.sample_elbo(data, target)
    grads = tape.gradient(loss, BNN.trainable_variables)
    optimizer.apply_gradients(zip(grads, BNN.trainable_variables))
    return loss

# Now we define the function that trains for all datapoints
def train(train_ds):
    for data, target in train_ds:
        train_step(data, target)

# %% [markdown]
# ### Now we TRAIN!!!

# %%
w_before = BNN.layers[0].weight.numpy().copy()

for epoch in range(epochs):
    train(train_ds)

w_after = BNN.layers[0].weight.numpy()
print("max change in layer-0 weights:", np.abs(w_after - w_before).max())
print("min change in layer-0 weights:", np.abs(w_after - w_before).min())

# %%
#np.set_printoptions(precision=4, suppress=True, linewidth=120)

def describe_layer(i, layer):
    print(f"--- layer {i} ---")
    # Now we check if the weight is Gaussian itself
    if isinstance(layer.weight, Gaussian):
        # Full BayesianLinear: weights are also stochastic
        print("weight mean shape:", layer.weight.mu.shape)
        print("weight mean (first row):", layer.weight.mu.numpy()[0])
        print("weight var  (first row):", tf.square(layer.weight.sigma).numpy()[0])
    else:
        # BayesianLinearOnlyBias: weights are deterministic
        print("weight (deterministic) shape:", layer.weight.shape)
    print("bias mean:", layer.bias.mu.numpy())
    print("bias var :", tf.square(layer.bias.sigma).numpy())

for i, layer in enumerate(BNN.layers):
    describe_layer(i, layer)

# %% [markdown]
# ### Now we save the values

# %%
weights_dict, bias_mean_dict, bias_var_dict = {}, {}, {}
for i, layer in enumerate(BNN.layers):
    weights_dict[str(i)] = np.transpose(layer.weight.numpy())
    bias_mean_dict[str(i)] = layer.bias_mu.numpy()
    bias_var_dict[str(i)] = layer.bias.sigma.numpy() ** 2

# %% [markdown]
# ### Now we save them

# %%
#np.savez("weights_per_layer.npz", **weights_dict)
#np.savez("biases_mean_per_layer.npz", **bias_mean_dict)
#np.savez("biases_var_per_layer.npz", **bias_var_dict)

# %%
n_layers = len(BNN.layers)
fig, axes = plt.subplots(2, n_layers, figsize=(5 * n_layers, 8), squeeze=False)
for i, layer in enumerate(BNN.layers):
    mu = bias_mean_dict[str(i)]
    sd = np.sqrt(bias_var_dict[str(i)])
    ax = axes[0, i]
    ax.scatter(mu, sd, s=10, alpha=0.6)
    #ax.axhline(prior_sigma_bias, color="r", ls="--", lw=0.8, label="prior std")
    ax.set_xlabel("Mean"); ax.set_ylabel("stdev")
    ax.set_title(f"Layer {i} ({activations_per_layer[i]}), {neurons_per_layer[i]} biases"); ax.grid(); 
    #ax.legend()

    ax = axes[1, i]
    ax.hist(weights_dict[str(i)].ravel(), bins=50, color="gray")
    ax.set_xlabel("weight value"); ax.set_ylabel("count"); ax.set_title(f"Layer {i} weights"); ax.grid()
plt.tight_layout()
plt.savefig("BiasPosteriors.pdf")
plt.show()

# %% [markdown]
# ### Now we perform inference on the neural network

# %%
n_samples_ann = 2**10
prob_ann_results = []
for datum_idx in range(len(X_test)):
    results = [BNN.forward(X_test[datum_idx], sample=True)[0].numpy()[0,0] for _ in range(n_samples_ann)]    
    prob_ann_results.append(to_original_units(results))

# %% [markdown]
# ### Now we save the mean predictions and stds

# %%
pred_means = []
pred_vars = []

for datum_idx in range(len(X_test)):
    mean_idx = np.mean( prob_ann_results[datum_idx] )
    var_idx = np.var( prob_ann_results[datum_idx], ddof = 1)

    # Now we save
    pred_means.append(mean_idx)
    pred_vars.append(var_idx)

pred_means, pred_vars = np.array(pred_means), np.array(pred_vars)

# %% [markdown]
# ### Now we just visualize the predictions and the means

# %%
# Means and variance
plt.scatter(scaler_X.inverse_transform(np.asarray(X_test).reshape(-1, 1)).ravel(), pred_means)
plt.scatter(scaler_X.inverse_transform(np.asarray(X_test).reshape(-1, 1)).ravel(), pred_vars)
plt.yscale("log")

# %% [markdown]
# ### And if we are doing the results of the paper, then we also do inference on the training dataset

# %%
if dataset == "paper":
    n_samples_training = 2**10
    #x_paper = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], dtype = np.float32)
    #x_paper_transformed = scaler_X.transform(x_paper.reshape(-1, 1))

    #x_paper_transformed = x_paper_transformed.astype(np.float32)

    prob_ann_training_results = []

    for datum_idx in range(len(X_train)):
        results = [BNN.forward(X_train[datum_idx], sample=True)[0].numpy()[0,0] for _ in range(n_samples_training)]
        prob_ann_training_results.append(to_original_units(results))


    pred_means_training = []
    pred_vars_training = []

    for datum_idx in range(len(X_train)):
        mean_idx = np.mean( prob_ann_training_results[datum_idx] )
        var_idx = np.var( prob_ann_training_results[datum_idx], ddof = 1 )

        # Now we save
        pred_means_training.append(mean_idx)
        pred_vars_training.append(var_idx)

    pred_means_training, pred_vars_training = np.array(pred_means_training), np.array(pred_vars_training)

# %% [markdown]
# ### Now we visualize

# %%
if dataset != "paper":
    order = np.argsort(y_test_original)
    #order = np.arange(0, len(X_test))
    idx = np.arange(len(order))

    plt.figure(figsize=(10, 4))
    plt.fill_between(idx, (pred_means - 2 * np.sqrt(pred_vars))[order], (pred_means + 2 * np.sqrt(pred_vars))[order],
                    color="C1", alpha=0.4, label="pm 2*stdev")
    plt.plot(idx, pred_means[order], "k", lw=1, label="predictive mean")
    plt.scatter(idx, y_test_original[order], color="red", marker="*", s=30, label="truth")
    plt.xlabel("Test points (sorted by predicted mean)"); plt.ylabel("y")
    plt.title("Predictions with model uncertainty (stochastic biases)")
    plt.legend(); plt.grid()
    plt.savefig("TestPredictionsWithModelUncertainty2.pdf")
    plt.show()

# %%
if dataset == "paper":
    plt.scatter(scaler_X.inverse_transform(np.asarray(X_train).reshape(-1, 1)).ravel(), 
                scaler_y.inverse_transform(np.asarray(y_train).reshape(-1, 1)).ravel(), color = "black", label = "Training data", alpha = 0.3)

    # Now we do the case of the TRAIN data
    plt.scatter(scaler_X.inverse_transform(np.asarray(X_train).reshape(-1, 1)).ravel(), pred_means_training, color = "red", s = 3, label = "Neural Network Solution")
    plt.fill_between(scaler_X.inverse_transform(np.asarray(X_train).reshape(-1, 1)).ravel(), 
                     (pred_means_training + 2 * np.sqrt(pred_vars_training)), (pred_means_training - 2 * np.sqrt(pred_vars_training)),
                color="orange", alpha=0.4, label="mean pm 2*stdev")

    # And now on the TEST data
    # First we do the first half
    plt.scatter(scaler_X.inverse_transform(np.asarray(X_test[0:n_left]).reshape(-1, 1)).ravel(), pred_means[0:n_left], color = "red", s = 3)
    plt.fill_between(scaler_X.inverse_transform(np.asarray(X_test[0:n_left]).reshape(-1, 1)).ravel(), 
                    (pred_means[0:n_left] + 2 * np.sqrt(pred_vars[0:n_left])), (pred_means[0:n_left] - 2 * np.sqrt(pred_vars[0:n_left])),
                color="orange", alpha=0.4)

    # Now the SECOND half
    plt.scatter(scaler_X.inverse_transform(np.asarray(X_test[n_left:]).reshape(-1, 1)).ravel(), pred_means[n_left:], color = "red", s = 3)
    plt.fill_between(scaler_X.inverse_transform(np.asarray(X_test[n_left:]).reshape(-1, 1)).ravel(), 
                    (pred_means[n_left:] + 2 * np.sqrt(pred_vars[n_left:])), (pred_means[n_left:] - 2 * np.sqrt(pred_vars[n_left:])),
               color="orange", alpha=0.4)

    plt.plot(scaler_X.inverse_transform(np.asarray(X_test[0:n_left]).reshape(-1, 1)).ravel(), y_noiseless[0:n_left], color = "gray", label = "Data generating function")
    plt.plot(scaler_X.inverse_transform(np.asarray(X_test[n_left:]).reshape(-1, 1)).ravel(), y_noiseless[n_left:], color = "gray")
    
    plt.legend()
    plt.grid()
    #plt.savefig("MyPaperFigureImitation.pdf")
    plt.show()

# %% [markdown]
# ### Now we save the dataset if we used the synthetic dataset from the paper !

# %% [markdown]
# ### The X dataset

# %%
if dataset == "paper":
    X_train_dict = {}
    for datum_idx in range(len(X_train)):
        X_train_dict[str(datum_idx)] = X_train[datum_idx]
        
    np.savez("X_train.npz", **X_train_dict)

    X_test_dict = {}
    for datum_idx in range(len(X_test)):
        X_test_dict[str(datum_idx)] = X_test[datum_idx]
        
    np.savez("X_test.npz", **X_train_dict)

# %% [markdown]
# ### The Y dataset

# %%
if dataset == "paper":
    y_train_dict = {}
    for datum_idx in range(len(y_train)):
        y_train_dict[str(datum_idx)] = y_train[datum_idx]

    np.savez("y_train.npz", **X_train_dict)

    y_test_dict = {}
    for datum_idx in range(len(y_test)):
        y_test_dict[str(datum_idx)] = y_test[datum_idx]
        
    np.savez("y_test.npz", **X_train_dict)

# %% [markdown]
# ### And now we save the result

# %%
#prob_ann_dict = {}
#for datum_idx in range(len(X_test)):
#    prob_ann_dict[str(datum_idx)] = prob_ann_results[datum_idx]
    
#np.savez("prob_ann_results.npz", **prob_ann_dict)


