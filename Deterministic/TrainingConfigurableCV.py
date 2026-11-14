# %% [markdown]
# ### In this notebook we train an FFN for regression
# %%
import numpy as np 
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns # Visualization

import tensorflow as tf
# Use 64 BITS !
tf.keras.backend.set_floatx('float64')
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %% [markdown]
#   ### We now read hte architecture layer

# %%
setup_dict = {}

with open("setup.txt") as f:
    for line in f:
        line = line.strip()
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

additional_regularisation = setup_dict.get("additional_regularisation", False)

if additional_regularisation:
    # L2 weight penalty on every Dense layer
    l2_reg       = setup_dict.get("l2_reg", 1e-3)
    # dropout after each hidden layer (0 = off)
    dropout_rate = setup_dict.get("dropout_rate", 0.0)
    # number of folds for cross validation
    n_splits     = setup_dict.get("n_splits", 5)
else:
    # Same behaviour as the original notebook
    l2_reg       = 0.0
    dropout_rate = 0.0
    n_splits     = 3

print("Additional regularisation:", additional_regularisation)
print(f"  l2_reg = {l2_reg}, dropout_rate = {dropout_rate}, n_splits = {n_splits}")

# Check the architecture works !
assert len(neurons_per_layer) == len(activations_per_layer), "Give exactly one activation per layer"
if neurons_per_layer[-1] != 1 or activations_per_layer[-1] != "linear":
    print("WARNING: for regression the last layer should be a single neuron with a 'linear' activation")

# %% [markdown]
### Now we choose the dataset

# %%
if dataset.lower() == "housing":
    data = fetch_california_housing()
elif dataset.lower() == "diabetes":
    data = load_diabetes()
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)

X = data.data
y = data.target

# We only select the ones less than 5 IF USING CALIFORNIA HOUSING DATASET
if dataset.lower() == "housing":
    idxs = (y <= 4.99)
    X = X[idxs]
    y = y[idxs]

# %%
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

# %% [markdown]
# ### Now we split the datasets

# %%
print(f"Total samples in the dataset: {len(X)}")
print("Number of testing data points: ", len(y_test))

# %% [markdown]
#   ### Now we scale the y axis

# %%
print("Features of the dataset: ", data.feature_names)

# %% [markdown]
# ### Now we set the architectures to try

# %%
# Candidate architectures for the cross validation
architectures = [
    # 1 hidden layer
    [8, 1],
    [16, 1],
    [32, 1],
    [64, 1],
    # 2 hidden layers
    [8, 8, 1],
    [16, 8, 1],
    [32, 16, 1],
    [64, 32, 1],
    [128, 128, 1]
    # Do not try the next architectures as our DSP FFN simulations are only three-layered !
    # 3 hidden layers
    #[16, 16, 8, 1],
    #[32, 16, 8, 1],
    #[64, 32, 16, 1]
    ]

# keep only the candidates that match the depth of activations_per_layer
n_layers = len(activations_per_layer)
skipped = [a for a in architectures if len(a) != n_layers]
architectures = [a for a in architectures if len(a) == n_layers]

print("Activations per layer:", activations_per_layer)
print(f"Architectures tested in the CV ({n_layers} layers):", architectures)
print("Skipped (different number of layers):", skipped)
if cross_validate:
    assert len(architectures) > 0, f"No candidate architecture has {n_layers} layers, add some to the list"

# with additional_regularisation the CV also searches the L2 strength.
# Cost: len(architectures) * len(l2_values) * n_splits fits.
if additional_regularisation:
    l2_values = [1e-4, 1e-3, 1e-2]
else:
    l2_values = [0.0]   # no L2, only the architectures are compared (as in the original)

# %% [markdown]
#   ## Now we define the model

# %%
def build_model(architecture=None, activations=None, input_dim=None,
                learning_rate=None, l2=None, dropout=None):
    architecture  = neurons_per_layer     if architecture  is None else architecture
    activations   = activations_per_layer if activations   is None else activations
    input_dim     = X_train.shape[1]      if input_dim     is None else input_dim
    learning_rate = lr                    if learning_rate is None else learning_rate
    l2            = l2_reg                if l2            is None else l2
    dropout       = dropout_rate          if dropout       is None else dropout

    model = models.Sequential(name = "Regression")
    model.add(layers.Input(shape=(input_dim,)))

    for i in range(len(architecture)):
        is_output_layer = (i == len(architecture) - 1)
        model.add(layers.Dense(
            architecture[i],
            activation=activations[i],
            kernel_initializer="he_normal" if activations[i] == "relu" else "glorot_normal",
            # l2 penalty on the weights (only if additional_regularisation = True,
            # otherwise l2 = 0 and no regulariser is added)
            kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None))

        # dropout after hidden layers (only if additional_regularisation = True). Dropout has no weights and is switched off
        # at prediction time, so the manual forward pass below is not affected.
        if dropout > 0 and not is_output_layer:
            model.add(layers.Dropout(dropout))

    optimizer = Adam(learning_rate = learning_rate)
    # report the pure MSE ('mse' / 'val_mse') instead of 'loss' / 'val_loss'.
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model


def make_callbacks():
    early_stop = EarlyStopping(monitor='val_mse', mode='min', patience=patience,
                               restore_best_weights=True)
    callbacks = [early_stop]

    # halve the learning rate when the validation error stalls
    # (only if additional_regularisation = True)
    if additional_regularisation:
        reduce_lr = ReduceLROnPlateau(monitor='val_mse', mode='min', factor=0.5,
                                      patience=max(10, patience // 4), min_lr=1e-6)
        callbacks.append(reduce_lr)
    return callbacks

# %% [markdown]
# ### And one for cross validation

# %%
def build_cv_model(architecture, input_dim, lr, l2=None):
    return build_model(architecture, activations_per_layer, input_dim,
                       learning_rate=lr, l2=l2)

# %% [markdown]
#   ### And now we train

# %%
if cross_validate:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_results = []

    # cross-validation over architectures AND L2 strength
    for architecture in architectures:
        for l2 in l2_values:
            print("\n" + "="*70)
            print("Architecture:", architecture, " L2:", l2)
            print("="*70)

            fold_mse = []
            fold_mae = []
            fold_epochs = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_dev), start=1):
                # Split development data
                X_train_fold = X_dev[train_idx]
                X_val_fold   = X_dev[val_idx]

                y_train_fold = y_dev[train_idx]
                y_val_fold   = y_dev[val_idx]

                # Scale X. Fit scaler ONLY on the training portion of this fold.
                scaler_X_fold = StandardScaler()
                X_train_fold = scaler_X_fold.fit_transform(X_train_fold)
                X_val_fold   = scaler_X_fold.transform(X_val_fold)

                # Scale y
                if scale_y:
                    scaler_y_fold = StandardScaler()
                    y_train_fold = scaler_y_fold.fit_transform(y_train_fold.reshape(-1, 1)).ravel()
                    y_val_fold = scaler_y_fold.transform(y_val_fold.reshape(-1, 1)).ravel()

                X_train_fold = X_train_fold.astype("float64")
                X_val_fold   = X_val_fold.astype("float64")
                y_train_fold = y_train_fold.astype("float64")
                y_val_fold   = y_val_fold.astype("float64")

                # Build fresh model
                tf.keras.backend.clear_session()
                model_cv = build_cv_model(architecture, input_dim=X_dev.shape[1], lr=lr, l2=l2)

                history = model_cv.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size,
                    validation_data = (X_val_fold, y_val_fold),
                    callbacks = make_callbacks(),
                    verbose = 0,
                    shuffle = True)

                # read the pure MSE / MAE (without the L2 penalty) by name
                scores = model_cv.evaluate(X_val_fold, y_val_fold, verbose = 0, return_dict = True)
                val_mse, val_mae = scores["mse"], scores["mae"]

                fold_mse.append(val_mse)
                fold_mae.append(val_mae)
                # store the epoch of the BEST model
                fold_epochs.append(int(np.argmin(history.history["val_mse"])) + 1)
                print(f"  Fold {fold}/{n_splits}: MSE = {val_mse:.4f}, MAE = {val_mae:.4f}, best epoch = {fold_epochs[-1]}")

            # Average across fold results
            mean_mse = np.mean(fold_mse)
            std_mse = np.std(fold_mse)
            mean_mae = np.mean(fold_mae)
            std_mae = np.std(fold_mae)
            mean_epochs = np.mean(fold_epochs)

            cv_results.append({
                "architecture": architecture,
                "l2": l2,
                "mean_mse": mean_mse,
                "std_mse": std_mse,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
                "mean_epochs": mean_epochs,
                "fold_mse": fold_mse,
                "fold_mae": fold_mae,
                "fold_epochs": fold_epochs})

            print(f"Mean CV MSE = {mean_mse:.4f} +/- {std_mse:.4f}")
            print(f"Mean CV MAE = {mean_mae:.4f} +/- {std_mae:.4f}")
            print(f"Mean best epoch = {mean_epochs:.1f}")
else:
    tf.keras.backend.clear_session()
    model = build_model()
    print("Trainable parameters:", model.count_params())
    print("Training samples    :", len(y_train))
    #if model.count_params() > 10 * len(y_train):
    #    print("NOTE: many more parameters than samples -> strong regularisation is needed")

    history = model.fit(X_train, y_train,
        epochs=epochs, batch_size=batch_size, validation_data = (X_val, y_val),
        callbacks=make_callbacks(), verbose=0, shuffle = True)
    print("Stopped after", len(history.history["loss"]), "epochs; best epoch:",
          int(np.argmin(history.history["val_mse"])) + 1)

    # Now we evaluate the model
    scores = model.evaluate(X_test, y_test, verbose = 0, return_dict = True)
    test_mse_score, test_mae_score = scores["mse"], scores["mae"]

# %% [markdown]
# ### Now we select the best model

# %%
# Select architecture + L2 using mean CV MSE
if cross_validate:
    best_result = min(cv_results, key = lambda result: result["mean_mse"])
    best_architecture = best_result["architecture"]
    best_l2 = best_result["l2"]

    print("\n")
    print("="*70)
    print("CROSS-VALIDATION RESULTS")
    print("="*70)

    for result in cv_results:
        print(
            f"Architecture: {str(result['architecture']):14s} "
            f"L2: {result['l2']:<8g} "
            f"Mean MSE: {result['mean_mse']:.4f} "
            f"Std: {result['std_mse']:.4f}")

    print("="*70)

    print("\nSelected architecture:", best_architecture, " L2:", best_l2)
    print(f"Mean CV MSE: {best_result['mean_mse']:.4f}")
    print(f"CV MSE standard deviation: {best_result['std_mse']:.4f}")
    print(f"Mean best epoch: {best_result['mean_epochs']:.1f}")
    # NOTE: if several rows are within one std of the best, the SMALLEST of them is usually
    # the safer choice; differences smaller than the std are just noise.

# %% [markdown]
# ### Now we save the results as a DataFrame

# %%
if cross_validate:
    cv_table = pd.DataFrame([
        {"Architecture": r["architecture"],
            "L2": r["l2"],
            "Mean CV MSE": r["mean_mse"],
            "Std CV MSE": r["std_mse"],
            "Mean CV MAE": r["mean_mae"],
            "Std CV MAE": r["std_mae"],
            "Mean Epochs": r["mean_epochs"]}
        for r in cv_results])

    cv_table = cv_table.sort_values("Mean CV MSE").reset_index(drop=True)
    print(cv_table)
    cv_table.head()

# %% [markdown]
# ### And now that we have our result, we can train the selected model !

# %%
if cross_validate:
    final_epochs = int(np.round(best_result["mean_epochs"]))

    neurons_per_layer = best_architecture
    l2_reg = best_l2

    print("Final architecture:", neurons_per_layer)
    print("Final activations :", activations_per_layer)
    print("Final L2          :", l2_reg)
    print("Final training epochs:", final_epochs)

# %%
if cross_validate:
    # Fit final scalers ONLY on development data
    # We have X_train 81%, X_test 10% and X_validation 9%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state = seed)
    
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

# %% [markdown]
# ### And we train the final model

# %%
if cross_validate:
    # Train final model
    tf.keras.backend.clear_session()

    model = build_model(neurons_per_layer, activations_per_layer, X_train.shape[1],
                        learning_rate=lr, l2=l2_reg)
    print("Final model parameters:", model.count_params())
    # allow `patience` extra epochs past the CV estimate
    history = model.fit(X_train, y_train, epochs = final_epochs + patience,
                        validation_data = (X_val, y_val), batch_size=batch_size,
                        callbacks = make_callbacks(), verbose=0, shuffle=True)

    # Now we evaluate the model 
    scores = model.evaluate(X_test, y_test, verbose = 0, return_dict = True)
    test_mse_score, test_mae_score = scores["mse"], scores["mae"]

# %%
# NOTE: these are in SCALED units if scale_y = True (see the cells below for original units)
print("Test MAE score:", test_mae_score)
print("Test MSE score:", test_mse_score)

# %% [markdown]
#   ### Now we visualize

# %%
plt.figure(figsize = (8, 4))
plt.plot(history.history['mse'], label='Train MSE')
plt.plot(history.history["val_mse"], label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.yscale('log')
plt.legend()
plt.title(f'Training vs Validation, test MAE = {test_mae_score:.4f}, test MSE = {test_mse_score:.4f}')
plt.grid()
plt.savefig("TrainingVsValidation.pdf")
plt.show()

# %%
y_pred = model.predict(X_test, verbose = 0).flatten()
y_test_original = y_test.copy()

# Now we inverse transform
if scale_y:
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

print(f"r2 score = {r2_score(y_test_original, y_pred)}")
print("Prediction mean:", np.mean(y_pred))
print("Prediction std :", np.std(y_pred))
print("Prediction min :", np.min(y_pred))
print("Prediction max :", np.max(y_pred))

plt.figure(figsize = (6, 4))
plt.scatter(y_test_original, y_pred, alpha=0.3)
plt.xlabel("True values")  
plt.ylabel("Predicted values")
plt.title(f"Predicted vs True Values in Test, r2 score = {r2_score(y_test_original, y_pred):.4f}")

# Perfect prediction line
plt.plot(y_test_original, y_test_original, 'r')

plt.grid()
plt.savefig("DifferenceInPrices.pdf")
plt.show()

# %% [markdown]
### compare with a linear-regression baseline

# %%
lin = LinearRegression().fit(X_train, y_train)
y_pred_lin = lin.predict(X_test)
if scale_y:
    y_pred_lin = scaler_y.inverse_transform(y_pred_lin.reshape(-1, 1)).flatten()

for name, pred in [("Neural network", y_pred), ("Linear regression", y_pred_lin)]:
    mse = np.mean((y_test_original - pred)**2)
    mae = np.mean(np.abs(y_test_original - pred))
    print(f"{name:18s}: R2 = {r2_score(y_test_original, pred):.4f}, "
          f"MSE = {mse:.2f}, MAE = {mae:.2f}  (original units)")
print("Test samples:", len(y_test_original))

# %%
residuals = y_test_original.flatten() - y_pred.flatten()

plt.figure(figsize = (6,4))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red')
plt.xlabel("Predicted value")
plt.ylabel("Residuals")
plt.title("Residual Plot on the test set")
plt.grid()
plt.savefig("ResidualplotTest.pdf")
plt.show()

# %%
print(f"Range of the target variable : {np.min(y)}-{np.max(y)}")

print("Minimum residual index on test set: ", np.argmin(np.abs(residuals)) )
print("Minimum residual value on test set: ", residuals[np.argmin(np.abs(residuals))] )

print("Maximum residual index on test set: ", np.argmax(np.abs(residuals)) )
print("Maximum residual value on test set: ", residuals[np.argmax(np.abs(residuals))] )

# %% [markdown]
#   ### Now we save the variational parameters

# %%
dense_layers = [layer for layer in model.layers if isinstance(layer, layers.Dense)]

# First the weights
weights_dict = {}
for i, layer in enumerate(dense_layers):
    w, _ = layer.get_weights()  # Get only weights
    weights_dict[str(i)] = np.transpose(w)

np.savez("weights_per_layer.npz", **weights_dict)

# %%
# Now the biases
biases_dict = {}
for i, layer in enumerate(dense_layers):
    _, b = layer.get_weights()  # Get only biases
    biases_dict[str(i)] = np.transpose(b)

np.savez("biases_per_layer.npz", **biases_dict)

# %% [markdown]
#   ### Now we print the information of the datum

# %%
print("Features: ", data.feature_names)
#print("Values: ", X[idx_data])

# %% [markdown]
#  ### Now we save the predictions of the neural network

# %%
X_scaled = scaler_X.transform(X)
mlp_tot_pred = model.predict(X_scaled, verbose = 0).flatten()
if scale_y:
    mlp_tot_pred = scaler_y.inverse_transform(mlp_tot_pred.reshape(-1, 1)).flatten()
np.savez("mlp_tot_pred.npz", mlp_tot_pred = mlp_tot_pred)

# %%
mlp_test_pred = model.predict(X_test, verbose = 0).flatten()
if scale_y:
    mlp_test_pred = scaler_y.inverse_transform(mlp_test_pred.reshape(-1, 1)).flatten()
np.savez("mlp_test_pred.npz", mlp_test_pred = mlp_test_pred)

# %% [markdown]
#  ### Now we obtain the results manually in order to compare with the TensorFlow results

# %%
# We load the obtained weights
weights = np.load("weights_per_layer.npz", allow_pickle = True)
weights = [weights[f"{i}"] for i in range(len(weights.files))]
#print("Weights: ", weights)

biases = np.load("biases_per_layer.npz", allow_pickle = True)
biases = [biases[f"{i}"] for i in range(len(biases.files))]

# %%
def g(x, activation = "sigmoid", beta = 1):
 
    if activation == "sigmoid":
        return 1/(1+np.exp(-x))
    
    elif activation == "linear":
        return x
    
    elif activation == "tanh":
        return np.tanh(x)

    elif activation == "softplus":
        #return 1/beta * np.log(1 + np.exp(beta*x))
        return x/beta + 1/beta * np.log(np.exp(-x) + np.exp(x*(beta-1)))

    elif activation == "relu":
        return np.maximum(0, x)

    # fail loudly instead of silently returning None
    raise ValueError(f"Unknown activation: {activation}")

# %%
# Now we obtain the results
X_test_numpy = X_test

manual_outputs = []
biases_per_layer = []
for layer_idx in range(len(neurons_per_layer)):
    vec = np.tile(biases[layer_idx], (len(X_test_numpy), 1))
    biases_per_layer.append(vec)

# %% [markdown]
# ### Now we obtain the results

# %%
print("Neurons per layer    :", neurons_per_layer)
print("Activations per layer:", activations_per_layer)

# %%
o1 = g(np.matmul(X_test, weights[0].T) + biases_per_layer[0], activation = activations_per_layer[0]).T
manual_outputs.append(o1)

for layer_idx in range(1, len(neurons_per_layer)):
    x_prev = manual_outputs[layer_idx - 1]
    o_output = g(np.matmul(weights[layer_idx], x_prev) + biases_per_layer[layer_idx].T, activation= activations_per_layer[layer_idx])
    #o_output = g(np.matmul(weights[layer_idx].T, x_prev) + biases_per_layer[layer_idx], activation = activations_per_layer[layer_idx])

    # Now we save the results
    manual_outputs.append(o_output)

# %%
# Now we scale the final layer
if scale_y:
    manual_outputs[-1] = scaler_y.inverse_transform(manual_outputs[-1])

# %% [markdown]
# ### Now we write a dict that we will be saving

# %%
manual_outputs_dict = {}
for layer_idx in range(len(neurons_per_layer)):
    manual_outputs_dict[str(layer_idx)] = manual_outputs[layer_idx]
    
np.savez("manual_outputs.npz", **manual_outputs_dict)

# %% [markdown]
# ### Now we obtain the differences in L2

# %%
from scipy.spatial.distance import minkowski

# %%
l2_distances = []

for datum_idx in range(len(X_test_numpy)):
    mlp_test_pred_idx = [mlp_test_pred[datum_idx]]
    manual_vector = manual_outputs[-1][:, datum_idx]
    # Now we do the neurons
    dist = minkowski(mlp_test_pred_idx, manual_vector, p = 2)
    #dist = np.abs(mlp_test_pred_idx - manual_vector)
    l2_distances.append(dist)

np.savez("l2_distances.npz", l2_distances = l2_distances)

# %%
plt.figure(figsize = (6, 3))
plt.scatter(np.arange(len(X_test_numpy)), l2_distances, color = "black", s = 3)
plt.xlabel("Datum Index")
plt.ylabel("L2 Distance")
plt.title("Distances Between TensorFlow and Manual predictions")
plt.grid()
plt.savefig("L2_distances.pdf")
plt.show()

# %% [markdown]
# ### Now we plot everything

# %%
# None = all samples (-1 would drop the last one)
n_samples_to_show = None

truth  = np.ravel(y_test_original)[:n_samples_to_show]
tf_pred = np.ravel(mlp_test_pred)[:n_samples_to_show]
manual  = manual_outputs[-1].flatten()[:n_samples_to_show]

# ascending; use order[::-1] for descending
plot_ascending = True
if plot_ascending:
    order = np.argsort(truth)
    x = np.arange(len(order))
else:
    order = np.arange(len(X_test))

plt.figure(figsize=(8, 4))
plt.scatter(x, tf_pred[order], label="TensorFlow", s=20, alpha=0.3, color="gray")
plt.scatter(x, manual[order],  label="Manual",     s=3,  alpha=0.8, color="black")
plt.scatter(x, truth[order],   label="Truth",      s=20, alpha=0.3, color="red", marker="*")
plt.title("Manual vs TensorFlow prediction Comparison")
plt.legend()
plt.xlabel("Sample (sorted by true value)")
plt.ylabel("Prediction")
plt.grid()
plt.savefig("ManualvsTensorFlow.pdf")
plt.show()

# %%



