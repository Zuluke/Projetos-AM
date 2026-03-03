# %% [markdown]
# **Table of contents**<a id='toc0_'></a>    
# - [Imports](#toc1_)    
# - [Read Datasets](#toc2_)    
# - [Experiments](#toc3_)    
#   - [m = 1.3](#toc3_1_)    
#     - [Dataset 0](#toc3_1_1_)    
#     - [Dataset 1](#toc3_1_2_)    
#     - [Dataset 2](#toc3_1_3_)    
#   - [m = 1.4](#toc3_2_)    
#     - [Dataset 0](#toc3_2_1_)    
#     - [Dataset 1](#toc3_2_2_)    
#     - [Dataset 2](#toc3_2_3_)
#   - [m = 1.5](#toc3_3_)    
#     - [Dataset 0](#toc3_3_1_)    
#     - [Dataset 1](#toc3_3_2_)    
#     - [Dataset 2](#toc3_3_3_)    
#   - [m = 1.6](#toc3_4_)    
#     - [Dataset 0](#toc3_4_1_)    
#     - [Dataset 1](#toc3_4_2_)    
#     - [Dataset 2](#toc3_4_3_)
#   - [m = 1.7](#toc3_5_)    
#     - [Dataset 0](#toc3_5_1_)    
#     - [Dataset 1](#toc3_5_2_)    
#     - [Dataset 2](#toc3_5_3_)    
#   - [m = 1.8](#toc3_6_)    
#     - [Dataset 0](#toc3_6_1_)    
#     - [Dataset 1](#toc3_6_2_)    
#     - [Dataset 2](#toc3_6_3_)
#   - [m = 1.9](#toc3_7_)    
#     - [Dataset 0](#toc3_7_1_)    
#     - [Dataset 1](#toc3_7_2_)    
#     - [Dataset 2](#toc3_7_3_)
# 
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# %% [markdown]
# # <a id='toc1_'></a>[Imports](#toc0_)

# %%
from kfcm_k_w1 import KFCM_K_W1
import numpy as np
from sklearn import datasets
import pandas as pd
import sklearn
import seaborn as sns

# %% [markdown]
# ----

# %% [markdown]
# # <a id='toc2_'></a>[Read Datasets](#toc0_)

# %%
paths = ["../Dataset/mfeat-fac", "../Dataset/mfeat-fou", "../Dataset/mfeat-zer"]

dfs = [pd.read_csv(path, delim_whitespace=True, header=None) for path in paths]

# %%
Xs = []
ys = []
for df in dfs:
    # Label creation
    df["digit"] = [i for i in range(10) for __ in range(200)]
    Xs.append(df.drop("digit", axis=1).values)
    ys.append(df["digit"].values)

# %% [markdown]
# # <a id='toc3_'></a>[Experiments](#toc0_)

# %% [markdown]
# ## <a id='toc3_1_'></a>[m = 1.3](#toc0_)

# %% [markdown]
# ### <a id='toc3_1_1_'></a>[Dataset 0](#toc0_)

# %%
n_clusters = 10
m = 1.3
max_iter = 100
tol = 1e-6
random_state = 42

n_runs = 50

model_arrays = []
metrics_3 = []
best_loss = np.inf
best_index = None
for i in range(n_runs):
    print("="*50)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    kfcm.fit(Xs[0], ys[0])
    metrics_3.append(kfcm.evaluate())
    model_arrays.append({
        "U": kfcm._U,
        "G": kfcm._G,
        "s": kfcm._s,
    })
    if metrics_3[i]["loss"] < best_loss:
        best_loss = metrics_3[i]["loss"]
        best_index = i

# %%
np.savetxt("metrics_3_s.csv", model_arrays[best_index]["s"], delimiter=",")
np.savetxt("metrics_3_G.csv", model_arrays[best_index]["G"], delimiter=",")

# %%
Xs[0].shape

# %%
best_index, metrics_3[best_index]

# %%
# Metrics for m = 1.3
acc_vec3 = [metrics_3[i]["accuracy"] for i in range(n_runs)]
rand_vec3 = [metrics_3[i]["rand"] for i in range(n_runs)]
mpc_vec3 = [metrics_3[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec3):.4f}" + u" \u00b1 " + f"{np.std(acc_vec3):.4f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec3):.4f}" + u" \u00b1 " + f"{np.std(rand_vec3):.4f}")
print(f"Mean MPC: {np.mean(mpc_vec3):.4f}" + u" \u00b1 " + f"{np.std(mpc_vec3):.4f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
best_u3 = np.array(model_arrays[best_index]["U"])
u3_crisp = np.zeros_like(best_u3)
for k in range(best_u3.shape[0]):
    max_index = np.argmax(best_u3[k])
    u3_crisp[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u3_crisp.shape[1]
clusters3 = np.zeros((num_groups, num_groups))
rows = u3_crisp.shape[0] // num_groups
for h in range(num_groups):
    clusters3[h] = np.sum(u3_crisp[h*rows:(h+1)*rows], axis=0)
clusters3 = pd.DataFrame(clusters3.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters3, annot=True, fmt="d")

# %%
clusters3

# %% [markdown]
# ### <a id='toc3_1_2_'></a>[Dataset 1](#toc0_)

# %%
n_clusters = 10
m = 1.3
max_iter = 100
tol = 1e-6
random_state = 42

n_runs = 50

model_arrays_1 = []
metrics_3_1 = []
best_loss = np.inf
best_index_1 = None
for i in range(n_runs):
    print("="*50)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_1 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 1
    kfcm_1.fit(Xs[1], ys[1])
    metrics_3_1.append(kfcm_1.evaluate())
    model_arrays_1.append({
        "U": kfcm_1._U,
        "G": kfcm_1._G,
        "s": kfcm_1._s,
    })
    if metrics_3_1[i]["loss"] < best_loss:
        best_loss = metrics_3_1[i]["loss"]
        best_index_1 = i

# %%
metrics_3_1

# %%
best_index_1, metrics_3_1[best_index_1]

# %%
metrics_3_1[i]

# %%


# %%
# Metrics for m = 1.3
acc_vec3_1 = [metrics_3_1[i]["accuracy"] for i in range(n_runs)]
rand_vec3_1 = [metrics_3_1[i]["rand"] for i in range(n_runs)]
mpc_vec3_1 = [metrics_3_1[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec3_1):.4f}" + u" \u00b1 " + f"{np.std(acc_vec3_1):.4f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec3_1):.4f}" + u" \u00b1 " + f"{np.std(rand_vec3_1):.4f}")
print(f"Mean MPC: {np.mean(mpc_vec3_1):.4f}" + u" \u00b1 " + f"{np.std(mpc_vec3_1):.4f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
best_u3_1 = np.array(model_arrays_1[best_index_1]["U"])
u3_crisp_1 = np.zeros_like(best_u3_1)
for k in range(best_u3_1.shape[0]):
    max_index = np.argmax(best_u3_1[k])
    u3_crisp_1[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u3_crisp_1.shape[1]
clusters3_1 = np.zeros((num_groups, num_groups))
rows = u3_crisp_1.shape[0] // num_groups
for h in range(num_groups):
    clusters3_1[h] = np.sum(u3_crisp_1[h*rows:(h+1)*rows], axis=0)
clusters3_1 = pd.DataFrame(clusters3_1.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters3_1, annot=True, fmt="d")

# %%
clusters3_1

# %%
np.savetxt("metrics_3_1_s.csv", model_arrays_1[best_index_1]["s"], delimiter=",")
np.savetxt("metrics_3_1_G.csv", model_arrays_1[best_index_1]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_1_3_'></a>[Dataset 2](#toc0_)

# %%
n_clusters = 10
m = 1.3
max_iter = 100
tol = 1e-6
random_state = 42

n_runs = 50

model_arrays_2 = []
metrics_3_2 = []
best_loss = np.inf
best_index_2 = None
for i in range(n_runs):
    print("="*50)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_2 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 2
    kfcm_2.fit(Xs[2], ys[2])
    metrics_3_2.append(kfcm_2.evaluate())
    model_arrays_2.append({
        "U": kfcm_2._U,
        "G": kfcm_2._G,
        "s": kfcm_2._s,
    })
    if metrics_3_2[i]["loss"] < best_loss:
        best_loss = metrics_3_2[i]["loss"]
        best_index_2 = i

# %%
best_index_2, metrics_3_2[best_index_2]

# %%
# Metrics for m = 1.3
acc_vec3_2 = [metrics_3_2[i]["accuracy"] for i in range(n_runs)]
rand_vec3_2 = [metrics_3_2[i]["rand"] for i in range(n_runs)]
mpc_vec3_2 = [metrics_3_2[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec3_2):.4f}" + u" \u00b1 " + f"{np.std(acc_vec3_2):.4f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec3_2):.4f}" + u" \u00b1 " + f"{np.std(rand_vec3_2):.4f}")
print(f"Mean MPC: {np.mean(mpc_vec3_2):.4f}" + u" \u00b1 " + f"{np.std(mpc_vec3_2):.4f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
best_u3_2 = np.array(model_arrays_2[best_index_2]["U"])
u3_crisp_2 = np.zeros_like(best_u3_2)
for k in range(best_u3_2.shape[0]):
    max_index = np.argmax(best_u3_2[k])
    u3_crisp_2[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u3_crisp_2.shape[1]
clusters3_2 = np.zeros((num_groups, num_groups))
rows = u3_crisp_2.shape[0] // num_groups
for h in range(num_groups):
    clusters3_2[h] = np.sum(u3_crisp_2[h*rows:(h+1)*rows], axis=0)
clusters3_2 = pd.DataFrame(clusters3_2.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters3_2, annot=True, fmt="d")

# %%
clusters3_2

# %%
np.savetxt("metrics_3_2_s.csv", model_arrays_2[best_index_2]["s"], delimiter=",")
np.savetxt("metrics_3_2_G.csv", model_arrays_2[best_index_2]["G"], delimiter=",")

# %% [markdown]
# ## <a id='toc3_2_'></a>[m = 1.4](#toc0_)

# %% [markdown]
# ### <a id='toc3_2_1_'></a>[Dataset 0](#toc0_)

# %%
n_clusters = 10
m = 1.4
max_iter = 100
tol = 1e-6
n_runs = 50

model_arrays = []
metrics_4 = []
best_loss = np.inf
best_index_4 = None
for i in range(n_runs):
    print("="*50)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_4 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    kfcm_4.fit(Xs[0], ys[0])
    metrics_4.append(kfcm_4.evaluate())
    model_arrays.append({
        "U": kfcm_4._U,
        "G": kfcm_4._G,
        "s": kfcm_4._s,
    })
    if metrics_4[i]["loss"] < best_loss:
        best_loss = metrics_4[i]["loss"]
        best_index_4 = i

# %%
best_index_4, metrics_4[best_index_4]

# %%
# Metrics for m = 1.3
acc_vec4 = [metrics_4[i]["accuracy"] for i in range(n_runs)]
rand_vec4 = [metrics_4[i]["rand"] for i in range(n_runs)]
mpc_vec4 = [metrics_4[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec4):.4f}" + u" \u00b1 " + f"{np.std(acc_vec4):.4f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec4):.4f}" + u" \u00b1 " + f"{np.std(rand_vec4):.4f}")
print(f"Mean MPC: {np.mean(mpc_vec4):.4f}" + u" \u00b1 " + f"{np.std(mpc_vec4):.4f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u4 = pd.DataFrame(u3[best_j3])
best_u4 = np.array(model_arrays[best_index_4]["U"])
u4_crisp = np.zeros_like(best_u4)
for k in range(best_u4.shape[0]):
    max_index = np.argmax(best_u4[k])
    u4_crisp[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u4_crisp.shape[1]
clusters4 = np.zeros((num_groups, num_groups))
rows = u4_crisp.shape[0] // num_groups
for h in range(num_groups):
    clusters4[h] = np.sum(u4_crisp[h*rows:(h+1)*rows], axis=0)
clusters4 = pd.DataFrame(clusters4.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters4, annot=True, fmt="d")

# %%
clusters4

# %%
np.savetxt("metrics_4_s.csv", model_arrays[best_index_4]["s"], delimiter=",")
np.savetxt("metrics_4_G.csv", model_arrays[best_index_4]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_2_2_'></a>[Dataset 1](#toc0_)

# %%
n_clusters = 10
m = 1.4
max_iter = 100
tol = 1e-6
random_state = 42

n_runs = 50

model_arrays_4_1 = []
metrics_4_1 = []
best_loss = np.inf
best_index_4_1 = None
for i in range(n_runs):
    print("="*50)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_4_1 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 1
    kfcm_4_1.fit(Xs[1], ys[1])
    metrics_4_1.append(kfcm_4_1.evaluate())
    model_arrays_4_1.append({
        "U": kfcm_4_1._U,
        "G": kfcm_4_1._G,
        "s": kfcm_4_1._s,
    })
    if metrics_4_1[i]["loss"] < best_loss:
        best_loss = metrics_4_1[i]["loss"]
        best_index_4_1 = i

# %%
best_index_4_1, metrics_4_1[best_index_4_1]

# %%
# Metrics for m = 1.3
acc_vec4_1 = [metrics_4_1[i]["accuracy"] for i in range(n_runs)]
rand_vec4_1 = [metrics_4_1[i]["rand"] for i in range(n_runs)]
mpc_vec4_1 = [metrics_4_1[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec4_1):.4f}" + u" \u00b1 " + f"{np.std(acc_vec4_1):.4f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec4_1):.4f}" + u" \u00b1 " + f"{np.std(rand_vec4_1):.4f}")
print(f"Mean MPC: {np.mean(mpc_vec4_1):.4f}" + u" \u00b1 " + f"{np.std(mpc_vec4_1):.4f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
best_u4_1 = np.array(model_arrays_4_1[best_index_4_1]["U"])
u4_crisp_1 = np.zeros_like(best_u4_1)
for k in range(best_u4_1.shape[0]):
    max_index = np.argmax(best_u4_1[k])
    u4_crisp_1[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u4_crisp_1.shape[1]
clusters4_1 = np.zeros((num_groups, num_groups))
rows = u4_crisp_1.shape[0] // num_groups
for h in range(num_groups):
    clusters4_1[h] = np.sum(u4_crisp_1[h*rows:(h+1)*rows], axis=0)
clusters4_1 = pd.DataFrame(clusters4_1.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters4_1, annot=True, fmt="d")

# %%
clusters4_1

# %%
np.savetxt("metrics_4_1_s.csv", model_arrays_4_1[best_index_4_1]["s"], delimiter=",")
np.savetxt("metrics_4_1_G.csv", model_arrays_4_1[best_index_4_1]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_2_3_'></a>[Dataset 2](#toc0_)

# %%
n_clusters = 10
m = 1.4
max_iter = 100
tol = 1e-6
random_state = 42

n_runs = 50

model_arrays_4_2 = []
metrics_4_2 = []
best_loss = np.inf
best_index_4_2 = None
for i in range(n_runs):
    print("="*50)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_4_2 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 2
    kfcm_4_2.fit(Xs[2], ys[2])
    metrics_4_2.append(kfcm_4_2.evaluate())
    model_arrays_4_2.append({
        "U": kfcm_4_2._U,
        "G": kfcm_4_2._G,
        "s": kfcm_4_2._s,
    })
    if metrics_4_2[i]["loss"] < best_loss:
        best_loss = metrics_4_2[i]["loss"]
        best_index_4_2 = i

# %%
best_index_4_2, metrics_4_2[best_index_4_2]

# %%
# Metrics for m = 1.3
acc_vec4_2 = [metrics_4_2[i]["accuracy"] for i in range(n_runs)]
rand_vec4_2 = [metrics_4_2[i]["rand"] for i in range(n_runs)]
mpc_vec4_2 = [metrics_4_2[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec4_2):.4f}" + u" \u00b1 " + f"{np.std(acc_vec4_2):.4f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec4_2):.4f}" + u" \u00b1 " + f"{np.std(rand_vec4_2):.4f}")
print(f"Mean MPC: {np.mean(mpc_vec4_2):.4f}" + u" \u00b1 " + f"{np.std(mpc_vec4_2):.4f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
# RODA DNV
best_u4_2 = np.array(model_arrays_4_2[best_index_4_2]["U"])
u4_crisp_2 = np.zeros_like(best_u4_2)
for k in range(best_u4_2.shape[0]):
    max_index = np.argmax(best_u4_2[k])
    u4_crisp_2[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u4_crisp_2.shape[1]
clusters4_2 = np.zeros((num_groups, num_groups))
rows = u4_crisp_2.shape[0] // num_groups
for h in range(num_groups):
    clusters4_2[h] = np.sum(u4_crisp_2[h*rows:(h+1)*rows], axis=0)
clusters4_2 = pd.DataFrame(clusters4_2.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters4_2, annot=True, fmt="d")

# %%
clusters4_2

# %%
np.savetxt("metrics_4_2_s.csv", model_arrays_4_2[best_index_4_2]["s"], delimiter=",")
np.savetxt("metrics_4_2_G.csv", model_arrays_4_2[best_index_4_2]["G"], delimiter=",")

# %% [markdown]
# ## <a id='toc3_3_'></a>[m = 1.5](#toc0_)

# %% [markdown]
# ### <a id='toc3_3_1_'></a>[Dataset 0](#toc0_)

# %%
n_clusters = 10
m = 1.5
max_iter = 100
tol = 1e-6
n_runs = 50

model_arrays_5 = []
metrics_5 = []
best_loss = np.inf
best_index_5 = None
for i in range(n_runs):
    print("="*50)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_5 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    kfcm_5.fit(Xs[0], ys[0])
    metrics_5.append(kfcm_5.evaluate())
    model_arrays_5.append({
        "U": kfcm_5._U,
        "G": kfcm_5._G,
        "s": kfcm_5._s,
    })
    if metrics_5[i]["loss"] < best_loss:
        best_loss = metrics_5[i]["loss"]
        best_index_5 = i

# %%
best_index_5, metrics_5[best_index_5]

# %%
# Metrics for m = 1.3
acc_vec5 = [metrics_5[i]["accuracy"] for i in range(n_runs)]
rand_vec5 = [metrics_5[i]["rand"] for i in range(n_runs)]
mpc_vec5 = [metrics_5[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec5):.5f}" + u" \u00b1 " + f"{np.std(acc_vec5):.5f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec5):.5f}" + u" \u00b1 " + f"{np.std(rand_vec5):.5f}")
print(f"Mean MPC: {np.mean(mpc_vec5):.5f}" + u" \u00b1 " + f"{np.std(mpc_vec5):.5f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u5 = pd.DataFrame(u3[best_j3])
#RODA DE NOVO
best_u5 = np.array(model_arrays_5[best_index_5]["U"])
u5_crisp = np.zeros_like(best_u5)
for k in range(best_u5.shape[0]):
    max_index = np.argmax(best_u5[k])
    u5_crisp[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u5_crisp.shape[1]
clusters5 = np.zeros((num_groups, num_groups))
rows = u5_crisp.shape[0] // num_groups
for h in range(num_groups):
    clusters5[h] = np.sum(u5_crisp[h*rows:(h+1)*rows], axis=0)
clusters5 = pd.DataFrame(clusters5.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters5, annot=True, fmt="d")

# %%
clusters5

# %%
np.savetxt("metrics_5_s.csv", model_arrays_5[best_index_5]["s"], delimiter=",")
np.savetxt("metrics_5_G.csv", model_arrays_5[best_index_5]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_3_2_'></a>[Dataset 1](#toc0_)

# %%
n_clusters = 10
m = 1.5
max_iter = 100
tol = 1e-6
random_state = 42

n_runs = 50

model_arrays_5_1 = []
metrics_5_1 = []
best_loss = np.inf
best_index_5_1 = None
for i in range(n_runs):
    print("="*50)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_5_1 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 1
    kfcm_5_1.fit(Xs[1], ys[1])
    metrics_5_1.append(kfcm_5_1.evaluate())
    model_arrays_5_1.append({
        "U": kfcm_5_1._U,
        "G": kfcm_5_1._G,
        "s": kfcm_5_1._s,
    })
    if metrics_5_1[i]["loss"] < best_loss:
        best_loss = metrics_5_1[i]["loss"]
        best_index_5_1 = i

# %%
best_index_5_1, metrics_5_1[best_index_5_1]

# %%
# Metrics for m = 1.3
acc_vec5_1 = [metrics_5_1[i]["accuracy"] for i in range(n_runs)]
rand_vec5_1 = [metrics_5_1[i]["rand"] for i in range(n_runs)]
mpc_vec5_1 = [metrics_5_1[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec5_1):.5f}" + u" \u00b1 " + f"{np.std(acc_vec5_1):.5f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec5_1):.5f}" + u" \u00b1 " + f"{np.std(rand_vec5_1):.5f}")
print(f"Mean MPC: {np.mean(mpc_vec5_1):.5f}" + u" \u00b1 " + f"{np.std(mpc_vec5_1):.5f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
# RODA DNVB
best_u5_1 = np.array(model_arrays_5_1[best_index_5_1]["U"])
u5_crisp_1 = np.zeros_like(best_u5_1)
for k in range(best_u5_1.shape[0]):
    max_index = np.argmax(best_u5_1[k])
    u5_crisp_1[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u5_crisp_1.shape[1]
clusters5_1 = np.zeros((num_groups, num_groups))
rows = u5_crisp_1.shape[0] // num_groups
for h in range(num_groups):
    clusters5_1[h] = np.sum(u5_crisp_1[h*rows:(h+1)*rows], axis=0)
clusters5_1 = pd.DataFrame(clusters5_1.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters5_1, annot=True, fmt="d")

# %%
clusters5_1

# %%
np.savetxt("metrics_5_1_s.csv", model_arrays_5_1[best_index_5_1]["s"], delimiter=",")
np.savetxt("metrics_5_1_G.csv", model_arrays_5_1[best_index_5_1]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_3_3_'></a>[Dataset 2](#toc0_)

# %%
n_clusters = 10
m = 1.5
max_iter = 100
tol = 1e-6

n_runs = 50

model_arrays_5_2 = []
metrics_5_2 = []
best_loss = np.inf
best_index_5_2 = None
for i in range(n_runs):
    print("="*50)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_5_2 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 2
    kfcm_5_2.fit(Xs[2], ys[2])
    metrics_5_2.append(kfcm_5_2.evaluate())
    model_arrays_5_2.append({
        "U": kfcm_5_2._U,
        "G": kfcm_5_2._G,
        "s": kfcm_5_2._s,
    })
    if metrics_5_2[i]["loss"] < best_loss:
        best_loss = metrics_5_2[i]["loss"]
        best_index_5_2 = i

# %%
best_index_5_2, metrics_5_2[best_index_5_2]

# %%
# Metrics for m = 1.3
acc_vec5_2 = [metrics_5_2[i]["accuracy"] for i in range(n_runs)]
rand_vec5_2 = [metrics_5_2[i]["rand"] for i in range(n_runs)]
mpc_vec5_2 = [metrics_5_2[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec5_2):.5f}" + u" \u00b1 " + f"{np.std(acc_vec5_2):.5f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec5_2):.5f}" + u" \u00b1 " + f"{np.std(rand_vec5_2):.5f}")
print(f"Mean MPC: {np.mean(mpc_vec5_2):.5f}" + u" \u00b1 " + f"{np.std(mpc_vec5_2):.5f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
best_u5_2 = np.array(model_arrays_5_2[best_index_5_2]["U"])
u5_crisp_2 = np.zeros_like(best_u5_2)
for k in range(best_u5_2.shape[0]):
    max_index = np.argmax(best_u5_2[k])
    u5_crisp_2[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u5_crisp_2.shape[1]
clusters5_2 = np.zeros((num_groups, num_groups))
rows = u5_crisp_2.shape[0] // num_groups
for h in range(num_groups):
    clusters5_2[h] = np.sum(u5_crisp_2[h*rows:(h+1)*rows], axis=0)
clusters5_2 = pd.DataFrame(clusters5_2.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters5_2, annot=True, fmt="d")

# %%
clusters5_2

# %%
np.savetxt("metrics_5_2_s.csv", model_arrays_5_2[best_index_5_2]["s"], delimiter=",")
np.savetxt("metrics_5_2_G.csv", model_arrays_5_2[best_index_5_2]["G"], delimiter=",")

# %% [markdown]
# ## <a id='toc3_4_'></a>[m = 1.6](#toc0_)

# %% [markdown]
# ### <a id='toc3_4_1_'></a>[Dataset 0](#toc0_)

# %%
n_clusters = 10
m = 1.6
max_iter = 100
tol = 1e-6
n_runs = 50

model_arrays_6 = []
metrics_6 = []
best_loss = np.inf
best_index_6 = None
for i in range(n_runs):
    print("="*60)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_6 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    kfcm_6.fit(Xs[0], ys[0])
    metrics_6.append(kfcm_6.evaluate())
    model_arrays_6.append({
        "U": kfcm_6._U,
        "G": kfcm_6._G,
        "s": kfcm_6._s,
    })
    if metrics_6[i]["loss"] < best_loss:
        best_loss = metrics_6[i]["loss"]
        best_index_6 = i

# %%
best_index_6, metrics_6[best_index_6]

# %%
# Metrics for m = 1.3
acc_vec6 = [metrics_6[i]["accuracy"] for i in range(n_runs)]
rand_vec6 = [metrics_6[i]["rand"] for i in range(n_runs)]
mpc_vec6 = [metrics_6[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec6):.6f}" + u" \u00b1 " + f"{np.std(acc_vec6):.6f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec6):.6f}" + u" \u00b1 " + f"{np.std(rand_vec6):.6f}")
print(f"Mean MPC: {np.mean(mpc_vec6):.6f}" + u" \u00b1 " + f"{np.std(mpc_vec6):.6f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u6 = pd.DataFrame(u3[best_j3])
#RODA DE NOVO
best_u6 = np.array(model_arrays_6[best_index_6]["U"])
u6_crisp = np.zeros_like(best_u6)
for k in range(best_u6.shape[0]):
    max_index = np.argmax(best_u6[k])
    u6_crisp[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u6_crisp.shape[1]
clusters6 = np.zeros((num_groups, num_groups))
rows = u6_crisp.shape[0] // num_groups
for h in range(num_groups):
    clusters6[h] = np.sum(u6_crisp[h*rows:(h+1)*rows], axis=0)
clusters6 = pd.DataFrame(clusters6.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters6, annot=True, fmt="d")

# %%
clusters6

# %%
np.savetxt("metrics_6_s.csv", model_arrays_6[best_index_6]["s"], delimiter=",")
np.savetxt("metrics_6_G.csv", model_arrays_6[best_index_6]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_4_2_'></a>[Dataset 1](#toc0_)

# %%
n_clusters = 10
m = 1.6
max_iter = 100
tol = 1e-6
random_state = 42

n_runs = 50

model_arrays_6_1 = []
metrics_6_1 = []
best_loss = np.inf
best_index_6_1 = None
for i in range(n_runs):
    print("="*60)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_6_1 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 1
    kfcm_6_1.fit(Xs[1], ys[1])
    metrics_6_1.append(kfcm_6_1.evaluate())
    model_arrays_6_1.append({
        "U": kfcm_6_1._U,
        "G": kfcm_6_1._G,
        "s": kfcm_6_1._s,
    })
    if metrics_6_1[i]["loss"] < best_loss:
        best_loss = metrics_6_1[i]["loss"]
        best_index_6_1 = i

# %%
best_index_6_1, metrics_6_1[best_index_6_1]

# %%
# Metrics for m = 1.3
acc_vec6_1 = [metrics_6_1[i]["accuracy"] for i in range(n_runs)]
rand_vec6_1 = [metrics_6_1[i]["rand"] for i in range(n_runs)]
mpc_vec6_1 = [metrics_6_1[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec6_1):.6f}" + u" \u00b1 " + f"{np.std(acc_vec6_1):.6f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec6_1):.6f}" + u" \u00b1 " + f"{np.std(rand_vec6_1):.6f}")
print(f"Mean MPC: {np.mean(mpc_vec6_1):.6f}" + u" \u00b1 " + f"{np.std(mpc_vec6_1):.6f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
# RODA DNVB
best_u6_1 = np.array(model_arrays_6_1[best_index_6_1]["U"])
u6_crisp_1 = np.zeros_like(best_u6_1)
for k in range(best_u6_1.shape[0]):
    max_index = np.argmax(best_u6_1[k])
    u6_crisp_1[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u6_crisp_1.shape[1]
clusters6_1 = np.zeros((num_groups, num_groups))
rows = u6_crisp_1.shape[0] // num_groups
for h in range(num_groups):
    clusters6_1[h] = np.sum(u6_crisp_1[h*rows:(h+1)*rows], axis=0)
clusters6_1 = pd.DataFrame(clusters6_1.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters6_1, annot=True, fmt="d")

# %%
clusters6_1

# %%
np.savetxt("metrics_6_1_s.csv", model_arrays_6_1[best_index_6_1]["s"], delimiter=",")
np.savetxt("metrics_6_1_G.csv", model_arrays_6_1[best_index_6_1]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_4_3_'></a>[Dataset 2](#toc0_)

# %%
n_clusters = 10
m = 1.6
max_iter = 100
tol = 1e-6

n_runs = 50

model_arrays_6_2 = []
metrics_6_2 = []
best_loss = np.inf
best_index_6_2 = None
for i in range(n_runs):
    print("="*60)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_6_2 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 2
    kfcm_6_2.fit(Xs[2], ys[2])
    metrics_6_2.append(kfcm_6_2.evaluate())
    model_arrays_6_2.append({
        "U": kfcm_6_2._U,
        "G": kfcm_6_2._G,
        "s": kfcm_6_2._s,
    })
    if metrics_6_2[i]["loss"] < best_loss:
        best_loss = metrics_6_2[i]["loss"]
        best_index_6_2 = i

# %%
best_index_6_2, metrics_6_2[best_index_6_2]

# %%
# Metrics for m = 1.3
acc_vec6_2 = [metrics_6_2[i]["accuracy"] for i in range(n_runs)]
rand_vec6_2 = [metrics_6_2[i]["rand"] for i in range(n_runs)]
mpc_vec6_2 = [metrics_6_2[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec6_2):.6f}" + u" \u00b1 " + f"{np.std(acc_vec6_2):.6f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec6_2):.6f}" + u" \u00b1 " + f"{np.std(rand_vec6_2):.6f}")
print(f"Mean MPC: {np.mean(mpc_vec6_2):.6f}" + u" \u00b1 " + f"{np.std(mpc_vec6_2):.6f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
best_u6_2 = np.array(model_arrays_6_2[best_index_6_2]["U"])
u6_crisp_2 = np.zeros_like(best_u6_2)
for k in range(best_u6_2.shape[0]):
    max_index = np.argmax(best_u6_2[k])
    u6_crisp_2[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u6_crisp_2.shape[1]
clusters6_2 = np.zeros((num_groups, num_groups))
rows = u6_crisp_2.shape[0] // num_groups
for h in range(num_groups):
    clusters6_2[h] = np.sum(u6_crisp_2[h*rows:(h+1)*rows], axis=0)
clusters6_2 = pd.DataFrame(clusters6_2.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters6_2, annot=True, fmt="d")

# %%
clusters6_2

# %%
np.savetxt("metrics_6_2_s.csv", model_arrays_6_2[best_index_6_2]["s"], delimiter=",")
np.savetxt("metrics_6_2_G.csv", model_arrays_6_2[best_index_6_2]["G"], delimiter=",")

# %% [markdown]
# ## <a id='toc3_5_'></a>[m = 1.7](#toc0_)

# %% [markdown]
# ### <a id='toc3_5_1_'></a>[Dataset 0](#toc0_)

# %%
n_clusters = 10
m = 1.7
max_iter = 100
tol = 1e-6
n_runs = 50

model_arrays_7 = []
metrics_7 = []
best_loss = np.inf
best_index_7 = None
for i in range(n_runs):
    print("="*70)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_7 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    kfcm_7.fit(Xs[0], ys[0])
    metrics_7.append(kfcm_7.evaluate())
    model_arrays_7.append({
        "U": kfcm_7._U,
        "G": kfcm_7._G,
        "s": kfcm_7._s,
    })
    if metrics_7[i]["loss"] < best_loss:
        best_loss = metrics_7[i]["loss"]
        best_index_7 = i

# %%
best_index_7, metrics_7[best_index_7]

# %%
# Metrics for m = 1.3
acc_vec7 = [metrics_7[i]["accuracy"] for i in range(n_runs)]
rand_vec7 = [metrics_7[i]["rand"] for i in range(n_runs)]
mpc_vec7 = [metrics_7[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec7):.7f}" + u" \u00b1 " + f"{np.std(acc_vec7):.7f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec7):.7f}" + u" \u00b1 " + f"{np.std(rand_vec7):.7f}")
print(f"Mean MPC: {np.mean(mpc_vec7):.7f}" + u" \u00b1 " + f"{np.std(mpc_vec7):.7f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u7 = pd.DataFrame(u3[best_j3])
#RODA DE NOVO
best_u7 = np.array(model_arrays_7[best_index_7]["U"])
u7_crisp = np.zeros_like(best_u7)
for k in range(best_u7.shape[0]):
    max_index = np.argmax(best_u7[k])
    u7_crisp[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u7_crisp.shape[1]
clusters7 = np.zeros((num_groups, num_groups))
rows = u7_crisp.shape[0] // num_groups
for h in range(num_groups):
    clusters7[h] = np.sum(u7_crisp[h*rows:(h+1)*rows], axis=0)
clusters7 = pd.DataFrame(clusters7.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters7, annot=True, fmt="d")

# %%
clusters7

# %%
np.savetxt("metrics_7_s.csv", model_arrays_7[best_index_7]["s"], delimiter=",")
np.savetxt("metrics_7_G.csv", model_arrays_7[best_index_7]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_5_2_'></a>[Dataset 1](#toc0_)

# %%
n_clusters = 10
m = 1.7
max_iter = 100
tol = 1e-6
random_state = 42

n_runs = 50

model_arrays_7_1 = []
metrics_7_1 = []
best_loss = np.inf
best_index_7_1 = None
for i in range(n_runs):
    print("="*70)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_7_1 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 1
    kfcm_7_1.fit(Xs[1], ys[1])
    metrics_7_1.append(kfcm_7_1.evaluate())
    model_arrays_7_1.append({
        "U": kfcm_7_1._U,
        "G": kfcm_7_1._G,
        "s": kfcm_7_1._s,
    })
    if metrics_7_1[i]["loss"] < best_loss:
        best_loss = metrics_7_1[i]["loss"]
        best_index_7_1 = i

# %%
best_index_7_1, metrics_7_1[best_index_7_1]

# %%
# Metrics for m = 1.3
acc_vec7_1 = [metrics_7_1[i]["accuracy"] for i in range(n_runs)]
rand_vec7_1 = [metrics_7_1[i]["rand"] for i in range(n_runs)]
mpc_vec7_1 = [metrics_7_1[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec7_1):.7f}" + u" \u00b1 " + f"{np.std(acc_vec7_1):.7f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec7_1):.7f}" + u" \u00b1 " + f"{np.std(rand_vec7_1):.7f}")
print(f"Mean MPC: {np.mean(mpc_vec7_1):.7f}" + u" \u00b1 " + f"{np.std(mpc_vec7_1):.7f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
# RODA DNVB
best_u7_1 = np.array(model_arrays_7_1[best_index_7_1]["U"])
u7_crisp_1 = np.zeros_like(best_u7_1)
for k in range(best_u7_1.shape[0]):
    max_index = np.argmax(best_u7_1[k])
    u7_crisp_1[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u7_crisp_1.shape[1]
clusters7_1 = np.zeros((num_groups, num_groups))
rows = u7_crisp_1.shape[0] // num_groups
for h in range(num_groups):
    clusters7_1[h] = np.sum(u7_crisp_1[h*rows:(h+1)*rows], axis=0)
clusters7_1 = pd.DataFrame(clusters7_1.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters7_1, annot=True, fmt="d")

# %%
clusters7_1

# %%

np.savetxt("metrics_7_1_s.csv", model_arrays_7_1[best_index_7_1]["s"], delimiter=",")
np.savetxt("metrics_7_1_G.csv", model_arrays_7_1[best_index_7_1]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_5_3_'></a>[Dataset 2](#toc0_)

# %%
n_clusters = 10
m = 1.7
max_iter = 100
tol = 1e-6

n_runs = 50

model_arrays_7_2 = []
metrics_7_2 = []
best_loss = np.inf
best_index_7_2 = None
for i in range(n_runs):
    print("="*70)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_7_2 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 2
    kfcm_7_2.fit(Xs[2], ys[2])
    metrics_7_2.append(kfcm_7_2.evaluate())
    model_arrays_7_2.append({
        "U": kfcm_7_2._U,
        "G": kfcm_7_2._G,
        "s": kfcm_7_2._s,
    })
    if metrics_7_2[i]["loss"] < best_loss:
        best_loss = metrics_7_2[i]["loss"]
        best_index_7_2 = i

# %%
best_index_7_2, metrics_7_2[best_index_7_2]

# %%
# Metrics for m = 1.3
acc_vec7_2 = [metrics_7_2[i]["accuracy"] for i in range(n_runs)]
rand_vec7_2 = [metrics_7_2[i]["rand"] for i in range(n_runs)]
mpc_vec7_2 = [metrics_7_2[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec7_2):.7f}" + u" \u00b1 " + f"{np.std(acc_vec7_2):.7f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec7_2):.7f}" + u" \u00b1 " + f"{np.std(rand_vec7_2):.7f}")
print(f"Mean MPC: {np.mean(mpc_vec7_2):.7f}" + u" \u00b1 " + f"{np.std(mpc_vec7_2):.7f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
best_u7_2 = np.array(model_arrays_7_2[best_index_7_2]["U"])
u7_crisp_2 = np.zeros_like(best_u7_2)
for k in range(best_u7_2.shape[0]):
    max_index = np.argmax(best_u7_2[k])
    u7_crisp_2[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u7_crisp_2.shape[1]
clusters7_2 = np.zeros((num_groups, num_groups))
rows = u7_crisp_2.shape[0] // num_groups
for h in range(num_groups):
    clusters7_2[h] = np.sum(u7_crisp_2[h*rows:(h+1)*rows], axis=0)
clusters7_2 = pd.DataFrame(clusters7_2.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters7_2, annot=True, fmt="d")

# %%
clusters7_2

# %%

np.savetxt("metrics_7_2_s.csv", model_arrays_7_2[best_index_7_2]["s"], delimiter=",")
np.savetxt("metrics_7_2_G.csv", model_arrays_7_2[best_index_7_2]["G"], delimiter=",")

# %% [markdown]
# ## <a id='toc3_6_'></a>[m = 1.8](#toc0_)

# %% [markdown]
# ### <a id='toc3_6_1_'></a>[Dataset 0](#toc0_)

# %%
n_clusters = 10
m = 1.8
max_iter = 100
tol = 1e-6
n_runs = 50

model_arrays_8 = []
metrics_8 = []
best_loss = np.inf
best_index_8 = None
for i in range(n_runs):
    print("="*80)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_8 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    kfcm_8.fit(Xs[0], ys[0])
    metrics_8.append(kfcm_8.evaluate())
    model_arrays_8.append({
        "U": kfcm_8._U,
        "G": kfcm_8._G,
        "s": kfcm_8._s,
    })
    if metrics_8[i]["loss"] < best_loss:
        best_loss = metrics_8[i]["loss"]
        best_index_8 = i

# %%
best_index_8, metrics_8[best_index_8]

# %%
# Metrics for m = 1.3
acc_vec8 = [metrics_8[i]["accuracy"] for i in range(n_runs)]
rand_vec8 = [metrics_8[i]["rand"] for i in range(n_runs)]
mpc_vec8 = [metrics_8[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec8):.8f}" + u" \u00b1 " + f"{np.std(acc_vec8):.8f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec8):.8f}" + u" \u00b1 " + f"{np.std(rand_vec8):.8f}")
print(f"Mean MPC: {np.mean(mpc_vec8):.8f}" + u" \u00b1 " + f"{np.std(mpc_vec8):.8f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u8 = pd.DataFrame(u3[best_j3])
#RODA DE NOVO
best_u8 = np.array(model_arrays_8[best_index_8]["U"])
u8_crisp = np.zeros_like(best_u8)
for k in range(best_u8.shape[0]):
    max_index = np.argmax(best_u8[k])
    u8_crisp[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u8_crisp.shape[1]
clusters8 = np.zeros((num_groups, num_groups))
rows = u8_crisp.shape[0] // num_groups
for h in range(num_groups):
    clusters8[h] = np.sum(u8_crisp[h*rows:(h+1)*rows], axis=0)
clusters8 = pd.DataFrame(clusters8.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters8, annot=True, fmt="d")

# %%
clusters8

# %%
np.savetxt("metrics_8_s.csv", model_arrays_8[best_index_8]["s"], delimiter=",")
np.savetxt("metrics_8_G.csv", model_arrays_8[best_index_8]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_6_2_'></a>[Dataset 1](#toc0_)

# %%
n_clusters = 10
m = 1.8
max_iter = 100
tol = 1e-6
random_state = 42

n_runs = 50

model_arrays_8_1 = []
metrics_8_1 = []
best_loss = np.inf
best_index_8_1 = None
for i in range(n_runs):
    print("="*80)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_8_1 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 1
    kfcm_8_1.fit(Xs[1], ys[1])
    metrics_8_1.append(kfcm_8_1.evaluate())
    model_arrays_8_1.append({
        "U": kfcm_8_1._U,
        "G": kfcm_8_1._G,
        "s": kfcm_8_1._s,
    })
    if metrics_8_1[i]["loss"] < best_loss:
        best_loss = metrics_8_1[i]["loss"]
        best_index_8_1 = i

# %%
best_index_8_1, metrics_8_1[best_index_8_1]

# %%
# Metrics for m = 1.3
acc_vec8_1 = [metrics_8_1[i]["accuracy"] for i in range(n_runs)]
rand_vec8_1 = [metrics_8_1[i]["rand"] for i in range(n_runs)]
mpc_vec8_1 = [metrics_8_1[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec8_1):.8f}" + u" \u00b1 " + f"{np.std(acc_vec8_1):.8f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec8_1):.8f}" + u" \u00b1 " + f"{np.std(rand_vec8_1):.8f}")
print(f"Mean MPC: {np.mean(mpc_vec8_1):.8f}" + u" \u00b1 " + f"{np.std(mpc_vec8_1):.8f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
# RODA DNVB
best_u8_1 = np.array(model_arrays_8_1[best_index_8_1]["U"])
u8_crisp_1 = np.zeros_like(best_u8_1)
for k in range(best_u8_1.shape[0]):
    max_index = np.argmax(best_u8_1[k])
    u8_crisp_1[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u8_crisp_1.shape[1]
clusters8_1 = np.zeros((num_groups, num_groups))
rows = u8_crisp_1.shape[0] // num_groups
for h in range(num_groups):
    clusters8_1[h] = np.sum(u8_crisp_1[h*rows:(h+1)*rows], axis=0)
clusters8_1 = pd.DataFrame(clusters8_1.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters8_1, annot=True, fmt="d")

# %%
clusters8_1

# %%

np.savetxt("metrics_8_1_s.csv", model_arrays_8_1[best_index_8_1]["s"], delimiter=",")
np.savetxt("metrics_8_1_G.csv", model_arrays_8_1[best_index_8_1]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_6_3_'></a>[Dataset 2](#toc0_)

# %%
n_clusters = 10
m = 1.8
max_iter = 100
tol = 1e-6

n_runs = 50

model_arrays_8_2 = []
metrics_8_2 = []
best_loss = np.inf
best_index_8_2 = None
for i in range(n_runs):
    print("="*80)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_8_2 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 2
    kfcm_8_2.fit(Xs[2], ys[2])
    metrics_8_2.append(kfcm_8_2.evaluate())
    model_arrays_8_2.append({
        "U": kfcm_8_2._U,
        "G": kfcm_8_2._G,
        "s": kfcm_8_2._s,
    })
    if metrics_8_2[i]["loss"] < best_loss:
        best_loss = metrics_8_2[i]["loss"]
        best_index_8_2 = i

# %%
best_index_8_2, metrics_8_2[best_index_8_2]

# %%
# Metrics for m = 1.3
acc_vec8_2 = [metrics_8_2[i]["accuracy"] for i in range(n_runs)]
rand_vec8_2 = [metrics_8_2[i]["rand"] for i in range(n_runs)]
mpc_vec8_2 = [metrics_8_2[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec8_2):.8f}" + u" \u00b1 " + f"{np.std(acc_vec8_2):.8f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec8_2):.8f}" + u" \u00b1 " + f"{np.std(rand_vec8_2):.8f}")
print(f"Mean MPC: {np.mean(mpc_vec8_2):.8f}" + u" \u00b1 " + f"{np.std(mpc_vec8_2):.8f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
best_u8_2 = np.array(model_arrays_8_2[best_index_8_2]["U"])
u8_crisp_2 = np.zeros_like(best_u8_2)
for k in range(best_u8_2.shape[0]):
    max_index = np.argmax(best_u8_2[k])
    u8_crisp_2[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u8_crisp_2.shape[1]
clusters8_2 = np.zeros((num_groups, num_groups))
rows = u8_crisp_2.shape[0] // num_groups
for h in range(num_groups):
    clusters8_2[h] = np.sum(u8_crisp_2[h*rows:(h+1)*rows], axis=0)
clusters8_2 = pd.DataFrame(clusters8_2.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters8_2, annot=True, fmt="d")

# %%
clusters8_2

# %%
np.savetxt("metrics_8_2_s.csv", model_arrays_8_2[best_index_8_2]["s"], delimiter=",")
np.savetxt("metrics_8_2_G.csv", model_arrays_8_2[best_index_8_2]["G"], delimiter=",")

# %% [markdown]
# ## <a id='toc3_7_'></a>[m = 1.9](#toc0_)

# %% [markdown]
# ### <a id='toc3_7_1_'></a>[Dataset 0](#toc0_)

# %%
n_clusters = 10
m = 1.9
max_iter = 100
tol = 1e-6
n_runs = 50

model_arrays_9 = []
metrics_9 = []
best_loss = np.inf
best_index_9 = None
for i in range(n_runs):
    print("="*90)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_9 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    kfcm_9.fit(Xs[0], ys[0])
    metrics_9.append(kfcm_9.evaluate())
    model_arrays_9.append({
        "U": kfcm_9._U,
        "G": kfcm_9._G,
        "s": kfcm_9._s,
    })
    if metrics_9[i]["loss"] < best_loss:
        best_loss = metrics_9[i]["loss"]
        best_index_9 = i

# %%
best_index_9, metrics_9[best_index_9]

# %%
# Metrics for m = 1.3
acc_vec9 = [metrics_9[i]["accuracy"] for i in range(n_runs)]
rand_vec9 = [metrics_9[i]["rand"] for i in range(n_runs)]
mpc_vec9 = [metrics_9[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec9):.9f}" + u" \u00b1 " + f"{np.std(acc_vec9):.9f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec9):.9f}" + u" \u00b1 " + f"{np.std(rand_vec9):.9f}")
print(f"Mean MPC: {np.mean(mpc_vec9):.9f}" + u" \u00b1 " + f"{np.std(mpc_vec9):.9f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u9 = pd.DataFrame(u3[best_j3])
#RODA DE NOVO
best_u9 = np.array(model_arrays_9[best_index_9]["U"])
u9_crisp = np.zeros_like(best_u9)
for k in range(best_u9.shape[0]):
    max_index = np.argmax(best_u9[k])
    u9_crisp[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u9_crisp.shape[1]
clusters9 = np.zeros((num_groups, num_groups))
rows = u9_crisp.shape[0] // num_groups
for h in range(num_groups):
    clusters9[h] = np.sum(u9_crisp[h*rows:(h+1)*rows], axis=0)
clusters9 = pd.DataFrame(clusters9.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters9, annot=True, fmt="d")

# %%
clusters9

# %%
np.savetxt("metrics_9_s.csv", model_arrays_9[best_index_9]["s"], delimiter=",")
np.savetxt("metrics_9_G.csv", model_arrays_9[best_index_9]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_7_2_'></a>[Dataset 1](#toc0_)

# %%
n_clusters = 10
m = 1.9
max_iter = 100
tol = 1e-6
random_state = 42

n_runs = 50

model_arrays_9_1 = []
metrics_9_1 = []
best_loss = np.inf
best_index_9_1 = None
for i in range(n_runs):
    print("="*90)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_9_1 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 1
    kfcm_9_1.fit(Xs[1], ys[1])
    metrics_9_1.append(kfcm_9_1.evaluate())
    model_arrays_9_1.append({
        "U": kfcm_9_1._U,
        "G": kfcm_9_1._G,
        "s": kfcm_9_1._s,
    })
    if metrics_9_1[i]["loss"] < best_loss:
        best_loss = metrics_9_1[i]["loss"]
        best_index_9_1 = i

# %%
best_index_9_1, metrics_9_1[best_index_9_1]

# %%
# Metrics for m = 1.3
acc_vec9_1 = [metrics_9_1[i]["accuracy"] for i in range(n_runs)]
rand_vec9_1 = [metrics_9_1[i]["rand"] for i in range(n_runs)]
mpc_vec9_1 = [metrics_9_1[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec9_1):.9f}" + u" \u00b1 " + f"{np.std(acc_vec9_1):.9f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec9_1):.9f}" + u" \u00b1 " + f"{np.std(rand_vec9_1):.9f}")
print(f"Mean MPC: {np.mean(mpc_vec9_1):.9f}" + u" \u00b1 " + f"{np.std(mpc_vec9_1):.9f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
# RODA DNVB
best_u9_1 = np.array(model_arrays_9_1[best_index_9_1]["U"])
u9_crisp_1 = np.zeros_like(best_u9_1)
for k in range(best_u9_1.shape[0]):
    max_index = np.argmax(best_u9_1[k])
    u9_crisp_1[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u9_crisp_1.shape[1]
clusters9_1 = np.zeros((num_groups, num_groups))
rows = u9_crisp_1.shape[0] // num_groups
for h in range(num_groups):
    clusters9_1[h] = np.sum(u9_crisp_1[h*rows:(h+1)*rows], axis=0)
clusters9_1 = pd.DataFrame(clusters9_1.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters9_1, annot=True, fmt="d")

# %%
clusters9_1

# %%
np.savetxt("metrics_9_1_s.csv", model_arrays_9_1[best_index_9_1]["s"], delimiter=",")
np.savetxt("metrics_9_1_G.csv", model_arrays_9_1[best_index_9_1]["G"], delimiter=",")

# %% [markdown]
# ### <a id='toc3_7_3_'></a>[Dataset 2](#toc0_)

# %%
n_clusters = 10
m = 1.9
max_iter = 100
tol = 1e-6

n_runs = 50

model_arrays_9_2 = []
metrics_9_2 = []
best_loss = np.inf
best_index_9_2 = None
for i in range(n_runs):
    print("="*90)
    print(f"-> Run {i+1}/{n_runs}")
    kfcm_9_2 = KFCM_K_W1(n_clusters=n_clusters, m=m, n_iter=max_iter, epsilon=tol, verbose=True, random_state=i)
    # Dataset 2
    kfcm_9_2.fit(Xs[2], ys[2])
    metrics_9_2.append(kfcm_9_2.evaluate())
    model_arrays_9_2.append({
        "U": kfcm_9_2._U,
        "G": kfcm_9_2._G,
        "s": kfcm_9_2._s,
    })
    if metrics_9_2[i]["loss"] < best_loss:
        best_loss = metrics_9_2[i]["loss"]
        best_index_9_2 = i

# %%
best_index_9_2, metrics_9_2[best_index_9_2]

# %%
# Metrics for m = 1.3
acc_vec9_2 = [metrics_9_2[i]["accuracy"] for i in range(n_runs)]
rand_vec9_2 = [metrics_9_2[i]["rand"] for i in range(n_runs)]
mpc_vec9_2 = [metrics_9_2[i]["MPC"] for i in range(n_runs)]

print(f"Mean Accuracy: {np.mean(acc_vec9_2):.9f}" + u" \u00b1 " + f"{np.std(acc_vec9_2):.9f}")
print(f"Mean Adjusted Rand Score: {np.mean(rand_vec9_2):.9f}" + u" \u00b1 " + f"{np.std(rand_vec9_2):.9f}")
print(f"Mean MPC: {np.mean(mpc_vec9_2):.9f}" + u" \u00b1 " + f"{np.std(mpc_vec9_2):.9f}")

# %%
# Crisp partition of the best result for m = 1.3
#best_u3 = pd.DataFrame(u3[best_j3])
best_u9_2 = np.array(model_arrays_9_2[best_index_9_2]["U"])
u9_crisp_2 = np.zeros_like(best_u9_2)
for k in range(best_u9_2.shape[0]):
    max_index = np.argmax(best_u9_2[k])
    u9_crisp_2[k][max_index] = 1

# %%
# Confusion matrix for m = 1.3
num_groups = u9_crisp_2.shape[1]
clusters9_2 = np.zeros((num_groups, num_groups))
rows = u9_crisp_2.shape[0] // num_groups
for h in range(num_groups):
    clusters9_2[h] = np.sum(u9_crisp_2[h*rows:(h+1)*rows], axis=0)
clusters9_2 = pd.DataFrame(clusters9_2.astype(int))

with sns.plotting_context():
    sns.heatmap(clusters9_2, annot=True, fmt="d")

# %%
clusters9_2

# %%
np.savetxt("metrics_9_2_s.csv", model_arrays_9_2[best_index_9_2]["s"], delimiter=",")
np.savetxt("metrics_9_2_G.csv", model_arrays_9_2[best_index_9_2]["G"], delimiter=",")


