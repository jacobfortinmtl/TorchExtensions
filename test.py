import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import nan_cpp as nanCPP
import time
from nan_ops import NaNPool2d, NormalPool2d, NaNConv2d, NormalConv2d

FILENAME = "mri_data//sample_data//outp_bn3_1.pkl"
# Load the data from the pickle file
data = pickle.load(open(FILENAME, 'rb'))

if isinstance(data, tuple):
    a = data[0]
else:
    a = data

nanoutput, nanindices = nanCPP.NaNPool2d(a, (2,2), 1, (2,2))
nanpoolPy = NaNPool2d(max_threshold=1)
nanoutputPy, nanindicesPy = nanpoolPy(a, pool_size=(2,2), strides=(2,2))  # pool_size == kernel_size

# Find the indices where nanoutput and nanoutputPy are not equal
diff_indices = torch.ne(nanoutput, nanoutputPy)

with open('output.txt', 'w') as f:
    f.write("Differing Output Values:\n")
    f.write(str(nanoutput[diff_indices]) + '\n')
    f.write("Differing Output Indices:\n")
    f.write(str(diff_indices.nonzero()) + '\n')
print(nanoutput, nanoutputPy)
print(torch.equal(nanoutput, nanoutputPy))

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

sns.heatmap(np.mean(nanoutput.detach().numpy().squeeze(), axis=0), ax=axes[0][0])
axes[0][0].set_title('CPP Pool Threshold 0.25')

sns.heatmap(np.mean(nanoutputPy.detach().numpy().squeeze(), axis=0), ax=axes[0][1])
axes[0][1].set_title('Python Pool Threshold 0.25')

plt.savefig('output.png')  # Save the plot to 'output.png'