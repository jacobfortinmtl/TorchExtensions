import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import nan_cpp as nanCPP
from nan_ops import NaNPool2d, NormalPool2d

FILENAME = "mri_data//sample_data//outp_bn3_1.pkl"
# Load the data from the pickle file
data = pickle.load(open(FILENAME, 'rb'))

if isinstance(data, tuple):
    a = data[0]
else:
    a = data

#! INES POOL nanpool = NaNPool2d(threshold=0.25)
#! INES POOL nanoutput, nanindices = nanpool(a, pool_size=(2,2), strides=(2,2))  # pool_size == kernel_size
nanoutput, nanindices = nanCPP.NaNPool2d(a, (2,2), 0.50, None)
normalpool = NormalPool2d(threshold=1)
normaloutput, normalindices = normalpool(a, pool_size=(2,2), strides=(2,2))  # pool_size == kernel_size
pool = torch.nn.MaxPool2d(2, 2, return_indices=True)
torchoutput, torchindices = pool(a)

# Visualize output
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(np.mean(nanoutput.detach().numpy().squeeze(), axis=0), ax=axes[0])
axes[0].set_title('CPP NaN Pool Threshold 0.50')
sns.heatmap(np.mean(normaloutput.detach().numpy().squeeze(), axis=0), ax=axes[1])
axes[1].set_title('Manual Pool')
sns.heatmap(np.mean(torchoutput.detach().numpy().squeeze(), axis=0), ax=axes[2])
axes[2].set_title('Torch Pool')

plt.savefig('/workspace/output_comparison.png')  # Save the first plot
plt.close(fig)  # Close the figure to free memory

# Visualize output differences -- should be 0
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(np.mean((nanoutput - torchoutput).detach().numpy().squeeze(), axis=0), ax=axes[0])
axes[0].set_title('NaN Pool Threshold 0.50')
sns.heatmap(np.mean((normaloutput - torchoutput).detach().numpy().squeeze(), axis=0), ax=axes[1])
axes[1].set_title('Manual Pool')
sns.heatmap(np.mean((torchoutput - torchoutput).detach().numpy().squeeze(), axis=0), ax=axes[2])
axes[2].set_title('Torch Pool')

plt.suptitle('Differences Across NaN and Torch Pool')
plt.savefig('/workspace/differences_comparison.png')  # Save the second plot
plt.close(fig)  # Close the figure to free memory
