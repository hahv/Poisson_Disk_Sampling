import numpy as np
from poisson_disk import PoissonDiskSampling
from matplotlib import pyplot as plt

def read_file_to_array(f_name):
    with open(f_name) as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    X = []
    for line in content:
        txt_array = line.split(" ")
        f_array = [float(text) for text in txt_array]
        X.append(f_array)

    X = np.array(X)

    return X


f_name = 'sample_input.txt'
X_transform = read_file_to_array(f_name)

# f_name = 'sample_output.txt'
# X_transform_poison = read_file_to_array(f_name)

# Poisson sampling
sampler = PoissonDiskSampling(X_transform)
output_size = 500
poisson_indices = sampler.do_sample_with_size(output_size)
X_transform_poison = X_transform[poisson_indices]

print 'Plot PCA'

plt.figure(figsize=(15, 15))

a = plt.scatter(X_transform[:, 0], X_transform[:, 1], color="grey", label='all')
b = plt.scatter(X_transform_poison[:, 0], X_transform_poison[:, 1], color="red", label='poisson')

plt.legend((a, b),
           ('all', 'poisson'),
           scatterpoints=1,
           loc='lower left',
           ncol=2,
           fontsize=8)
plt.savefig('possion_sampling_result.png', bbox_inches='tight')
plt.show()