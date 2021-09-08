import gpumap
from sklearn.datasets import load_digits

digits = load_digits()

embedding = gpumap.GPUMAP().fit_transform(digits.data)
print(embedding)
