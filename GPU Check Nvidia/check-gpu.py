from numba import cuda
import numpy as np

@cuda.jit
def multiply_by_two(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] *= 2

n = 1024
a = np.arange(n).astype(np.float32)
d_a = cuda.to_device(a)

threads_per_block = 128
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

multiply_by_two[blocks_per_grid, threads_per_block](d_a)
result = d_a.copy_to_host()

print(result[:10])  # Debería mostrar [0. 2. 4. 6. 8. 10. ...]
