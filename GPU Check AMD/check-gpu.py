import pyopencl as cl
import numpy as np

# Automatically configure context for AMD GPU
platform = cl.get_platforms()[0]  # AMD
devices = [d for d in platform.get_devices() if d.type == cl.device_type.GPU]
ctx = cl.Context(devices)
queue = cl.CommandQueue(ctx)

# Data and kernel
n = 1024
a = np.arange(n).astype(np.float32)
mf = cl.mem_flags
d_a = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

prg = cl.Program(ctx, """
__kernel void multiply_by_two(__global float *arr) {
    int idx = get_global_id(0);
    if (idx < %d) {
        arr[idx] *= 2;
    }
}
""" % n).build()

# Execute
prg.multiply_by_two(queue, (n,), None, d_a)
result = np.empty_like(a)
cl.enqueue_copy(queue, result, d_a)

print("Result:", result[:10])  # [0. 2. 4. 6. 8. ...]
print("GPU used:", devices[0].name)
