import string
import time
import numpy as np
import pyopencl as cl
import os

# AMD-specific settings
os.environ['PYOPENCL_CTX'] = '0'  # Automatically selects AMD GPU
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'  # Sample compiler output

# Define the character set
charset = string.ascii_lowercase + string.digits  + string.punctuation  # a-z, 0-9, and special characters
max_length = 6  # Maximum password length

# Initialize OpenCL for AMD
def init_opencl():
    platforms = cl.get_platforms()
    amd_platform = None
    for p in platforms:
        if 'AMD' in p.name:
            amd_platform = p
            break
    
    if not amd_platform:
        raise RuntimeError("No AMD platform found")
    
    devices = amd_platform.get_devices(device_type=cl.device_type.GPU)
    if not devices:
        raise RuntimeError("No AMD GPU devices found")
    
    # Use the first AMD GPU found
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    print(f"Selected device: {devices[0].name}")
    print(f"Computing units: {devices[0].max_compute_units}")
    print(f"Maximum work-group size: {devices[0].max_work_group_size}")
    
    return ctx, queue

ctx, queue = init_opencl()

# Function to execute brute force on the GPU
def execute_kernel(password, charset, max_length):
    # Prepare Data
    charset_array = np.array([ord(c) for c in charset], dtype=np.uint8)
    password_array = np.array([ord(c) for c in password], dtype=np.uint8)
    password_length = len(password)
    charset_size = len(charset)
    num_combinations = charset_size ** max_length

    # Create buffers
    mf = cl.mem_flags
    d_password = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=password_array)
    d_charset = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=charset_array)
    d_result = cl.Buffer(ctx, mf.WRITE_ONLY, size=4)  # Buffer for the result (int32)

    # Fixed kernel
    kernel_code = """
    __kernel void gpu_brute_force(
        __global const uchar* password,
        __global const uchar* charset,
        __global int* result,
        const int max_length,
        const int password_length,
        const int charset_size,
        const int num_combinations)
    {
        int idx = get_global_id(0);
        if (idx >= num_combinations) return;
        
        int temp_idx = idx;
        uchar candidate[5];  // Adjust this if max_length changes

        for (int i = 0; i < max_length; i++) {
            candidate[i] = charset[temp_idx % charset_size];
            temp_idx = temp_idx / charset_size;
        }

        int match = 1;
        for (int i = 0; i < password_length; i++) {
            if (candidate[i] != password[i]) {
                match = 0;
                break;
            }
        }

        if (match) {
            result[0] = 1;
            barrier(CLK_GLOBAL_MEM_FENCE);  // Ensure all threads complete
            return;  // Immediately stop further processing for this work item
        }
    }
    """
    
    try:
        # compile the kernel
        prg = cl.Program(ctx, kernel_code).build()
        
        # Calculate the best size for AMD RDNA3
        max_workgroup_size = ctx.devices[0].max_work_group_size
        local_size = min(256, max_workgroup_size)  # Size conservator
        global_size = ((num_combinations + local_size - 1) // local_size) * local_size
        
        # Execute the kernel with profiling
        event = prg.gpu_brute_force(
            queue, (global_size,), (local_size,),
            d_password, d_charset, d_result,
            np.int32(max_length), np.int32(password_length),
            np.int32(charset_size), np.int32(num_combinations)
        )
        
        # Wait for it to end
        event.wait()
        
        # Read the result
        result = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(queue, result, d_result).wait()
        
        return result
        
    except Exception as e:
        print(f"Error when executing the kernel: {str(e)}")
        return np.array([0], dtype=np.int32)

# Main Function
def main():
    password = 'Boro2:'  # Password for Test (must be lowercase letters and digits)
    print(f"Searching for password: {password}")
    print(f"Charset size: {len(charset)}")
    print(f"Total combinations: {len(charset)**max_length:,}")
    
    start_time = time.time()
    result = execute_kernel(password, charset, max_length)
    elapsed_time = time.time() - start_time
    
    if result[0] == 1:
        print(f"Password Found!: {password}")
    else:
        print("Password not found")
    
    print(f"Time of execution: {elapsed_time:.4f} seconds")
    print(f"Speed: {len(charset)**max_length/elapsed_time:,.0f} Combinations/Seconds")

if __name__ == "__main__":
    main()
