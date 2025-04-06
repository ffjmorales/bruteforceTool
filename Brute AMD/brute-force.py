import string
import time
import numpy as np
import pyopencl as cl
import os

# Configuration for AMD GPU
os.environ['PYOPENCL_CTX'] = '0'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# Full character set
charset = string.ascii_letters + string.digits + string.punctuation
max_length = 5  # Maximum password length to try

def init_opencl():
    # Select AMD platform
    platforms = cl.get_platforms()
    amd_platform = next((p for p in platforms if 'AMD' in p.name), None)
    
    if not amd_platform:
        raise RuntimeError("No AMD platform found")
    
    # Select AMD GPU device
    devices = amd_platform.get_devices(device_type=cl.device_type.GPU)
    if not devices:
        raise RuntimeError("No AMD GPUs found")
    
    # Create context and command queue
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    # Get device limits
    max_work_group_size = devices[0].max_work_group_size
    max_work_item_sizes = devices[0].max_work_item_sizes
    
    print(f"\nSelected device: {devices[0].name}")
    print(f"Compute units: {devices[0].max_compute_units}")
    print(f"Maximum work group size: {max_work_group_size}")
    print(f"Maximum work item sizes per dimension: {max_work_item_sizes}\n")
    
    return ctx, queue, max_work_group_size

ctx, queue, max_wg_size = init_opencl()

def execute_kernel(password, charset, max_length, max_wg_size):
    # Prepare data for the device
    password = password.ljust(max_length, '\0')  # Pad with null characters
    charset_array = np.array([ord(c) for c in charset], dtype=np.uint8)
    password_array = np.array([ord(c) for c in password], dtype=np.uint8)
    password_length = len(password.strip('\0'))  # Actual length
    charset_size = np.int32(len(charset))
    total_combinations = np.uint64(len(charset)**max_length)
    
    # Limit the maximum work size to avoid overflow
    MAX_GLOBAL_SIZE = 2**28  # 268 million work items
    actual_global_size = min(total_combinations, MAX_GLOBAL_SIZE)
    
    # Create buffers on the device
    mf = cl.mem_flags
    d_password = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=password_array)
    d_charset = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=charset_array)
    d_result = cl.Buffer(ctx, mf.WRITE_ONLY, size=4)  # For the result (1=found)

    # Optimized OpenCL kernel
    kernel_code = """
    __kernel void brute_force(
        __global const uchar* password,
        __global const uchar* charset,
        __global volatile int* result,
        const int max_length,
        const int password_length,
        const int charset_size,
        const ulong total_combinations,
        const ulong start_index)
    {
        ulong idx = get_global_id(0) + start_index;
        if (idx >= total_combinations || *result) return;
        
        ulong temp_idx = idx;
        uchar candidate[5];  // Adjust if max_length changes

        // Generate candidate
        for (int i = 0; i < max_length; i++) {
            candidate[i] = charset[temp_idx % charset_size];
            temp_idx = temp_idx / charset_size;
        }

        // Check for match
        bool match = true;
        for (int i = 0; i < password_length; i++) {
            if (candidate[i] != password[i]) {
                match = false;
                break;
            }
        }

        if (match) {
            atomic_xchg(result, 1);  // Atomic operation
        }
    }
    """
    
    try:
        # Compile the kernel
        prg = cl.Program(ctx, kernel_code).build()
        
        # Configure work sizes
        local_size = min(256, max_wg_size)
        global_size = ((actual_global_size + local_size - 1) // local_size) * local_size
        
        # Execute the kernel in blocks if necessary
        result = np.zeros(1, dtype=np.int32)
        blocks = (total_combinations + MAX_GLOBAL_SIZE - 1) // MAX_GLOBAL_SIZE
        
        for block in range(blocks):
            start_idx = block * MAX_GLOBAL_SIZE
            current_size = min(MAX_GLOBAL_SIZE, total_combinations - start_idx)
            global_size = ((current_size + local_size - 1) // local_size) * local_size
            
            if global_size == 0:
                break
                
            event = prg.brute_force(
                queue, (global_size,), (local_size,),
                d_password, d_charset, d_result,
                np.int32(max_length), np.int32(password_length),
                charset_size, total_combinations, np.uint64(start_idx)
            )
            event.wait()
            
            # Check if found
            cl.enqueue_copy(queue, result, d_result).wait()
            if result[0] == 1:
                break
        
        return result[0] == 1
        
    except Exception as e:
        print(f"\nError in the kernel:\n{str(e)}\n")
        return False

def main():
    password = "Hello"  # Change this to test
    print(f"Searching for password: {password}")
    print(f"Charset size: {len(charset)}")
    print(f"Total combinations: {len(charset)**max_length:,}")
    
    start_time = time.time()
    found = execute_kernel(password, charset, max_length, max_wg_size)
    elapsed_time = time.time() - start_time
    
    print(f"\nResult: {'PASSWORD FOUND' if found else 'not found'}")
    print(f"Execution time: {elapsed_time:.4f} seconds")
    print(f"Speed: {len(charset)**max_length/elapsed_time:,.0f} combinations/second")

if __name__ == "__main__":
    main()
