import string
import itertools
import time
import numpy as np
from numba import cuda

# Define the character set (you can modify it)
charset = string.ascii_letters + string.digits + string.punctuation  # a-z, 0-9, special characters
max_length = 6  # Maximum password length

# CUDA kernel for brute-force attack
@cuda.jit
def gpu_brute_force(password, charset, result, max_length, num_combinations):
    idx = cuda.grid(1)
    if idx < num_combinations:
        temp_idx = idx
        match = True
        
        # Check each character of the password
        for i in range(max_length):
            if i < len(password):
                char_idx = temp_idx % len(charset)
                if password[i] != charset[char_idx]:
                    match = False
                    break
                temp_idx = temp_idx // len(charset)
        
        # If all characters match
        if match:
            result[0] = 1  # Password found

# Function to run brute-force on the GPU
def run_gpu_brute_force(password, charset, max_length):
    num_combinations = len(charset) ** max_length  # Total number of possible combinations
    result = np.zeros(1, dtype=np.int32)  # Variable to store the result

    # Convert charset and password to numpy arrays
    charset_array = np.array([ord(c) for c in charset], dtype=np.uint8)
    password_array = np.array([ord(c) for c in password], dtype=np.uint8)
    
    # Transfer data to the GPU
    d_charset = cuda.to_device(charset_array)
    d_result = cuda.to_device(result)
    d_password = cuda.to_device(password_array)

    # Configure the number of blocks and threads
    threads_per_block = 128
    blocks = (num_combinations + (threads_per_block - 1)) // threads_per_block

    # Launch the kernel
    gpu_brute_force[blocks, threads_per_block](d_password, d_charset, d_result, max_length, num_combinations)

    # Copy result back to the CPU
    d_result.copy_to_host(result)
    
    # Display the result
    if result[0] == 1:
        print(f"Password found: {password}")
    else:
        print("Password not found")

# Define the password to be found
password = 'me3j-s'

# Run brute-force on the GPU
start_time = time.time()
run_gpu_brute_force(password, charset, max_length)
end_time = time.time()

# Display execution time
print(f"Execution time: {end_time - start_time:.2f} seconds")
