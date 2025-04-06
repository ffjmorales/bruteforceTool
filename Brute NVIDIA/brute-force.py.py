import string
import itertools
import time
import numpy as np
from numba import cuda

# Definir el conjunto de caracteres (puedes modificarlo)
charset = string.ascii_lowercase + string.digits  # a-z, 0-9
max_length = 6  # Longitud máxima de la contraseña

# Kernel de CUDA para fuerza bruta
@cuda.jit
def gpu_brute_force(password, charset, result, max_length, num_combinations):
    idx = cuda.grid(1)
    if idx < num_combinations:
        # Generar combinación basada en el índice
        temp_idx = idx
        match = True
        
        # Verificar cada carácter de la contraseña
        for i in range(max_length):
            if i < len(password):
                char_idx = temp_idx % len(charset)
                if password[i] != charset[char_idx]:
                    match = False
                    break
                temp_idx = temp_idx // len(charset)
        
        # Si todos los caracteres coinciden
        if match:
            result[0] = 1  # Se encontró la contraseña

# Función para ejecutar la fuerza bruta en la GPU
def run_gpu_brute_force(password, charset, max_length):
    num_combinations = len(charset) ** max_length  # Total de combinaciones posibles
    result = np.zeros(1, dtype=np.int32)  # Variable para almacenar el resultado

    # Convertir charset y password a arrays numpy
    charset_array = np.array([ord(c) for c in charset], dtype=np.uint8)
    password_array = np.array([ord(c) for c in password], dtype=np.uint8)
    
    # Transferir datos a la GPU
    d_charset = cuda.to_device(charset_array)
    d_result = cuda.to_device(result)
    d_password = cuda.to_device(password_array)

    # Configurar la cantidad de bloques y hilos
    threads_per_block = 128
    blocks = (num_combinations + (threads_per_block - 1)) // threads_per_block

    # Llamar al kernel
    gpu_brute_force[blocks, threads_per_block](d_password, d_charset, d_result, max_length, num_combinations)

    # Transferir el resultado de vuelta a la CPU
    d_result.copy_to_host(result)
    
    # Mostrar el resultado
    if result[0] == 1:
        print(f"¡Contraseña encontrada: {password}")
    else:
        print("Contraseña no encontrada")

# Definir la contraseña a encontrar
password = 'me3j8ss'

# Ejecutar fuerza bruta en la GPU
start_time = time.time()
run_gpu_brute_force(password, charset, max_length)
end_time = time.time()

# Mostrar tiempo de ejecución
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")