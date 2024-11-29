import multiprocessing as mp
import time
import numpy as np
from numba import cuda

@cuda.jit
def calculate_pi_integral_gpu(steps, step, results):
    idx = cuda.grid(1)
    if idx < steps:
        x = (idx + 0.5) * step
        results[idx] = 4.0 / (1.0 + x * x)

def calculate_pi_integral_cuda(steps):
    step = 1.0 / steps
    result = np.zeros(steps, dtype=np.float64)
    device = cuda.to_device(result)

    threads_per_block = 256
    blocks_per_grid = (steps + threads_per_block - 1) // threads_per_block

    start = time.time()
    calculate_pi_integral_gpu[blocks_per_grid, threads_per_block](steps, step, device)
    cuda.synchronize()
    end = time.time()

    result = device.copy_to_host()
    total = result.sum() * step
    print(f"CUDA Time: {end - start:.6f}s")
    return total

def calculate_pi_integral(steps):
    step = 1.0 / steps
    partial_sum = 0.0
    for i in range(steps):
        x = (i + 0.5) * step
        partial_sum += 4.0 / (1.0 + x * x)
    return partial_sum * step

def calculate_pi_integral_range(step, start, end):
    partial_sum = 0.0
    for i in range(start, end):
        x = (i + 0.5) * step
        partial_sum += 4.0 / (1.0 + x * x)
    return partial_sum

def calculate_pi_integral_parallel(steps, num_processes):
    step = 1.0 / steps
    range_size = steps // num_processes

    tasks = [
        (step, t * range_size, steps if t == num_processes - 1 else (t + 1) * range_size)
        for t in range(num_processes)
    ]

    with mp.Pool(processes=num_processes) as pool:
        partial_sums = pool.starmap(calculate_pi_integral_range, tasks)

    total_sum = sum(partial_sums)
    return total_sum * step

if __name__ == "__main__":
    iterations = 100000000
    num_processes = 6

    start_time = time.time()
    pi = calculate_pi_integral(iterations)
    end_time = time.time()

    single_elapsed_time = end_time - start_time
    print(f"Sequential: \t{pi} ({single_elapsed_time:.6f}s)")

    start_time = time.time()
    pi = calculate_pi_integral_parallel(iterations, num_processes)
    end_time = time.time()

    multi_elapsed_time = end_time - start_time
    print(f"Parallel: \t{pi} ({multi_elapsed_time:.6f}s)")
    print(f"Real PI: \t{3.141592653589793}")

    start_time = time.time()
    pi = calculate_pi_integral_cuda(iterations)
    end_time = time.time()

    cuda_elapsed_time = end_time - start_time
    print(f"Total CUDA: \t{pi} ({cuda_elapsed_time:.6f}s)")

    print(f"Speedup: {single_elapsed_time / multi_elapsed_time:.2f}")
    print(f"Speedup CUDA: {single_elapsed_time / cuda_elapsed_time:.2f}")
