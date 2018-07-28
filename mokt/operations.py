import numpy as np
import pyopencl as cl
from time import time


# possible operations representation way
class ArraySum:
    def __init__(self, sum_ctx):
        self.sum_prg = cl.Program(
            sum_ctx, """
        __kernel void sum(
            __global const float *a_g, __global const float *b_g, __global float *res_g)
        {
          int gid = get_global_id(0);
          res_g[gid] = a_g[gid] + b_g[gid];
        }
        """).build()

    def __call__(self, sum_queue, sum_a_np, sum_a_g, sum_b_g, sum_res_g):
        return self.sum_prg.sum(
            sum_queue, sum_a_np.shape, None, sum_a_g, sum_b_g, sum_res_g)

    # CPU
    def sum_on_cpu(self, sum_a_g, sum_b_g):
        res_cpu = np.empty_like(sum_a_g)
        cpu_start_time = time()  # Get the CPU start time
        for i in range(len(sum_a_g)):
            res_cpu[i] = sum_a_g[i] + sum_b_g[i]
        cpu_end_time = time()
        return [res_cpu, cpu_start_time, cpu_end_time]


class ReLU:
    def __init__(self, relu_ctx):
        self.relu_prg = cl.Program(
            relu_ctx, """
        __kernel void relu(
            __global const float *a_g, __global float *res_g)
        {
          int gid = get_global_id(0);
          if (a_g[gid] < 0)
              res_g[gid] = 0;
          else res_g[gid] = a_g[gid];
        }
        """).build()

    def __call__(self, relu_queue, relu_a_np, relu_a_g, relu_res_g):
        return self.relu_prg.relu(
            relu_queue, relu_a_np.shape, None, relu_a_g, relu_res_g)

    def relu_on_cpu(self, relu_a_g):
        res_cpu = np.empty_like(relu_a_g)
        cpu_start_time = time()  # Get the CPU start time
        for i in range(len(relu_a_g)):
            res_cpu[i] = relu_a_g[i] if (relu_a_g[i] > 0) else 0
        cpu_end_time = time()
        return [res_cpu, cpu_start_time, cpu_end_time]
