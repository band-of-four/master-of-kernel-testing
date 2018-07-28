# Main framework code
import pyopencl as cl
import Operations as operations
import numpy as np
from time import time  # Import time tools


class MasterOfKernel:
    def __init__(self):
        self.f_ctx = cl.create_some_context()
        self.f_queue = cl.CommandQueue(
            self.f_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.sumOp = operations.ArraySum(self.f_ctx)
        self.ReLU = operations.ReLU(self.f_ctx)

    # function for checking accuracy
    def check_accuracy(self, arr_f, arr_s):
        deltas = []
        if np.array_equal(arr_f, arr_s):
            return [1.0, 0.0]
        for i in range(len(arr_f)):
            if arr_f[i] != arr_s[i]:
                deltas.append(
                    max(arr_f[i], arr_s[i]) - min(arr_f[i], arr_s[i]))
        return [
            (len(arr_f) - len(deltas)) / len(arr_f),
            np.sum(deltas) / len(deltas)
        ]

    def test(self, inp1, inp2, expected, operation):
        mf = cl.mem_flags
        f_a_g = cl.Buffer(
            self.f_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp1)
        f_res_g = cl.Buffer(self.f_ctx, mf.WRITE_ONLY, inp1.nbytes)

        if (inp2 is not None):
            f_b_g = cl.Buffer(
                self.f_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp2)

        if (operation == "sum"):
            f_event = self.sumOp.__call__(
                self.f_queue, inp1, f_a_g, f_b_g, f_res_g)
            cpu_stats = self.sumOp.sum_on_cpu(inp1, inp2)  # CPU
        elif (operation == "relu"):
            f_event = self.ReLU.__call__(self.f_queue, inp1, f_a_g, f_res_g)
            cpu_stats = self.ReLU.relu_on_cpu(inp1)  # CPU

        f_event.wait()
        elapsed = 1e-9 * (
            f_event.profile.end - f_event.profile.start
        )  # Calculate the time it took to execute the kernel
        mem_bw = (inp1.nbytes + inp2.nbytes) / (elapsed * 1024 * 1024 * 1024)
        print(
            "GPU Kernel Time: {0}s".format(elapsed) + ", " + str(mem_bw) +
            " Gb/s")  # Print the time it took to execute the kernel
        f_res_np = np.zeros_like(
            inp1
        )  # TODO: why 'inp1' and not 'expected'? Program crash with 'expected'
        cl.enqueue_copy(self.f_queue, f_res_np, f_res_g)
        accur = self.check_accuracy(expected, f_res_np)
        print(
            "Testing results:\nResult is %g percent accurate, delta = %f" %
            (accur[0] * 100, accur[1]))

        # CPU
        cpu_time = cpu_stats[2] - cpu_stats[1]
        mem_cpu = (inp1.nbytes + inp2.nbytes) / (cpu_time * 1024 * 1024 * 1024)
        print(
            "CPU Time: {0}s".format(cpu_time) + ", " + str(mem_cpu) + " Gb/s")
        cpu_accur = self.check_accuracy(expected, cpu_stats[0])
        print(
            "Testing results:\nResult is %g percent accurate, delta = %f" %
            (cpu_accur[0] * 100, cpu_accur[1]))
