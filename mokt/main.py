# Testing of framework
import numpy as np
import pyopencl as cl
import Master as Master
import Operations as operations

framework = Master.MasterOfKernel()
a_np = np.random.rand(5000000).astype(np.float32)
b_np = np.random.rand(5000000).astype(np.float32)
perfect_res = []
for i in range(len(a_np)):
    perfect_res.append(a_np[i] + b_np[i])

relu_np = np.random.rand(15).astype(np.float32) * 5 - 3
perfect_res_relu = operations.ReLU(
    cl.create_some_context()).relu_on_cpu(relu_np)[0]

print("ArrayAdd:")
framework.test(a_np, b_np, perfect_res, "sum")
print("\nReLU:")
framework.test(relu_np, None, perfect_res_relu, "relu")
