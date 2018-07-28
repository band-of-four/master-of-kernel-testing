# Testing of framework
import numpy as np
import pyopencl as cl
import master as Master
import operations as operations
from test_data import TestData

@TestData(
	tf_checkpoint_dir='/mnt/f/benchmark/benchmarks-master/resnet50v1_traindir',
	tf_values={'add_inp1': 'tower_0/v/cg/resnet_v10/conv1/batchnorm1/FusedBatchNorm:0',
				'add_inp2': 'tower_0/v/cg/resnet_v10/conv4/batchnorm4/FusedBatchNorm:0',
				'add_res': 'tower_0/v/cg/resnet_v10/add:0'}
)
def get_data(test_data):
	return [test_data['add_inp1'], test_data['add_inp2'], test_data['add_res']]

framework = Master.MasterOfKernel()
a_np = get_data()[0].flatten()
b_np = get_data()[1].flatten()
perfect_res = get_data()[2].flatten()

# relu_np = np.random.rand(15).astype(np.float32) * 5 - 3
# perfect_res_relu = operations.ReLU(
#     cl.create_some_context()).relu_on_cpu(relu_np)[0]

print("ArrayAdd:")
framework.test(a_np, b_np, perfect_res, "sum")
# print("\nReLU:")
# framework.test(relu_np, None, perfect_res_relu, "relu")
