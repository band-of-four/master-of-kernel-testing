# Testing of framework
import numpy as np
import pyopencl as cl
import master as Master
import operations as operations
from test_data import TestData
import progressbar

@TestData(
	tf_checkpoint_dir='/mnt/f/benchmark/benchmarks-master/resnet50v1_traindir',
	tf_values={'add_inp1': 'tower_0/v/cg/resnet_v10/conv1/batchnorm1/FusedBatchNorm:0',
				'add_inp2': 'tower_0/v/cg/resnet_v10/conv4/batchnorm4/FusedBatchNorm:0',
				'add_res': 'tower_0/v/cg/resnet_v10/add:0',
				'relu_res': 'tower_0/v/cg/resnet_v10/Relu:0'}
)
def get_data(test_data):
	return [test_data['add_inp1'], test_data['add_inp2'], test_data['add_res'], test_data['relu_res']]


def values_testing_add(tf_in1, tf_in2, tf_out):
	matches = 0
	for i in range(100):
		if((tf_in1[i] + tf_in2[i]) in tf_out):
			out_index = np.where(tf_out==(tf_in1[i] + tf_in2[i]))[0][0]
			index =  np.unravel_index(out_index, (8, 56, 56, 256))
			matches += 1
			print("in[" + str(i) + "] is out in " + str(index)) 
	return (len(tf_out) - matches)/(len(tf_out))

framework = Master.MasterOfKernel()
a_np = get_data()[0].flatten()
b_np = get_data()[1].flatten()
perfect_res = get_data()[2].flatten()
relu_res = get_data()[3].flatten()

print(values_testing_add(a_np, b_np, perfect_res))

# print("ArrayAdd:")
# framework.test(a_np, b_np, perfect_res, "sum")
# print("\nReLU:")
# framework.test(perfect_res, None, relu_res, "relu")
