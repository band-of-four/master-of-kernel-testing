import sys
sys.path.append('.')

from mokt import TestData, test_kernel_from_file


@TestData(
    tf_checkpoint_dir='/mnt/f/benchmark/benchmarks-master/resnet50v1_traindir',
    tf_values={
        'first_input': 'tower_0/v/cg/resnet_v15/conv20/batchnorm20/FusedBatchNorm:0',
        'second_input': 'tower_0/v/cg/resnet_v14/Relu:0',
        'output': 'tower_0/v/cg/resnet_v15/add:0'
    })
def test_add(test_data):
    add_first_in = test_data['first_input'].flatten()
    add_second_in = test_data['second_input'].flatten()
    add_out = test_data['output'].flatten()

    test_kernel_from_file(
        'examples/add.cl',
        kernel_name='add',
        inputs=[add_first_in.flatten(), add_second_in.flatten()],
        expected_outputs=[add_out.flatten()],
        global_size=add_first_in.shape,
        local_size=(1, ))


test_add()