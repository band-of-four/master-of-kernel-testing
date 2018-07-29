import sys
sys.path.append('.')

from mokt import TestData, test_kernel_from_file


@TestData(
    tf_checkpoint_dir='../../Documents/resnet50v1_traindir',
    tf_values={
        'input': 'v/tower_0/cg/resnet_v10/add:0',
        'output': 'v/tower_0/cg/resnet_v10/Relu:0'
    })
def test_relu(test_data):
    relu_in = test_data['input'].flatten()
    relu_out = test_data['output'].flatten()

    test_kernel_from_file(
        'examples/relu.cl',
        kernel_name='relu',
        inputs=[relu_in.flatten()],
        expected_outputs=[relu_out.flatten()],
        global_size=relu_in.shape,
        local_size=(1, ))


test_relu()
