import sys
sys.path.append('.')

from mokt import TestData, test_kernel_from_file
from mokt.cli_runner import get_keys

keys = get_keys()
'''
keys: {'--chkp_dir', '--input', '--output', '--file'(optionally)}
'''


@TestData(
    tf_checkpoint_dir=keys.get('--chkp_dir'),
    tf_values={
        'input': keys.get('--input'),
        'output': keys.get('--output')
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
