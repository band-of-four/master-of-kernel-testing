import sys
sys.path.append('.')

from mokt import TestData, test_kernel_from_file

# command example:
# python3 examples/add.py --chkp_dir='/mnt/f/benchmark/benchmarks-master/resnet50v1_traindir'
# --first_input='tower_0/v/cg/resnet_v15/conv20/batchnorm20/FusedBatchNorm:0' 
# --second_input='tower_0/v/cg/resnet_v14/Relu:0' --output='tower_0/v/cg/resnet_v15/add:0'
@TestData(
    tf_checkpoint_dir=sys.argv[1].split('=', maxsplit = 1)[1],
    tf_values={
        'first_input': sys.argv[2].split('=', maxsplit = 1)[1],
        'second_input': sys.argv[3].split('=', maxsplit = 1)[1],
        'output': sys.argv[4].split('=', maxsplit = 1)[1]
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