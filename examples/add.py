import sys
sys.path.append('.')

from mokt import TestData, test_kernel_from_file


keys = {}
for a in sys.argv:
    temp = a.split('=', maxsplit = 1)
    if (len(temp) > 1):
        keys[temp[0]] = temp[1]

if(keys.get('--file') is not None):
    file = open(keys['--file'], 'r')
    for line in file:
        line = line.rstrip() 
        temp = line.split('=', maxsplit = 1)
        keys[temp[0]] = temp[1]

''' command example:
 python3 examples/add.py --chkp_dir='/mnt/f/benchmark/benchmarks-master/resnet50v1_traindir'
 --first_input='tower_0/v/cg/resnet_v15/conv20/batchnorm20/FusedBatchNorm:0' 
 --second_input='tower_0/v/cg/resnet_v14/Relu:0' --output='tower_0/v/cg/resnet_v15/add:0'

 or python3 examples/add.py --file=examples/add_properties.txt '''
 
@TestData(
    tf_checkpoint_dir=keys.get('--chkp_dir'),
    tf_values={
        'first_input': keys.get('--first_input'),
        'second_input': keys.get('--second_input'),
        'output': keys.get('--output')
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