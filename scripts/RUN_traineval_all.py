from codeLib.subprocess import run, run_python
import os
import sys
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger_py = logging.getLogger(__name__)

py_exe = 'main.py'
config_pattern = 'configs/config_{}_{}_{}.yaml'
method_types = ['IMP', 'VGfM', '3DSSG', 'SGFN', 'JointSSG']
input_types = ['full', 'inseg', 'orbslam']
label_types = ['l160', 'l20']

# For debug
input_types = ['full', 'inseg']
label_types = ['l20']

for method_type in method_types:
    for input_type in input_types:
        for label_type in label_types:
            cmd = [
                py_exe,
                '-c',
                config_pattern.format(method_type, input_type, label_type),
                '--cache'
            ]
            # Train
            run_python(cmd+['-m', 'train'])

            if method_type == "IMP":
                # do it again because the pre-compute feature step has a bug that requires rerun (at the first time).
                run_python(cmd+['-m', 'train'])

            # Eval
            run_python(cmd+['-m', 'eval'])
