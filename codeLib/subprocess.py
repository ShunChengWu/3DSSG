import subprocess, os, sys

def execute(cmd,cwd):
    '''
    Executate something with realtime stdout catch
    '''
    shell = isinstance(cmd,str)
    popen = subprocess.Popen(cmd,cwd=cwd, stdout=subprocess.PIPE, universal_newlines=True,
                             shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        print(cmd)
        raise subprocess.CalledProcessError(return_code, cmd)

def run(bashCommand,cwd='./',verbose:bool=True):
    '''
    execute an exe with realtime stdout catch
    '''
    for path in execute(bashCommand,cwd):
        if verbose:
            print(path, end="")
        else:
            pass
        
def run_python(bashCommand,cwd="./", pythonpath:str="",verbose:bool=True):
    ''' 
    execute a python script with realtime stdout catch
    '''
    if len(pythonpath)>0: os.environ['PYTHONPATH'] = pythonpath
    bashCommand=[sys.executable]+bashCommand
    run(bashCommand,cwd,verbose)