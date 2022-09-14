import argparse, codeLib
import torch
import ssg.config as config

def process(cfg1,cfg2):
    db_1  = config.get_dataset(cfg1,'test')
    db_2  = config.get_dataset(cfg2,'test')
    
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config1')
    parser.add_argument('config2')
    args = parser.parse_args()
    
    cfg1 = codeLib.Config(args.config1)
    cfg2 = codeLib.Config(args.config2)
    # init device
    device = 'cuda' if torch.cuda.is_available() and len(cfg1.GPU) > 0 else 'cpu'
    cfg1.DEVICE=torch.device(device)
    cfg2.DEVICE=torch.device(device)
    
    process(cfg1,cfg2)