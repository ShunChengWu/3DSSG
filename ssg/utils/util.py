import json
import numpy as np


def gen_foo_input_data(n_nodes:int=10, n_kfs:int=3, img_width:int=640,img_height:int=480,
                       scan_id='4acaebcc-6c10-2a2a-858b-29c7e4fb410d'):
    '''
    node: dict.
        'center': float array[3]
        'dimension': float array[3]
        'kfs': dict.
            'idx': [x_min,y_min,x_max,y_max]
    kfs: dict.
        'idx': path
    
    Returns
    -------
    None.

    '''
    data = dict()    
    nodes = dict()
    kfs   = dict()
    
    # img_width, img_height = 640, 480
    # n_nodes = 5
    # n_kfs = 5
    for i in range(n_nodes):
        node = dict()
        node['center'] = np.random.uniform(0,1,[3]).tolist()
        node['dimension'] = np.random.uniform(0,1,[3]).tolist()
                
        node['kfs'] = dict()
        n_visible_kfs = np.random.choice(np.arange(0,n_kfs), 1)[0]+1
        kf_list = np.random.choice(np.arange(0,n_kfs), n_visible_kfs, replace=False).tolist()
        for kf in sorted(kf_list):
            xmm = np.random.randint(0,img_width, [2]).tolist()
            ymm = np.random.randint(0,img_height, [2]).tolist()
            node['kfs'][str(kf)] = [min(xmm), min(ymm), max(xmm), max(ymm)]
        
        nodes[str(i)] = node
    
    
    for i in range(n_kfs):
        # kf = dict()
        # n_visible_nodes = np.random.choice(np.arange(0,n_nodes), 1)[0]+1
        # node_list = np.random.choice(np.arange(0,n_nodes), n_visible_nodes, replace=False)
        # kf[i] = node_list
        kfs[str(i)] = str(i)
    
    
    data['nodes'] = nodes
    data['kfs'] = kfs
    data['scan_id'] = scan_id
    return data


def find_neighbors():
    pass


if __name__ == '__main__':
    import pprint
    pp = pprint.PrettyPrinter(indent=1)
    
    data = gen_foo_input_data()
    # pp.pprint(data)
    pth = 'foo_data_input.json'
    with open(pth, 'w') as f:
        json.dump(data,f,indent=2)
    data2 = load_data(pth)
    pp.pprint(data2)
    # print(data2)