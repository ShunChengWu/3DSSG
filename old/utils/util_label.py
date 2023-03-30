try: import define
except: from utils import define
try: import util
except: from utils import util


def get_NYU40_color_palette():
    return [
        (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]



NYU40_Label_Names = [
    'wall',
'floor',
'cabinet',
'bed',
'chair',
'sofa',
'table',
'door',
'window',
'bookshelf',
'picture',
'counter',
'blinds',
'desk',
'shelves',
'curtain',
'dresser',
'pillow',
'mirror',
'floor mat',
'clothes',
'ceiling',
'books',
'refridgerator',
'television',
'paper',
'towel',
'shower curtain',
'box',
'whiteboard',
'person',
'night stand',
'toilet',
'sink',
'lamp',
'bathtub',
'bag',
'otherstructure',
'otherfurniture',
'otherprop',
]

def nyu40_name_to_id(name:str):
    return NYU40_Label_Names.index(name.lower())


SCANNET20_Label_Names = [
'wall',
'floor',
'cabinet',
'bed',
'chair',
'sofa',
'table',
'door',
'window',
'bookshelf',
'picture',
'counter',
'desk',
'curtain',
'refridgerator',
'shower curtain',
'toilet',
'sink',
"bathtub",
'otherfurniture',
]

def getLabelNames(path):
    import csv
    Scan3R=dict()
    NYU40=dict()
    Eigen13=dict()
    RIO27=dict()
    RIO7=dict()
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # print(', '.join(row))
            if not row[0].isnumeric():
                continue
            Scan3R[int(row[0])]=row[1]
            NYU40[int(row[2])] = row[3]
            Eigen13[int(row[4])] = row[5]
            RIO27[int(row[6])] = row[7]
            RIO7[int(row[8])] = row[9]
            # break
    return dict(sorted(Scan3R.items())),dict(sorted(NYU40.items())),\
           dict(sorted(Eigen13.items())),dict(sorted(RIO27.items())),dict(sorted(RIO7.items()))
def getLabelNameMapping(path):
    """
    return  toNameNYU40,toNameEigen,toNameRIO27,ttoNameRIO7
    """
    import csv
    raw=dict()
    toNameNYU40=dict()
    toNameEigen=dict()
    toNameRIO27=dict()
    toNameRIO7 =dict()
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if not row[0].isnumeric():
                continue
            raw[row[1]]=row[1]
            toNameNYU40[row[1]] = row[3] if row[3] != '-' else 'none'
            toNameEigen[row[1]] = row[5] if row[5] != '-' else 'none'
            toNameRIO27[row[1]] = row[7] if row[7] != '-' else 'none'
            toNameRIO7[row[1]] = row[9] if row[9] != '-' else 'none'
    return raw, toNameNYU40,toNameEigen,toNameRIO27,toNameRIO7
def getLabelIdxMapping(path):
    """
    return  toNYU40,toEigen,toRIO27,toRIO7
    """
    import csv
    raw=dict()
    toNYU40=dict()
    toEigen=dict()
    toRIO27=dict()
    toRIO7 =dict()
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # print(', '.join(row))
            if not row[0].isnumeric():
                continue
            raw[int(row[0])] = int(row[0])
            toNYU40[int(row[0])] = int(row[2])
            toEigen[int(row[0])] = int(row[4])
            toRIO27[int(row[0])] = int(row[6])
            toRIO7[int(row[0])] = int(row[8])
            # break
    return raw,toNYU40,toEigen,toRIO27,toRIO7

def getLabelMapping(label_type:str,pth_mapping:str = ""):
    if pth_mapping == "":
        pth_mapping = define.LABEL_MAPPING_FILE
        pth_160 = define.CLASS160_FILE
    Scan3R528, NYU40,Eigen13,RIO27,RIO7 = getLabelNames(pth_mapping)
    NameScan3R528, toNameNYU40,toNameEigen13,toNameRIO27,toNameRIO7=getLabelNameMapping(pth_mapping)
    IdxScan3R528, toNYU40,toEigen13,toRIO27,toRIO7 = getLabelIdxMapping(pth_mapping)
    label_names=None
    label_name_mapping=None
    label_id_mapping=None
    label_type=label_type.lower()
    if label_type == 'nyu40':
        label_names = NYU40
        label_name_mapping = toNameNYU40
        label_id_mapping = toNYU40
    elif label_type == 'eigen13':
        label_names = Eigen13
        label_name_mapping = toNameEigen13
        label_id_mapping = toEigen13
    elif label_type == 'rio27':
        label_names = RIO27
        label_name_mapping = toNameRIO27
        label_id_mapping = toRIO27
    elif label_type == 'rio7':
        label_names = RIO7
        label_name_mapping = toNameRIO7
        label_id_mapping = toRIO7
    elif label_type == '3rscan':
        label_names=Scan3R528
        label_name_mapping=NameScan3R528
        label_id_mapping=IdxScan3R528
    elif label_type == '3rscan160':
        names = sorted(util.read_classes(pth_160))
        label_names = {k:v for k,v in enumerate(names,1)}
        n_to_id = {v:k for k,v in enumerate(names,1)}
        label_name_mapping = dict()
        label_id_mapping = dict()
        for k,v in NameScan3R528.items():
            label_name_mapping[k] = v if v in names else 'none'
        for k,v in Scan3R528.items():
            label_id_mapping[k] = 0 if v not in names else n_to_id[v]
        
    elif label_type == 'scannet20':
        label_names = NYU40
        label_name_mapping = toNameNYU40
        label_id_mapping = toNYU40
        
        label_names = {i+1:SCANNET20_Label_Names[i] for i in range(len(SCANNET20_Label_Names)) }
        for name, name40 in label_name_mapping.items():
            label_name_mapping[name] = name40 if name40 in label_names.values() else 'none'
        for id_f, id_40 in label_id_mapping.items():
            nyu40name = NYU40_Label_Names[id_40-1]
            if nyu40name in label_names.values():
                id_scan20 = list(label_names.values()).index(nyu40name)+1
                label_id_mapping[id_f] = id_scan20
            else:
                label_id_mapping[id_f] = 0
    else:
        raise RuntimeError('')
    return label_names, label_name_mapping, label_id_mapping

if __name__ == '__main__':
    for name in SCANNET20_Label_Names:
        print(name in NYU40_Label_Names)