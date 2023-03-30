import os, trimesh
import numpy as np
try: from util import check_file_exist
except: from utils.util import check_file_exist
try: import define
except: from utils import define

def read_labels(plydata):
    data = plydata.metadata['ply_raw']['vertex']['data']
    try:
        labels = data['objectId']
    except:
        labels = data['label']
    return labels


def get_label(ply_in, dataset_type, label_type):
    data = ply_in.metadata['ply_raw']['vertex']['data']
    if dataset_type == None or dataset_type == '' or label_type == None:
        try:
            labels = data['objectId']
        except:
            labels = data['label']
    elif dataset_type == 'ScanNet':        
        labels = data['label']
    elif dataset_type == '3RScan':
        if label_type == 'Segment':
            labels = data['objectId']
        elif label_type == 'NYU40':
            labels = data['NYU40']
        elif label_type == 'Eigen13':
            labels = data['Eigen13']             
        elif label_type == 'RIO27':
            labels = data['RIO27']
        else:
            raise RuntimeError('unsupported label type:',label_type)
    else:
        raise RuntimeError('unsupported dataset type:',dataset_type)
    return labels

def load_rgb(path, target_name = define.LABEL_FILE_NAME):
    '''
    path: path to the folder contains config.OBJ_NAME, config.MTL_NAME, config.TEXTURE_NAME,
    config.LABEL_FILE_NAME and config.LABEL_FILE_NAME_RAW
    '''
    dirname = path
    pth_label = os.path.join(dirname,target_name)
    if path.find('scene') >=0:    
        scan_id = os.path.basename(path)
        pth_obj = os.path.join(dirname,scan_id+'_vh_clean_2.ply')
        pth_label_raw = pth_label
        pass
    else:
        pth_label_raw = os.path.join(dirname,define.LABEL_FILE_NAME_RAW)
        if not os.path.exists(os.path.join(dirname, 'color.align.ply')):
            pth_obj = os.path.join(dirname,define.OBJ_NAME)
            pth_mtl = os.path.join(dirname,define.MTL_NAME)
            pth_tex = os.path.join(dirname,define.TEXTURE_NAME)
            check_file_exist(pth_mtl)
            check_file_exist(pth_tex)
        else:
            pth_obj = os.path.join(dirname, 'color.align.ply')
    
    # check file exist
    check_file_exist(pth_obj)
    check_file_exist(pth_label_raw)
    check_file_exist(pth_label)
    
    # retrieve rgb from obj.
    mesh = trimesh.load(pth_obj, process=False)
    
    label_mesh_align = trimesh.load(pth_label, process=False)
    if pth_label != pth_label_raw:
        label_mesh = trimesh.load(pth_label_raw, process=False)
        import open3d as o3d
        query_points = label_mesh.vertices        
        if isinstance(mesh, trimesh.base.Trimesh):
            tree = o3d.geometry.KDTreeFlann(mesh.vertices.transpose())
            colors = trimesh.visual.uv_to_color(mesh.visual.uv,mesh.visual.material.image)
        else:
            colors = mesh.visual.vertex_colors
        
        if 'nx' not in label_mesh_align.metadata['ply_raw']['vertex']['data']:
            label_mesh_align.metadata['ply_raw']['vertex']['properties']['nx'] = '<f4'
            label_mesh_align.metadata['ply_raw']['vertex']['properties']['ny'] = '<f4'
            label_mesh_align.metadata['ply_raw']['vertex']['properties']['nz'] = '<f4'
            label_mesh_align.metadata['ply_raw']['vertex']['data']['nx'] = np.zeros([len(query_points),1],dtype=np.float32)
            label_mesh_align.metadata['ply_raw']['vertex']['data']['ny'] = np.zeros([len(query_points),1],dtype=np.float32)
            label_mesh_align.metadata['ply_raw']['vertex']['data']['nz'] = np.zeros([len(query_points),1],dtype=np.float32)
        
        for i in range(len(query_points)):
            if isinstance(mesh, trimesh.base.Trimesh):
                point = query_points[i]
                [k, idx, distance] = tree.search_radius_vector_3d(point,0.001)
            else:
                idx = [i]
            label_mesh_align.visual.vertex_colors[i] = colors[idx[0]]
            label_mesh_align.metadata['ply_raw']['vertex']['data']['red'][i] = colors[idx[0]][0]
            label_mesh_align.metadata['ply_raw']['vertex']['data']['green'][i] = colors[idx[0]][1]
            label_mesh_align.metadata['ply_raw']['vertex']['data']['blue'][i] = colors[idx[0]][2]
            if hasattr(mesh, 'vertex_normals'):
                label_mesh_align.metadata['ply_raw']['vertex']['data']['nx'][i] = mesh.vertex_normals[idx[0]][0]
                label_mesh_align.metadata['ply_raw']['vertex']['data']['ny'][i] = mesh.vertex_normals[idx[0]][1]
                label_mesh_align.metadata['ply_raw']['vertex']['data']['nz'][i] = mesh.vertex_normals[idx[0]][2]
    else:
        #trimesh.base.Trimesh
        label_mesh_align.visual.vertex_colors = mesh.visual.vertex_colors
        label_mesh_align.metadata['ply_raw']['vertex']['data']['nx'] = mesh.vertex_normals[:,0]
        label_mesh_align.metadata['ply_raw']['vertex']['data']['ny'] = mesh.vertex_normals[:,1]
        label_mesh_align.metadata['ply_raw']['vertex']['data']['nz'] = mesh.vertex_normals[:,2]
        label_mesh_align.metadata['ply_raw']['vertex']['data']['red']   = mesh.visual.vertex_colors[:,0]
        label_mesh_align.metadata['ply_raw']['vertex']['data']['green'] = mesh.visual.vertex_colors[:,1]
        label_mesh_align.metadata['ply_raw']['vertex']['data']['blue']  = mesh.visual.vertex_colors[:,2]
    return label_mesh_align