import os
import trimesh
import numpy as np
from codeLib.utils.util import check_file_exist
import ssg.define as define
from plyfile import PlyElement, PlyData


def read_labels(plydata):
    ply_raw = 'ply_raw' if 'ply_raw' in plydata.metadata else '_ply_raw'
    data = plydata.metadata[ply_raw]['vertex']['data']
    try:
        labels = data['objectId']
    except:
        labels = data['label']
    return labels


def save_trimesh_to_ply(trimeshply, pth, binary=False):
    ply_raw = 'ply_raw' if 'ply_raw' in trimeshply.metadata else '_ply_raw'
    dtypes = [(k, v) for k, v in trimeshply.metadata[ply_raw]
              ['vertex']['properties'].items()]
    tmp = [v.flatten() for k, v in trimeshply.metadata[ply_raw]
           ['vertex']['data'].items()]
    tmp = np.array(tmp)
    vertex = [tuple(tmp[:, v]) for v in range(tmp.shape[1])]
    vertex = np.array(vertex, dtype=dtypes)
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=not binary).write(pth)
    pass


def get_label(ply_in, dataset_type, label_type):
    ply_raw = 'ply_raw' if 'ply_raw' in ply_in.metadata else '_ply_raw'
    data = ply_in.metadata[ply_raw]['vertex']['data']
    if dataset_type == None or dataset_type == '' or label_type == None:
        try:
            labels = data['objectId']
        except:
            labels = data['label']
    elif dataset_type == 'ScanNet':
        labels = data['label']
    elif dataset_type == '3RScan':
        if label_type == 'Segment':
            if type(data) is dict:
                labels = data['objectId'] if 'objectId' in data else data['label']
            else:
                labels = data['objectId'] if 'objectId' in data.dtype.names else data['label']
        elif label_type == 'NYU40':
            labels = data['NYU40']
        elif label_type == 'Eigen13':
            labels = data['Eigen13']
        elif label_type == 'RIO27':
            labels = data['RIO27']
        else:
            raise RuntimeError('unsupported label type:', label_type)
    else:
        raise RuntimeError('unsupported dataset type:', dataset_type)
    return labels


def load_rgb(path, target_name=define.LABEL_FILE_NAME, with_worker=True):
    '''
    path: path to the folder contains config.OBJ_NAME, config.MTL_NAME, config.TEXTURE_NAME,
    config.LABEL_FILE_NAME and config.LABEL_FILE_NAME_RAW
    '''
    dirname = path
    pth_label = os.path.join(dirname, target_name)
    if path.find('scene') >= 0:  # ScanNet
        scan_id = os.path.basename(path)
        pth_obj = os.path.join(dirname, scan_id+'_vh_clean_2.ply')
        pth_label_raw = pth_label
        pass
    else:  # 3RScan
        pth_label_raw = os.path.join(dirname, define.LABEL_FILE_NAME_RAW)
        if not os.path.exists(os.path.join(dirname, 'color.align.ply')):
            pth_obj = os.path.join(dirname, define.OBJ_NAME)
            pth_mtl = os.path.join(dirname, define.MTL_NAME)
            pth_tex = os.path.join(dirname, define.TEXTURE_NAME)
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
        query_points = label_mesh.vertices
        if isinstance(mesh, trimesh.base.Trimesh):
            # mesh.refined.v2.obj should have exactly the same vertices as in labels.instances.annotated.v2.ply
            # if not np.isclose(mesh.vertices,query_points):
            #     raise RuntimeError('there is a problem with the input file.')
            import torch
            from knn_cuda import KNN

            knn = KNN(k=1, transpose_mode=True)
            tmp_q = torch.tensor(mesh.vertices).cuda().unsqueeze(0)
            tmp_t = torch.tensor(query_points).unsqueeze(0)

            indices = []
            for split in torch.split(tmp_t, int(64), dim=1):
                dist, idx = knn(tmp_q, split.cuda())
                # dist = dist > 0.1
                # if dist.any():
                #     raise RuntimeError('there is a problem with the input file.')
                idx = idx.cpu().squeeze()
                if len(idx.shape) == 0:
                    idx = idx.unsqueeze(00)
                indices.append(idx)

            # dist,idx = knn(tmp_q,tmp_t)
            # del tmp_q,tmp_t

            indices = torch.cat(indices, dim=0)
            # idx = idx.cpu().squeeze()
            # import open3d as o3d
            # o3d.core.Device
            # tree = o3d.geometry.KDTreeFlann(mesh.vertices.transpose())

            colors = trimesh.visual.uv_to_color(
                mesh.visual.uv, mesh.visual.material.image)
            colors = colors[indices]
        else:
            colors = mesh.visual.vertex_colors
        # colors = mesh.visual.vertex_colors
        ply_raw = 'ply_raw' if 'ply_raw' in label_mesh_align.metadata else '_ply_raw'

        if 'nx' not in label_mesh_align.metadata[ply_raw]['vertex']['data']:
            label_mesh_align.metadata[ply_raw]['vertex']['properties']['nx'] = '<f4'
            label_mesh_align.metadata[ply_raw]['vertex']['properties']['ny'] = '<f4'
            label_mesh_align.metadata[ply_raw]['vertex']['properties']['nz'] = '<f4'
            label_mesh_align.metadata[ply_raw]['vertex']['data']['nx'] = np.zeros(
                [len(query_points), 1], dtype=np.float32)
            label_mesh_align.metadata[ply_raw]['vertex']['data']['ny'] = np.zeros(
                [len(query_points), 1], dtype=np.float32)
            label_mesh_align.metadata[ply_raw]['vertex']['data']['nz'] = np.zeros(
                [len(query_points), 1], dtype=np.float32)

        if hasattr(label_mesh_align, 'vertex_normals'):
            label_mesh_align.metadata[ply_raw]['vertex']['data']['nx'] = label_mesh_align.vertex_normals[:, 0]
            label_mesh_align.metadata[ply_raw]['vertex']['data']['ny'] = label_mesh_align.vertex_normals[:, 1]
            label_mesh_align.metadata[ply_raw]['vertex']['data']['nz'] = label_mesh_align.vertex_normals[:, 2]

        label_mesh_align.metadata[ply_raw]['vertex']['data']['red'] = colors[:, 0]
        label_mesh_align.metadata[ply_raw]['vertex']['data']['green'] = colors[:, 1]
        label_mesh_align.metadata[ply_raw]['vertex']['data']['blue'] = colors[:, 2]

        # for i in range(len(query_points)):
        #     if isinstance(mesh, trimesh.base.Trimesh):
        #         point = query_points[i]
        #         if with_worker: raise RuntimeError('open3d doesn\'t work with nn.DataLoader workers')
        #         # [k, idx, distance] = tree.search_radius_vector_3d(point,0.001)
        #         [k, idx, distance] = tree.search_knn_vector_3d(point.transpose(),1)
        #     else:
        #         idx = [i]
        #     label_mesh_align.visual.vertex_colors[i] = colors[idx[0]]
        #     label_mesh_align.metadata[ply_raw]['vertex']['data']['red'][i] = colors[idx[0]][0]
        #     label_mesh_align.metadata[ply_raw]['vertex']['data']['green'][i] = colors[idx[0]][1]
        #     label_mesh_align.metadata[ply_raw]['vertex']['data']['blue'][i] = colors[idx[0]][2]

        #     # if hasattr(mesh, 'vertex_normals'):
        #     #     label_mesh_align.metadata[ply_raw]['vertex']['data']['nx'][i] = mesh.vertex_normals[idx[0]][0]
        #     #     label_mesh_align.metadata[ply_raw]['vertex']['data']['ny'][i] = mesh.vertex_normals[idx[0]][1]
        #     #     label_mesh_align.metadata[ply_raw]['vertex']['data']['nz'][i] = mesh.vertex_normals[idx[0]][2]
        del label_mesh
    else:
        # trimesh.base.Trimesh
        label_mesh_align.visual.vertex_colors = mesh.visual.vertex_colors
        label_mesh_align.metadata[ply_raw]['vertex']['data']['nx'] = mesh.vertex_normals[:, 0]
        label_mesh_align.metadata[ply_raw]['vertex']['data']['ny'] = mesh.vertex_normals[:, 1]
        label_mesh_align.metadata[ply_raw]['vertex']['data']['nz'] = mesh.vertex_normals[:, 2]
        label_mesh_align.metadata[ply_raw]['vertex']['data']['red'] = mesh.visual.vertex_colors[:, 0]
        label_mesh_align.metadata[ply_raw]['vertex']['data']['green'] = mesh.visual.vertex_colors[:, 1]
        label_mesh_align.metadata[ply_raw]['vertex']['data']['blue'] = mesh.visual.vertex_colors[:, 2]
    del mesh
    return label_mesh_align
