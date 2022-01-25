import json, glob, csv, sys,os, argparse
import numpy as np
import trimesh
from util_scannet import load_scannet
import multiprocessing as mp

def get_intrinsic_color(fn):
    data = np.loadtxt(fn)
    mat = data
    wi = 1296
    hi = 968
    # for line in open(fn):
    #   fds = line.split(' ')
    #   #if fds[0]=='m_calibrationColorIntrinsic':
    #     #mat=np.asarray([ float(x) for x in fds[2:18]])
    #     #mat = np.reshape(mat,(4,4))
    #   if fds[0] == 'fx_color':
    #     fxc = fds[2]
    #   if fds[0] == 'fy_color':
    #     fyc = fds[2]
    #   if fds[0] == 'mx_color':
    #     mxc = fds[2]
    #   if fds[0] == 'my_color':
    #     myc = fds[2]
    #     mat = np.asarray([float(fxc), 0, float(mxc), 0, 0, float(fyc), float(myc), 0, 0, 0, 1, 0, 0, 0, 0, 1 ])    
    #     mat = np.reshape(mat,(4,4))
    	
    #   #if fds[0]=='m_colorWidth':
    #   if fds[0]=='colorWidth':
    #     wi=int(fds[2])
    #   #if fds[0]=='m_colorHeight':
    #   if fds[0]=='colorHeight':
    #     hi=int(fds[2])
    return (mat,wi,hi)

def get_pose(fn):
    return np.loadtxt(open(fn, "rb"), delimiter=" ")

def get_full_pc(fn):
  #return np.loadtxt(open(fn, "rb"), delimiter=" ")
  return np.genfromtxt(open(fn, "rb"), delimiter=" ")

def frame_num_from_name(filename):
  return int(filename.split('/')[-1].split('.')[0])# when name is 'scene0000_00/pose/1.txt'
  # return int(filename.split('/')[-1].split('-')[1].split('.')[0]) # when name is scene0000_00/frame-000000.pose.txt

def getOcclusion(camref, behind, full_pc2d, pc3di, bbx):
    (oid,l,x1,y1,x2,y2)=bbx
    inter=5 # resolution of the grid
    d=0.1# distance beyond we consider that a point does not belong anymore to an object
    xs=list(range(int(x1),int(x2),int((x2-x1)/inter)))
    ys=list(range(int(y1),int(y2),int((y2-y1)/inter)))
    # occlus_count=0
    obj_idx=np.zeros((1, len(behind)), dtype=bool)
    obj_idx[[0],[pc3di]]=True
    oclus=0
    outsideobject=0
    for i in range(len(xs)-1):
      for j in range(len(ys)-1):
        all_idx=np.logical_and( xs[i]<full_pc2d[0] , full_pc2d[0]<xs[i+1])
        all_idx=np.logical_and(all_idx, ys[j]<full_pc2d[1])
        all_idx=np.logical_and( all_idx, full_pc2d[1]<ys[j+1])
        all_idx=np.logical_and(all_idx, np.logical_not(behind) ) # Collecting point cloud inside this rectangle and not begind the camera
        o_idx=np.logical_and( all_idx, obj_idx) # selecting the 2D points which are inside the rectangle and belong to the object
        all_idx=np.logical_and(all_idx,np.logical_not(obj_idx))# removing the points linked to the object
        if not o_idx.any() or not all_idx.any():
          if not o_idx.any():
            outsideobject+=1 # this part of the bounding box does not contain object points, so we don't count it
          continue # no object points for this rectangle, or no point who does not belong to the object
        o_depth=np.min(np.linalg.norm(camref[0:3,o_idx[0,:]],axis=0))/3 #mean depth of the object for this rectangle
        if sum((o_depth-np.linalg.norm(camref[0:3,all_idx[0,:]],axis=0)/3)>d)>0:# if any point not from the object is in front (i.e. has a lower depth) of the object and make an occlusion ...
          oclus+=1 # ... we count this part as occluded
    if ((len(xs)-1)*(len(ys)-1)-outsideobject)==0:
      print('something is wrong: no points projected on the bbx')
      return 1
    return float(oclus)/((len(xs)-1)*(len(ys)-1)-outsideobject)

# main code
"""
usage so far : python makebb_over.py
it will produce a txt file containing for each line: frame_name object_id object_label occlustion_rate x1 y1 x2 y2
The last 4 coordinates are a bounding box enclosing the object
"""

parser = argparse.ArgumentParser(description="for each scene, create a .2dgt file which  containing for each line: frame_name object_id object_label occlustion_rate x1 y1 x2 y2.  The last 4 coordinates are a bounding box enclosing the object",
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--datadir', default="/media/sc/space1/dataset/scannet/scans/")
parser.add_argument('-l','--scene_list',default="/media/sc/space1/dataset/scannet/Tasks/Benchmark/scannetv2_val.txt", help="file containing on each line the id of a scene, it should also correspond to the name of its directory")
parser.add_argument('-o','--outdir',default='/media/sc/space1/dataset/scannet_detections/2dgt/')
parser.add_argument('--thread', type=int, default=4, help='The number of threads to be used.')
args = parser.parse_args()

datadir=args.datadir
scene_list=args.scene_list
outdir=args.outdir

def process(scene):
    '''check all files exist'''
    if os.path.isfile(os.path.join(outdir,scene+'.2dgt')):
      print('file exists, skipping',scene)
      return
    print('processing scene',scene)
    ag_f=datadir+'/'+scene+'/'+scene+'_vh_clean.aggregation.json'
    if not os.path.isfile(ag_f):
      print('no aggregation.json found for scene',ag_f)
      return
    with open(ag_f) as fp:
        aggreg=json.load(fp)
    objs={}#will contain the objects for this scene
    segs_f=datadir+'/'+scene+'/'+scene+'_vh_clean_2.0.010000.segs.json'
    if not os.path.isfile(segs_f):
      print('no segs.json found for scene',segs_f)
      return
    # with open(segs_f) as fp:
    #     segs=json.load(fp)
        
    xyz_f=datadir+'/'+scene+'/'+scene+'_vh_clean_2.ply'		
    if not os.path.isfile(xyz_f):
      print('no ply found for scene',xyz_f)
      return
  
    label_f=datadir+'/'+scene+'/'+scene+'_vh_clean_2.labels.ply'
    if not os.path.isfile(label_f):
      print('no label file found for scene',label_f)
      return
  
    label_map = dict()
    for segGroup in aggreg['segGroups']:
        label_map[segGroup['id']] = segGroup['label']
        
    #pth_ply, pth_agg, pth_seg, verbose=False, random_color = False):
    plydata, points, labels, instances = load_scannet(label_f, ag_f,segs_f,verbose=False)
        
    # for o in aggreg['segGroups']: # for each object in the json file
    #   objs[o['objectId']]=[ o['label']]#
    #   pc3di=[]
    #   for segid in o['segments']:#for each segment of the object o
    #     pc3di+=[x for (x,y) in enumerate(segs['segIndices']) if y==segid ] # collecting the 3D points associated to the segment segid
    #   objs[o['objectId']].append(pc3di)
    seg_ids = np.unique(instances)
    for segid in seg_ids:
        indices = np.where(instances==segid)[0]
        objs[segid] = [label_map[segid]]
        objs[segid].append(indices)
    # for segid in instances:
    #     objs[segid]
        
    
    # xyz_f = trimesh.load(xyz_f, process=False)
    # pc3d=get_full_pc(xyz_f)#getting the full 3D point cloud
    pc3d = points
    # print('Loading 3D point cloud done, number of objects:', len(list(objs.keys())))
    #(m_calibrationColorIntrinsic,wi,hi)=get_intrinsic_color(datadir+'/'+scene+'/_info.txt')#getting intrinsic parameter
    (m_calibrationColorIntrinsic,wi,hi)=get_intrinsic_color(os.path.join(datadir,scene,'intrinsic','intrinsic_color.txt'))#getting intrinsic parameter
    obj_by_img={}# dictionnary that will contains the bounding boxes of the objects appearing in each image.
    for pose in glob.glob(os.path.join(datadir,scene,'pose','*.txt')):
      cam2world=get_pose(pose)# getting the inverse of the extrinsic parameters from the frame_0XXXX.pose.txt
      if np.logical_not(np.isfinite(cam2world)).any() or  np.isnan(cam2world).any() or cam2world.shape[0]==0:
        # print('erroneous camera value, skipping', cam2world)
        continue #the values of the camera pose are wrong, so we skip
      world2cam=np.linalg.inv(cam2world)#getting the actual parameters
      obj_by_img[frame_num_from_name(pose)]=[pose.split('/')[-1].split('.')[0],[]]# will contains a list with two elements the name of the  frame and the list of objects bbxs 
      camref=np.dot(world2cam,np.vstack((pc3d.transpose(),np.ones((1,pc3d.shape[0])))))
      behind=camref[2]<=0 # boolean array which is true for the points which are behind the camera
      full_pc2d=np.dot(m_calibrationColorIntrinsic, camref)
      full_pc2d=np.divide(full_pc2d,np.tile(full_pc2d[2],(4,1)))#normalising the homogeneous point: [x y 1]
      for (oid,(l,pc3di)) in list(objs.items()):
        if behind[pc3di].any():# if any of the object point is behind the camera, we skip
          continue
        rows=np.array( [len(pc3di)*[0], len(pc3di)*[1]] )
        cols=np.array([pc3di,pc3di])
        pc2d=full_pc2d[rows,cols]   # get the points related to the objects, from the indices
        (x1,y1,x2,y2)=(min(pc2d[0]),min(pc2d[1]),max(pc2d[0]),max(pc2d[1])) # getting the bounding box coordinate
        if x1<0 or y1<0 or wi<x2 or hi<y2 or int(x2-x1)<5 or int(y2-y1)<5: # I don't keep it if the bounding box is outside the image or if the object is small on the image
          continue
        o=getOcclusion(camref, behind, full_pc2d, pc3di, (oid,l,x1,y1,x2,y2))  
        obj_by_img[frame_num_from_name(pose)][1].append([oid,l,o,x1,y1,x2,y2]) #note here I am saving only the bounding box coordinate. You might want to add the full 2D point cloud
        
    # writting the results
    with open(outdir+'/'+scene+'.2dgt','w') as fp:
        fp.write('frame_id object_id label occlution_ratio x_min y_min x_max y_max\n')
        for (fnum,v) in list(obj_by_img.items()):
          if len(v)==1:
            continue
          fn=v[0]
          for detection in v[1]:
            detection[1]=detection[1].encode('utf8')
            fp.write(' '.join([fn]+[str(x).replace(' ','_') for x in detection ])+'\n')

pool = mp.Pool(args.thread)
pool.daemon = True
results=[]
# count=0
for scene in open(scene_list): #['scene0000_00']: #glob.glob('scene*'):
    scene=scene.replace('\n','')
    
    results.append( pool.apply_async(process,args=[(scene)] ) )
    # if count>4:
    #     break
    # count+=1

if args.thread > 1:
    pool.close()
    results = [r.get() for r in results]
    pool.join()
    print('done')
    