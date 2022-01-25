if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../../')
import os
import ssg.define as define
import torch.utils.data as data
# from PIL import Image
# from torchvision import transforms
from torchvision.io import read_image
import ssg
import torch
class Sequence_Loader (data.Dataset):
    def __init__(self,
             scan_id, transform=None):
        super().__init__()
        info = ssg.read_3rscan_info(
            os.path.join(define.DATA_PATH,scan_id,define.IMG_FOLDER_NAME,define.INFO_NAME)
            )
        self.num_images = int(info['m_frames.size'])
        self.path = os.path.join(define.DATA_PATH,scan_id,define.IMG_FOLDER_NAME,define.RGB_NAME_FORMAT)
        self.transform = transform
        pass
    def __len__(self):
        return self.num_images
    def __getitem__(self,idx):
        img_path = self.path.format(idx)
        image = read_image(img_path)
        image = torch.rot90(image,1,[-1,-2])
        if self.transform:
            image = self.transform(image)
        return image
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    transform = None #RotateTransform(-90)
    scan_id = '4acaebcc-6c10-2a2a-858b-29c7e4fb410d'
    sequence = Sequence_Loader(scan_id,transform=transform)
    num_images = len(sequence)
    
    for i in range(6,num_images, 3):
        img = sequence.__getitem__(i)
        # pth = os.path.join(define.DATA_PATH,scan_id,define.IMG_FOLDER_NAME,define.RGB_NAME_FORMAT.format(i))
        # input_image = Image.open(pth).rotate(-90,expand=True)
        # input_image.show()
        plt.imshow(img.permute(1,2,0), cmap="gray")
        
        plt.show()
        break
       