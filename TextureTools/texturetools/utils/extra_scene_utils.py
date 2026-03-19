'''
Copied from https://github.com/fangchuan/cat3d, process_data.py, process_koolai_single_scene
NOTE: This script is just for consistency checking, not as a formal part of texturetools.
'''
# this script is used to process the downloaded data from koolai
import cv2
import numpy as np


class Perspective2Panorama:
    """convert perspective image to panorama image
    """
    def __init__(
        self, image:np.array, FOV:float, PHI:float, THETA:float, channel=3,
        interpolation=cv2.INTER_NEAREST
    ):
        self._img = image
        [self._height, self._width, c] = self._img.shape
        self.wFOV = FOV
        self.PHI = PHI
        self.THETA = THETA
        self.hFOV = float(self._height) / self._width * FOV

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))
        self.channel = channel
        self.interpolation = interpolation
        assert self.channel == c

    def GetEquirec(self, height, width):
        #
        # PHI is left/right angle, THETA is up/down angle, both in degree
        #
        x,y = np.meshgrid(np.linspace(-180, 180,width),np.linspace(90,-90,height))
        
        x_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map,y_map,z_map),axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.PHI))
        [R2, _] = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(self.THETA))  # +y axis forward

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height , width, 3])
        inverse_mask = np.where(xyz[:,:,1]>0, 1, 0) # +y axis forward
        whatsit = np.repeat(xyz[:,:,1][:, :, np.newaxis], 3, axis=2) # +y axis forward
        xyz[:,:] = xyz[:,:]/whatsit
        
        # +y axis forward
        lon_map = np.where(
            (-self.w_len < xyz[:,:,0]) & 
            (xyz[:,:,0] < self.w_len) & 
            (-self.h_len < xyz[:,:,2]) & 
            (xyz[:,:,2] < self.h_len),
            (xyz[:,:,0]+self.w_len)/2/self.w_len*self._width, 
            0
        )
        lat_map = np.where(
            (-self.w_len<xyz[:,:,0]) & (xyz[:,:,0] < self.w_len) &
            (-self.h_len<xyz[:,:,2]) & (xyz[:,:,2] < self.h_len),
            (-xyz[:,:,2]+self.h_len)/2/self.h_len*self._height, 
            0
        )
        mask = np.where(
            (-self.w_len < xyz[:,:,0]) & (xyz[:,:,0] < self.w_len) &
            (-self.h_len < xyz[:,:,2]) & (xyz[:,:,2] < self.h_len), 
            1, 
            0
        )

        # INTER_NEAREST needed to avoid interpolation for depth, semantic, and instance map
        # otherwise it will average nearby pixels
        persp = cv2.remap(self._img, lon_map.astype(np.float32), lat_map.astype(np.float32), self.interpolation, borderMode=cv2.BORDER_REPLICATE)
        
        mask = mask * inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], self.channel, axis=2)
        while len(persp.shape) != len(mask.shape):
            persp = persp[..., np.newaxis]
        persp = persp * mask
        
        
        return persp , mask
        
class MultiPers2Panorama:
    def __init__(
        self, img_array , F_P_T_array, channel=3,
        interpolation=cv2.INTER_NEAREST, average=True
    ):
        
        assert len(img_array)==len(F_P_T_array)
        
        self.img_array = img_array
        self.F_P_T_array = F_P_T_array
        self.channel = channel
        self.interpolation = interpolation
        self.average = average

    def GetEquirec(self, height:int=512, width:int=1024):
        #
        # PHI is left/right angle, THETA is up/down angle, both in degree
        #
        merge_image = np.zeros((height, width, self.channel))
        merge_mask = np.zeros((height, width, self.channel))

        for img, [F,P,T] in zip (self.img_array, self.F_P_T_array):
            per = Perspective2Panorama(img, F, P, T, channel=self.channel, interpolation=self.interpolation)        # Load equirectangular image
            img , mask = per.GetEquirec(height,width)   # Specify parameters(FOV, theta, phi, height, width)
            if self.average:
                merge_image += img
            else:
                merge_image = np.where(merge_image==0, img, merge_image)
            merge_mask +=mask

        if self.average:
            merge_mask = np.where(merge_mask==0,1,merge_mask)
            merge_image = (np.divide(merge_image,merge_mask))
        else:
            merge_mask = np.where(merge_mask>0,1,0)

        return merge_image, merge_mask



if __name__ == '__main__':
    cubemap_path = [f'/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/cubemap_rgb/0_skybox{i}_sami.png' for i in [2,3,4,1,0,5]]
    
    # cubemap fovs, phi and theta angles
    F_P_T_lst = [[90, 0, 0],  # front
                [90, -90, 0], # right
                [90, -180, 0], # back
                [90, -270, 0], # left
                [90, 0, 90], # up
                [90, 0, -90]] # down
    faces_img_lst = [cv2.imread(p, -1) for p in cubemap_path]
    faces_img_lst[4] = np.flip(np.transpose(faces_img_lst[4], (1,0,2)), axis=1)
    faces_img_lst[5] = np.flip(np.transpose(faces_img_lst[5], (1,0,2)), axis=0)
    img_channel = faces_img_lst[0].shape[-1]
    kwargs = {}
    kwargs['interpolation'] = cv2.INTER_CUBIC
    kwargs['average'] = True

    per = MultiPers2Panorama(faces_img_lst, F_P_T_lst, channel=img_channel, **kwargs)
    img, mask = per.GetEquirec(1024, 2048)
    img = img.astype(np.uint8)
    cv2.imwrite('test_result/test_panorama/extra_utils_cube_to_pano.png', img)

