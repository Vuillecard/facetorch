import torch
import torchvision
import numpy as np
from codetiming import Timer
from facetorch.base import BaseUtilizer
from facetorch.datastruct import ImageData
from facetorch.logger import LoggerJsonFile
from torchvision import transforms
import cv2 
from typing import List
logger = LoggerJsonFile().logger


class BoxDrawer(BaseUtilizer):
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        color: str,
        line_width: int,
    ):
        """Initializes the BoxDrawer class. This class is used to draw the face boxes to the image tensor.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transform.
            color (str): Color of the boxes.
            line_width (int): Line width of the boxes.

        """
        super().__init__(transform, device, optimize_transform)
        self.color = color
        self.line_width = line_width

    @Timer("BoxDrawer.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Draws face boxes to the image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor and face locations.
        Returns:
            ImageData: ImageData object containing the image tensor with face boxes.
        """
        loc_tensor = data.aggregate_loc_tensor()
        labels = [str(face.indx) for face in data.faces]
        data.img = torchvision.utils.draw_bounding_boxes(
            image=data.img,
            boxes=loc_tensor,
            labels=labels,
            colors=self.color,
            width=self.line_width,
        )

        return data


class LandmarkDrawerTorch(BaseUtilizer):
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        width: int,
        color: str,
    ):
        """Initializes the LandmarkDrawer class. This class is used to draw the 3D face landmarks to the image tensor.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transform.
            width (int): Marker keypoint width.
            color (str): Marker color.

        """
        super().__init__(transform, device, optimize_transform)
        self.width = width
        self.color = color

    @Timer("LandmarkDrawer.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Draws 3D face landmarks to the image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor and 3D face landmarks.
        Returns:
            ImageData: ImageData object containing the image tensor with 3D face landmarks.
        """
        data = self._draw_landmarks(data)

        return data

    def _draw_landmarks(self, data: ImageData) -> ImageData:
        """Draws 3D face landmarks to the image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor, 3D face landmarks, and faces.

        Returns:
            (ImageData): ImageData object containing the image tensor with 3D face landmarks.
        """

        if len(data.faces) > 0:
            pts = [face.preds["align"].other["lmk3d"].cpu() for face in data.faces if face.preds["align"].other["lmk3d"] is not None]


            if len(pts) == 0:
                return data

            img_in = data.img.clone()
            pts = torch.stack(pts)
            pts = torch.swapaxes(pts, 2, 1)

            img_out = torchvision.utils.draw_keypoints(
                img_in,
                pts,
                colors=self.color,
                radius=self.width,
            )
            data.img = img_out

        return data

class HeadPoseDrawerTorch(BaseUtilizer):
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        size: int,
    ):
        """Initializes the HeadPoseDrawerTorch class. This class is used to draw the 3D face landmarks to the image tensor.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transform.
            width (int): Marker keypoint width.
            color (str): Marker color.

        """
        super().__init__(transform, device, optimize_transform)
        self.size = size

    @Timer("LandmarkDrawer.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Draws 3D face landmarks to the image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor and 3D face landmarks.
        Returns:
            ImageData: ImageData object containing the image tensor with 3D face landmarks.
        """
        data = self._draw_headpose(data)

        return data

    def _draw_headpose(self, data: ImageData) -> ImageData:
        """Draws 3D face landmarks to the image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor, 3D face landmarks, and faces.

        Returns:
            (ImageData): ImageData object containing the image tensor with 3D face landmarks.
        """

        if len(data.faces) > 0:
            
            size = self.size
            for face in data.faces:
               
                pose = face.preds["align"].other["pose"]
                if pose is None:
                    continue
                yaw, pitch, roll = pose['angles'][0], pose['angles'][1], pose['angles'][2]
                #tdx, tdy = pose['translation'][0], pose['translation'][1]
                tdx, tdy = None, None
                pts68 = face.preds["align"].other["lmk3d"].cpu().numpy()
                
                
                # in degree
                pitch = pitch * np.pi / 180
                yaw = -(yaw * np.pi / 180)
                roll = roll * np.pi / 180

                img = data.img.clone()
                # convert torch tensor to cv2 image 
                img = img.permute(1, 2, 0).cpu().numpy()

                if tdx != None and tdy != None:
                    tdx = tdx
                    tdy = tdy
                else:
                    height, width = img.shape[:2]
                    tdx = width / 2
                    tdy = height / 2

                # nose from mediapipe
                if len(pts68[0,:])> 300:
                    tdx = pts68[0,1]
                    tdy = pts68[1,1]
                else :
                    tdx = pts68[0,30]
                    tdy = pts68[1,30]

                minx, maxx = np.min(pts68[0, :]), np.max(pts68[0, :])
                miny, maxy = np.min(pts68[1, :]), np.max(pts68[1, :])
                llength = np.sqrt((maxx - minx) * (maxy - miny))
                size = llength * 0.5

                # if pts8 != None:
                #     tdx = 

                # X-Axis pointing to right. drawn in red
                x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
                y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

                # Y-Axis | drawn in green
                #        v
                x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
                y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

                # Z-Axis (out of the screen) drawn in blue
                x3 = size * (np.sin(yaw)) + tdx
                y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

                minus=0

                cv2.line(img, (int(tdx), int(tdy)-minus), (int(x1),int(y1)),(255,0,0),4)
                cv2.line(img, (int(tdx), int(tdy)-minus), (int(x2),int(y2)),(0,255,0),4)
                cv2.line(img, (int(tdx), int(tdy)-minus), (int(x3),int(y3)),(0,0,255),4)

                # convert cv2 image to torch tensor
                img = torch.from_numpy(img).permute(2, 0, 1)
                data.img = img

        return data
    

class TextDrawerTorch(BaseUtilizer):
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        text_output: List[str],
    ):
        """Initializes the HeadPoseDrawerTorch class. This class is used to draw the 3D face landmarks to the image tensor.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transform.
            width (int): Marker keypoint width.
            color (str): Marker color.

        """
        super().__init__(transform, device, optimize_transform)
        self.text_output = text_output

    @Timer("LandmarkDrawer.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Draws 3D face landmarks to the image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor and 3D face landmarks.
        Returns:
            ImageData: ImageData object containing the image tensor with 3D face landmarks.
        """
        data = self._draw_headpose(data)

        return data

    def _draw_headpose(self, data: ImageData) -> ImageData:
        """Draws 3D face landmarks to the image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor, 3D face landmarks, and faces.

        Returns:
            (ImageData): ImageData object containing the image tensor with 3D face landmarks.
        """

        if len(data.faces) > 0:
            
            color = (255, 0, 0)
            face = data.faces[0]
            face_idx = face.indx
            image_height, image_width = data.dims.height, data.dims.width
            img = data.img.clone()
            # convert torch tensor to cv2 image 
            img = img.permute(1, 2, 0).cpu().numpy()
            # write the prediction at the top corner of the image 
            cv2.putText(img, f'Face {face_idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            space_text = 0
            for key_i, key in enumerate(self.text_output):
                if key == 'au': 
                    if len(face.preds[key].other['multi']) == 0:
                        cv2.putText(img, f'{key}: None', (10, 60 + space_text*30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        space_text += 1
                    else:
                        cv2.putText(img, f'{key}: {face.preds[key].other["multi"][0]}', (10, 60 + space_text*30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        space_text += 1
                        for au_key_i, au_key in enumerate(face.preds[key].other['multi'][1:]):
                            cv2.putText(img, f'    {au_key}', (10, 60 + space_text*30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            space_text += 1   
                elif key == 'va':
                    cv2.putText(img, f'{key}: val {round(face.preds[key].other["valence"],2)},aro {round(face.preds[key].other["arousal"],2)}',
                                 (10, 60 + space_text*30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    space_text += 1
                else:
                    cv2.putText(img, f'{key}: {face.preds[key].label}', (10, 60 + key_i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    space_text += 1
            # convert cv2 image to torch tensor
            img = torch.from_numpy(img).permute(2, 0, 1)
            data.img = img

        return data