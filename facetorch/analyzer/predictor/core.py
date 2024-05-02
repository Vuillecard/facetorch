from typing import List

import torch
from codetiming import Timer
from facetorch.base import BaseDownloader, BaseModel
from facetorch.datastruct import Prediction
from facetorch.logger import LoggerJsonFile

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .post import BasePredPostProcessor
from .pre import BasePredPreProcessor

import cv2

logger = LoggerJsonFile().logger

class FacePredictor(BaseModel):
    @Timer(
        "FacePredictor.__init__", "{name}: {milliseconds:.2f} ms", logger=logger.debug
    )
    def __init__(
        self,
        downloader: BaseDownloader,
        device: torch.device,
        preprocessor: BasePredPreProcessor,
        postprocessor: BasePredPostProcessor,
        **kwargs
    ):
        """FacePredictor is a wrapper around a neural network model that is trained to predict facial features.

        Args:
            downloader (BaseDownloader): Downloader that downloads the model.
            device (torch.device): Torch device cpu or cuda for the model.
            preprocessor (BasePredPostProcessor): Preprocessor that runs before the model.
            postprocessor (BasePredPostProcessor): Postprocessor that runs after the model.
        """
        self.__dict__.update(kwargs)
        super().__init__(downloader, device)

        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @Timer("FacePredictor.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self, faces: torch.Tensor) -> List[Prediction]:
        """Predicts facial features.

        Args:
            faces (torch.Tensor): Torch tensor containing a batch of faces with values between 0-1 and shape (batch_size, channels, height, width).

        Returns:
            (List[Prediction]): List of Prediction data objects. One for each face in the batch.
        """
        faces = self.preprocessor.run(faces)
        preds = self.inference(faces)
        preds_list = self.postprocessor.run(preds)

        return preds_list


class FacePredictorMediapipe(BaseModel):
    @Timer(
        "FacePredictor.__init__", "{name}: {milliseconds:.2f} ms", logger=logger.debug
    )
    def __init__(
        self,
        downloader: BaseDownloader,
        device: torch.device,
        preprocessor: BasePredPreProcessor,
        postprocessor: BasePredPostProcessor,
        **kwargs
    ):
        """FacePredictor is a wrapper around a neural network model that is trained to predict facial features.

        Args:
            downloader (BaseDownloader): Downloader that downloads the model.
            device (torch.device): Torch device cpu or cuda for the model.
            preprocessor (BasePredPostProcessor): Preprocessor that runs before the model.
            postprocessor (BasePredPostProcessor): Postprocessor that runs after the model.
        """
        self.__dict__.update(kwargs)
        super().__init__(downloader, device)

        
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.model_setup()

    def load_model(self): 
        pass 

    def model_setup(self):
        base_options = python.BaseOptions(model_asset_path='/idiap/temp/pvuillecard/libs/facetorch_extra/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       min_face_detection_confidence=0.2,
                                       min_face_presence_confidence=0.2,     
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
        self.model = vision.FaceLandmarker.create_from_options(options)

    def inference(self, faces: torch.Tensor):
        
        detected_results = []
        for face in faces: 

            # convert tensor to opencv style 
            img = face.permute(1, 2, 0).cpu().numpy()
            img = (img * 255)
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # save the image to debug 
            # cv2.imwrite("/idiap/temp/pvuillecard/libs/facetorch_extra/data/output/testmediapipeinput.jpg", img)
            rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            detection_result = self.model.detect(rgb_frame)
            detected_results.append(detection_result)
        
        return detected_results

    def postprocess(self,preds):

        pred_list = []
        for detection_result in preds:
            face_pred = []
            # only the first face
            
            if len(detection_result.face_landmarks) == 0:
                pred_list.append([-1]*(478*3 + 4*4 +52))
            else:
                ldmrks = []
                for face_landmarks in detection_result.face_landmarks[0]:
                    ldmrks += [face_landmarks.x, face_landmarks.y, face_landmarks.z]
                R = detection_result.facial_transformation_matrixes[0]
                rotation_matrix = list(np.array(R).reshape(-1))
                blend = []
                for cat in detection_result.face_blendshapes[0]:
                    blend.append(cat.score)
                pred_list.append( ldmrks + rotation_matrix + blend  )
            
        return torch.tensor(pred_list)
    
    @Timer("FacePredictor.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self, faces: torch.Tensor) -> List[Prediction]:
        """Predicts facial features.

        Args:
            faces (torch.Tensor): Torch tensor containing a batch of faces with values between 0-1 and shape (batch_size, channels, height, width).

        Returns:
            (List[Prediction]): List of Prediction data objects. One for each face in the batch.
        """
        faces = self.preprocessor.run(faces)
        preds = self.inference(faces)
        preds = self.postprocess(preds)
        preds_list = self.postprocessor.run(preds)


        return preds_list
