"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask

        """
        # Reshape the volume to be of shape [X, patch_size, patch_size]
        volume = med_reshape(volume, (volume.shape[0], self.patch_size, self.patch_size))
        return self.single_volume_inference(volume)

        raise NotImplementedError

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>x

        # Reshape volume to have consistent Y and Z dimensions and get slices for X dimension
        # volume = med_reshape(volume, (volume.shape[0], self.patch_size, self.patch_size))
        
        # Iterate through the X dimension and get slices.
        for slice_idx in range(volume.shape[0]):
            slice_2d = volume[slice_idx, :, :]
            
            # Convert the slice into a torch tensor, add batch and channel dimensions.
            tensor_slice = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().to(self.device)

            # Model prediction for the slice.
            prediction = self.model(tensor_slice)

            # Extracting the channel with the highest probability.
            slice_prediction = torch.argmax(prediction, dim=1).squeeze().detach().cpu().numpy()
            slices.append(slice_prediction)

        # Stack the 2D slices into a 3D volume.
        prediction_volume = np.stack(slices, axis=0)
        return prediction_volume







