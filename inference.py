# Importing models
from anomalib.deploy import TorchInferencer
import torch
import cv2
from PIL import Image
from torchvision.transforms.v2.functional import to_dtype, to_image
from pathlib import Path
import os

# Defining inference function
def inference(images_path: str, model_path: str, save_path: str): # Takes in test images, model and save paths

    # Creating model instance. Inference will perform on CPU
    model = TorchInferencer(path= model_path, device='cpu')

    # Creating a list of test images
    test_images_path = Path(images_path)
    test_images = list(test_images_path.glob('*.png'))

    # Creating the save directory
    save_dir = save_path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Looping through each image in the test images and performing inference
    for test_image in test_images:

        # Processing the image into proper data type
        image = Image.open(test_image)
        image = to_dtype(to_image(image), torch.float32, scale=True)

        # Running inference
        result = model.predict(image=image)

        # Creating file name and path
        filename = test_image.name
        file_path = os.path.join(save_dir, filename)

        # Saving the image
        cv2.imwrite(file_path, result.heat_map)
        
        print(f'{filename} saved in {save_dir}')

    print(f'Inference done')

# Intiating the inference function
inference(model_path=r"C:\Users\Parker\Downloads\capsule\weights\weights\torch\model.pt",
          images_path=r"C:\Users\Parker\Downloads\capsule\capsule\test\good",
          save_path=r"D:\anomalib\inference\good")
