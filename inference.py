from anomalib.deploy import TorchInferencer
import torch
import cv2
from PIL import Image
from torchvision.transforms.v2.functional import to_dtype, to_image
from pathlib import Path
import os

def inference(images_path: str, model_path: str, save_path: str):

    model = TorchInferencer(path= model_path, device='cpu')

    test_images_path = Path(images_path)
    test_images = list(test_images_path.glob('*.png'))
    
    save_dir = save_path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for test_image in test_images:

        image = Image.open(test_image)
        image = to_dtype(to_image(image), torch.float32, scale=True)
        result = model.predict(image=image)

        filename = test_image.name
        file_path = os.path.join(save_dir, filename)

        cv2.imwrite(file_path, result.heat_map)
        
        print(f'{filename} saved in {save_dir}')

    print(f'Inference done')

inference(model_path=r"C:\Users\Parker\Downloads\capsule\weights\weights\torch\model.pt",
          images_path=r"C:\Users\Parker\Downloads\capsule\capsule\test\good",
          save_path=r"D:\anomalib\inference\good")