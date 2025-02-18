# Importing modules
from anomalib.data import Folder
from anomalib.models import Patchcore, Padim
from anomalib.engine import Engine
from anomalib import TaskType
from anomalib.deploy import ExportType

# Defining train function
def train(images_path: str, save_path: str): # Takes in train images and save paths

    # Defining dataset
    dataset = Folder(
        name='capsule',
        root= images_path,
        normal_dir='good',
        task= TaskType.CLASSIFICATION
    )

    dataset.setup()

    # Creating model instance
    model = Padim()

    # Creating engine instance
    engine = Engine(max_epochs = 50, 
                    task= TaskType.CLASSIFICATION)

    # Training the model
    engine.fit(datamodule=dataset, model=model)

    # Saving the model
    engine.export(export_type=ExportType.TORCH,
                model=model,
                export_root=save_path)

# Initiating training
if __name__ == "__main__":
    train(images_path= path, save_path= path)
