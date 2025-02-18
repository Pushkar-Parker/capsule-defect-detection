from anomalib.data import Folder
from anomalib.models import Patchcore, Padim
from anomalib.engine import Engine
from anomalib import TaskType
from anomalib.deploy import ExportType

def train():

    dataset = Folder(
        name='capsule',
        root= r"C:\Users\Parker\Downloads\capsule\capsule\train",
        normal_dir='good',
        task= TaskType.CLASSIFICATION
    )

    dataset.setup()
    print(dataset)

    model = Padim()
    engine = Engine(max_epochs = 50, 
                    task= TaskType.CLASSIFICATION)

    
    engine.fit(datamodule=dataset, model=model)

    engine.export(export_type=ExportType.TORCH,
                model=model,
                export_root=r'C:\Users\Parker\Downloads\capsule\weights')
    
if __name__ == "__main__":
    train()