import os
import sys
import torch
import argparse
import torch.nn as nn
import albumentations as A
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from helper_utils import load_config, load_checkpoint
from eval_utils import model_eval
from data_utils import Group_Activity_DataSet, group_activity_labels
from torch.utils.data import DataLoader
from helper_utils import load_config
from torchinfo import summary

class Group_Activity_Classifer(nn.Module):
    def __init__(self, num_classes):
        super(Group_Activity_Classifer, self).__init__()
        
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50.fc = nn.Linear(in_features=self.resnet50.fc.in_features, out_features=num_classes)
    
    def forward(self, x):
        return self.resnet50(x)


def model_summary(args):
    sys.path.append(os.path.abspath(args.project_root))

    config = load_config(args.config_path)
    model = Group_Activity_Classifer(num_classes=config.model['num_classes'])

    summary(model)


def eval(args, checkpoint_path):
    sys.path.append(os.path.abspath(args.project_root))

    config = load_config(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Group_Activity_Classifer(num_classes=config.model['num_classes'])
    model = load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device, optimizer=None)

    model = model.to(device)

    test_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    

    test_dataset = Group_Activity_DataSet(
        videos_path=f"{args.project_root}/{config.data['videos_path']}",
        annot_path=f"{args.project_root}/{config.data['annot_path']}",
        split=config.data['video_splits']['test'],
        labels=group_activity_labels, 
        transform=test_transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
   
    path = f"{args.project_root}/Baselines/baseline 1/outputs"
    prefix = "Group Activity Baseline 1 eval on testset"

    metrics = model_eval(model=model, data_loader=test_loader, criterion=criterion, device=device , path=path, prefix=prefix, class_names=config.model["num_clases_label"])

    return metrics


if __name__ == "__main__":

    ROOT = r"D:\VollyBall Project\project" 
    MODEL_CONFIG = r"D:\VollyBall Project\project\Baselines\configs\Baseline B1-tuned.yml"
    CHECKPOINT_PATH = r"D:\VollyBall Project\project\Baselines\baseline 1/outputs/Baseline_B1/checkpoint.pkl"

    parser = argparse.ArgumentParser(description="Group Activity Recognition Model Configuration")
    parser.add_argument("--project_root", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=MODEL_CONFIG,
                        help="Path to the YAML configuration file")

    args = parser.parse_args()

    eval(args, CHECKPOINT_PATH)

    # ==================================================
    # Group Activity Baseline 1 eval on testset
    # ==================================================
    # Accuracy : 72.66%
    # Average Loss: 1.1451
    # F1 Score (Weighted): 0.7263