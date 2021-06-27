import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from helpers import image_utils
import numpy as np
from helpers.utils import (
    overlay_ann,
    overlay_mask,
    show
)

CATEGORIES2LABELS = {
    0: "bg",
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure"
}
NUM_CLASSES = 6
MODEL_PATH = "./models/layout_structure/final.pth"


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    return model

# Load model
model = get_instance_segmentation_model(NUM_CLASSES)
model.cuda()

# Load checkpoint
checkpoint_path = MODEL_PATH
checkpoint = torch.load(checkpoint_path, map_location='cpu')

test_datas = image_utils.load_datasets('./datasets/document_classify/test')


for file_path in test_datas:
    image = image_utils.load_image_transform(file_path)
    with torch.no_grad():
        prediction = model([image.cuda()])
        prediction = model([image])

    image = torch.squeeze(image, 0).permute(1, 2, 0).mul(255).numpy().astype(np.uint8)

    for pred in prediction:
        for idx, mask in enumerate(pred['masks']):
            if pred['scores'][idx].item() < 0.7:
                continue
            m = mask[0].mul(255).byte().cpu().numpy()
            box = list(map(int, pred["boxes"][idx].tolist()))
            label = CATEGORIES2LABELS[pred["labels"][idx].item()]
            score = pred["scores"][idx].item()
            image = overlay_mask(image, m)
            image = overlay_ann(image, m, box, label, score)
    show(image)