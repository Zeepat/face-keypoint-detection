import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_face_detector(num_classes=2, 
                      pretrained=False, 
                      pretrained_backbone=False):
    """
    Returns a Faster R-CNN model for face detection,
    from scratch if pretrained=False.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone
    )
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# "num_classes = 2" -> [background, face]


#alternatively, if not pretrained
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_face_detector_np(num_classes=2, 
                      pretrained=False, 
                      pretrained_backbone=False):
    """
    Returns a Faster R-CNN model for face detection,
    from scratch if pretrained=False.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone
    )
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

