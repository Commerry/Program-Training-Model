"""
Faster R-CNN training pieces: dataset, model, train/eval loops, exporters.

Deliberately depends only on torch, torchvision, OpenCV and NumPy. The previous
version used albumentations purely for resize-and-normalise, which added a
fragile dependency and, more importantly, normalised the images a second time
on top of the normalisation torchvision's detector already applies internally.
Feeding a pre-normalised tensor to the model shifts the input distribution away
from what the pretrained backbone expects and quietly costs accuracy.

Resizing is likewise left to the model. torchvision's GeneralizedRCNNTransform
scales each image so its shorter side hits min_size while preserving aspect
ratio, then pads a batch to a common size. Squashing images to a square in the
dataset — which is what this used to do — meant the model was trained on
distorted objects and then run at inference on undistorted ones, a silent
accuracy loss that nothing in the pipeline would flag.
"""

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def _imread(path):
    """Unicode-safe image read (see services/imaging.py for the rationale)."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class DetectionDataset(Dataset):
    """
    Images plus boxes in torchvision detection format.

    Class ids are 1-based because Faster R-CNN reserves 0 for background.
    """

    def __init__(self, images_dir, annotations_dir, class_map, image_list,
                 img_size=640, train=True):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.class_map = class_map
        # Not a target size — an upper bound on the longest side, applied only
        # to keep a 4000px source from blowing up memory before the model's
        # own transform gets to it.
        self.max_pixels = max(320, int(img_size) * 2)
        self.train = train
        self.image_files = [name for name in image_list
                            if (self.images_dir / name).is_file()]

    def __len__(self):
        return len(self.image_files)

    def _load_regions(self, filename):
        path = self.annotations_dir / f'{filename}.json'
        if not path.exists():
            return []
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                return json.load(f).get('regions', [])
        except (OSError, json.JSONDecodeError):
            return []

    def __getitem__(self, index):
        filename = self.image_files[index]
        image = _imread(self.images_dir / filename)
        if image is None:
            # Returning an empty sample keeps one unreadable file from killing
            # a run that is otherwise fine.
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            regions = []
        else:
            regions = self._load_regions(filename)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # A very large image is scaled down here purely to bound memory; the
        # aspect ratio is preserved so the model still sees true proportions.
        scale = 1.0
        longest = max(width, height)
        if longest > self.max_pixels:
            scale = self.max_pixels / longest
            image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                               interpolation=cv2.INTER_AREA)
            height, width = image.shape[:2]

        boxes, labels = [], []
        for region in regions:
            if not isinstance(region, dict):
                continue
            tag = str(region.get('tag') or '').strip()
            if tag not in self.class_map:
                continue
            try:
                x = float(region['x']) * scale
                y = float(region['y']) * scale
                w = float(region['width']) * scale
                h = float(region['height']) * scale
            except (KeyError, TypeError, ValueError):
                continue
            x1 = min(max(x, 0.0), width - 1.0)
            y1 = min(max(y, 0.0), height - 1.0)
            x2 = min(max(x + w, x1 + 1.0), float(width))
            y2 = min(max(y + h, y1 + 1.0), float(height))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_map[tag] + 1)  # 0 is background

        if self.train and boxes and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            boxes = [[width - x2, y1, width - x1, y2] for x1, y1, x2, y2 in boxes]

        # Only the /255 scaling here: GeneralizedRCNNTransform applies the
        # ImageNet mean/std normalisation and the resize itself.
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # torchvision accepts genuinely empty targets; the old dummy
            # background box taught the model to predict a 1x1 corner object.
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes_t,
            'labels': labels_t,
            'image_id': torch.tensor([index]),
            'area': (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0]),
            'iscrowd': torch.zeros((len(boxes_t),), dtype=torch.int64),
        }
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes, pretrained=True, img_size=640):
    """
    Faster R-CNN ResNet50-FPN with the head resized to num_classes + background.

    img_size drives the model's own resize transform, so the same value must be
    used at training and at inference; otherwise the model sees objects at a
    different scale than it was trained on.

    With pretrained=False the backbone weights are skipped too. torchvision
    otherwise still downloads the ImageNet ResNet50 checkpoint, which is pure
    waste when the caller is about to overwrite every parameter by loading a
    trained state_dict.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone='DEFAULT' if pretrained else None,
        progress=False,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    img_size = int(img_size)
    model.transform.min_size = (img_size,)
    model.transform.max_size = int(img_size * 1.5)
    return model


def train_one_epoch(model, loader, optimizer, device, epoch, scaler=None,
                    log=None, progress_every=25, should_stop=None):
    """One pass over the training set. Returns averaged loss components."""
    model.train()
    totals, seen = {}, 0
    batches = len(loader)

    for index, (images, targets) in enumerate(loader):
        if should_stop and should_stop():
            break

        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                   for t in targets]

        with torch.autocast('cuda', enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            # A non-finite loss poisons every later weight update, so the batch
            # is dropped rather than propagated.
            if log:
                log(f'  batch {index}: non-finite loss, skipped')
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        seen += 1
        totals['loss'] = totals.get('loss', 0.0) + loss_value
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())

        if log and progress_every and index % progress_every == 0:
            log(f'  epoch {epoch} batch {index}/{batches} loss={loss_value:.4f}')

    if not seen:
        return {'loss': float('nan')}
    return {key: value / seen for key, value in totals.items()}


@torch.no_grad()
def validation_loss(model, loader, device):
    """
    Average loss on the validation set.

    torchvision only returns losses in train mode, so the module is kept in
    train mode while gradients stay off. BatchNorm in the FPN backbone is
    frozen (FrozenBatchNorm2d), so this does not update any running statistics.
    """
    model.train()
    totals, seen = {}, 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = float(sum(loss_dict.values()))
        if not math.isfinite(loss):
            continue
        seen += 1
        totals['val_loss'] = totals.get('val_loss', 0.0) + loss
        for key, value in loss_dict.items():
            totals[f'val_{key}'] = totals.get(f'val_{key}', 0.0) + float(value)
    if not seen:
        return {'val_loss': float('nan')}
    return {key: value / seen for key, value in totals.items()}


@torch.no_grad()
def evaluate_map(model, loader, device, num_classes, iou_threshold=0.5,
                 score_threshold=0.5, max_batches=None):
    """
    mAP at a single IoU threshold, plus precision and recall.

    Average precision is threshold-independent and uses every prediction.
    Precision and recall are not: torchvision emits ~100 boxes per image down
    to a score of 0.05, so counting all of them gave a precision near zero no
    matter how good the model was. They are computed at score_threshold, which
    is the operating point a user would actually deploy.

    Implemented here rather than pulled from pycocotools/torchmetrics to keep
    the dependency list small.
    """
    model.eval()
    # per class: list of (score, is_true_positive), and the ground-truth count
    predictions = {c: [] for c in range(1, num_classes + 1)}
    gt_counts = {c: 0 for c in range(1, num_classes + 1)}

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = [image.to(device) for image in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            gt_boxes = target['boxes'].to(device)
            gt_labels = target['labels'].to(device)
            for label in gt_labels.tolist():
                if label in gt_counts:
                    gt_counts[label] += 1

            order = output['scores'].argsort(descending=True)
            pred_boxes = output['boxes'][order]
            pred_scores = output['scores'][order]
            pred_labels = output['labels'][order]

            matched = set()
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                label = int(label)
                if label not in predictions:
                    continue
                candidates = (gt_labels == label).nonzero(as_tuple=True)[0]
                best_iou, best_index = 0.0, -1
                for candidate in candidates.tolist():
                    if candidate in matched:
                        continue
                    iou = float(torchvision.ops.box_iou(
                        box.unsqueeze(0), gt_boxes[candidate].unsqueeze(0)
                    )[0, 0])
                    if iou > best_iou:
                        best_iou, best_index = iou, candidate
                is_tp = best_iou >= iou_threshold and best_index >= 0
                if is_tp:
                    matched.add(best_index)
                predictions[label].append((float(score), is_tp))

    average_precisions = {}
    total_tp = total_fp = 0
    for label, entries in predictions.items():
        total_gt = gt_counts[label]
        if total_gt == 0:
            continue
        entries.sort(key=lambda item: item[0], reverse=True)
        tp = fp = 0
        recall_points, precision_points = [], []
        for score, is_tp in entries:
            if is_tp:
                tp += 1
            else:
                fp += 1
            recall_points.append(tp / total_gt)
            precision_points.append(tp / (tp + fp))
            # Counted separately at the operating point, for precision/recall.
            if score >= score_threshold:
                if is_tp:
                    total_tp += 1
                else:
                    total_fp += 1

        # All-point interpolation: make precision monotonically decreasing,
        # then integrate over recall.
        for i in range(len(precision_points) - 2, -1, -1):
            precision_points[i] = max(precision_points[i], precision_points[i + 1])
        area, previous_recall = 0.0, 0.0
        for recall, precision in zip(recall_points, precision_points):
            area += (recall - previous_recall) * precision
            previous_recall = recall
        average_precisions[label] = area

    total_gt = sum(gt_counts.values())
    mean_ap = (sum(average_precisions.values()) / len(average_precisions)
               if average_precisions else 0.0)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / total_gt if total_gt else 0.0

    return {
        'mAP50': round(mean_ap, 5),
        'precision': round(precision, 5),
        'recall': round(recall, 5),
        'score_threshold': score_threshold,
        'per_class_ap': {str(label): round(value, 5)
                         for label, value in sorted(average_precisions.items())},
    }


# ── exporters ───────────────────────────────────────────────────────────────

def export_to_onnx(model, save_path, img_size=640, device=None, opset=11):
    """
    Export to ONNX. Faster R-CNN takes List[Tensor], not a batched tensor.

    The dummy input is square at img_size; the exported graph still accepts
    other shapes because the RPN is traced with dynamic spatial dimensions.

    torch 2.6+ defaults torch.onnx.export to the dynamo exporter, which needs
    the separate `onnxscript` package and does not handle Faster R-CNN's
    data-dependent control flow. The TorchScript exporter is requested
    explicitly; without it the export failed with
    "No module named 'onnxscript'" on an otherwise complete install.
    """
    device = device or next(model.parameters()).device
    model.eval()
    dummy = [torch.zeros(3, img_size, img_size, device=device)]
    kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=False,  # constant folding breaks the RPN's dynamic shapes
        input_names=['input'],
        output_names=['boxes', 'labels', 'scores'],
    )
    try:
        torch.onnx.export(model, (dummy,), str(save_path), dynamo=False, **kwargs)
    except TypeError:
        # Older torch has no `dynamo` argument and is legacy-only anyway.
        torch.onnx.export(model, (dummy,), str(save_path), **kwargs)
    return Path(save_path)


def export_to_torchscript(model, save_path, img_size=640, device=None):
    """Export to TorchScript by scripting — tracing bakes in the box count."""
    device = device or next(model.parameters()).device
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(str(save_path))
    return Path(save_path)


def export_to_blob(onnx_path, save_path, shaves=6):
    """Compile an ONNX model to a Luxonis OAK .blob via blobconverter."""
    try:
        import blobconverter
    except ImportError:
        raise RuntimeError('blobconverter is not installed. Run: pip install blobconverter')

    produced = blobconverter.from_onnx(model=str(onnx_path), data_type='FP16', shaves=shaves)
    import shutil
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(save_path))
    return save_path
