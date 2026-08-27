#!/usr/bin/env python
"""
Faster R-CNN (torchvision) training worker.

Run as: python frcnn_worker.py <training_config.json>

Reads the same prepared dataset the YOLO worker uses, so both model families
train on exactly the same train/val split.
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.worker_common import (  # noqa: E402
    append_history, describe_device, load_config, make_logger, quiet_environment,
    read_config, stop_requested, update_status,
)

quiet_environment()

# Upper bound on validation batches scored for mAP each epoch. Keeps the
# per-epoch cost predictable on a large validation split.
MAP_EVAL_BATCHES = 50


def _split_filenames(dataset_dir, split):
    """Filenames in one split of the prepared dataset."""
    directory = Path(dataset_dir) / 'images' / split
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.iterdir() if p.is_file())


def main():
    config_file, config = load_config(sys.argv)
    log = make_logger(config_file.parent / 'training.log')

    model_name = config['model_name']
    epochs = int(config['epochs'])
    batch_size = int(config['batch_size'])
    img_size = int(config['img_size'])
    learning_rate = float(config.get('learning_rate') or 0.005)
    export_formats = config.get('export_formats', ['pt'])
    classes = config['classes']
    project_path = Path(config['project_path'])
    dataset_dir = Path(config['dataset_path'])
    results_dir = Path(config['results_dir'])

    try:
        if stop_requested(config_file):
            log('Cancelled before start')
            update_status(config_file, {'status': 'stopped', 'pid': None})
            return

        import torch
        from torch.utils.data import DataLoader

        from training.frcnn_lib import (
            DetectionDataset, collate_fn, create_model, evaluate_map,
            export_to_blob, export_to_onnx, export_to_torchscript,
            train_one_epoch, validation_loss,
        )

        _, device_label = describe_device()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log(f'Device: {device_label}')
        update_status(config_file, {'status': 'running', 'device': device_label})

        class_map = {name: index for index, name in enumerate(classes)}
        train_names = _split_filenames(dataset_dir, 'train')
        val_names = _split_filenames(dataset_dir, 'val')
        if not train_names:
            raise RuntimeError(f'No training images found in {dataset_dir}')
        log(f'Dataset: {len(train_names)} train / {len(val_names)} val images')

        images_dir = project_path / 'images'
        annotations_dir = project_path / 'annotations'

        train_loader = DataLoader(
            DetectionDataset(images_dir, annotations_dir, class_map, train_names,
                             img_size=img_size, train=True),
            batch_size=batch_size, shuffle=True, num_workers=0,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            DetectionDataset(images_dir, annotations_dir, class_map, val_names,
                             img_size=img_size, train=False),
            batch_size=max(1, batch_size), shuffle=False, num_workers=0,
            collate_fn=collate_fn,
        )

        log('Building Faster R-CNN ResNet50-FPN (downloads weights on first run)...')
        model = create_model(num_classes=len(classes), img_size=img_size).to(device)

        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9,
                                    weight_decay=0.0005)
        # Cosine decay over the whole run: the old fixed StepLR(step=3, gamma=0.1)
        # drove the learning rate to ~0 within 10 epochs, so every epoch after
        # that changed nothing.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

        weights_dir = results_dir / model_name / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)
        best_path = weights_dir / 'best.pth'
        last_path = weights_dir / 'last.pth'

        history = []
        best_score = -1.0
        stopped_early = False

        for epoch in range(1, epochs + 1):
            if stop_requested(config_file):
                stopped_early = True
                log('Stop requested — ending training')
                break

            update_status(config_file, {'current_epoch': epoch})
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, device, epoch, scaler=scaler,
                log=log, should_stop=lambda: stop_requested(config_file),
            )
            scheduler.step()

            metrics = {'epoch': epoch,
                       'train_loss': round(train_metrics.get('loss', 0.0), 5),
                       'lr': round(optimizer.param_groups[0]['lr'], 8)}
            for key in ('loss_classifier', 'loss_box_reg', 'loss_objectness',
                        'loss_rpn_box_reg'):
                if key in train_metrics:
                    metrics[key] = round(train_metrics[key], 5)

            if val_names:
                metrics.update({k: round(v, 5)
                                for k, v in validation_loss(model, val_loader, device).items()})
                # mAP every epoch, but capped to a fixed number of batches so
                # the cost is bounded on a large validation set. Measuring it
                # every epoch is what makes "best checkpoint" a single
                # comparable scale; the old schedule mixed mAP on some epochs
                # with negative validation loss on others, so once one mAP
                # epoch landed no later epoch could ever win.
                metrics.update(evaluate_map(model, val_loader, device, len(classes),
                                            max_batches=MAP_EVAL_BATCHES))

            history.append(metrics)
            update_status(config_file, {'current_epoch': epoch, 'metrics': metrics,
                                        'metrics_history': history[-500:]})
            log(f'Epoch {epoch}/{epochs} ' + ' '.join(
                f'{k}={v}' for k, v in metrics.items() if k != 'epoch'))

            torch.save(model.state_dict(), last_path)
            # One scale throughout: mAP when there is a validation set, and
            # negative training loss when there is not.
            score = metrics.get('mAP50')
            if score is None:
                val_loss = metrics.get('val_loss')
                score = -(val_loss if val_loss is not None else metrics['train_loss'])
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), best_path)
                log(f'  new best checkpoint (score={round(score, 5)})')

        if not best_path.exists():
            if last_path.exists():
                best_path = last_path
            else:
                raise RuntimeError('Training produced no checkpoint')

        # ── load the best checkpoint before doing anything else ───────────
        # Everything below — the final metrics and every export — must describe
        # the same weights. Loading this only inside the validation branch
        # meant a run without a validation set exported the last epoch while
        # reporting best.pth alongside it.
        try:
            model.load_state_dict(torch.load(best_path, map_location=device))
            log(f'Loaded best checkpoint for export: {best_path.name}')
        except Exception as exc:  # noqa: BLE001 - keep going with what we have
            log(f'WARNING: could not load {best_path.name} ({exc}); '
                'exports will describe the final epoch instead')

        final_metrics = dict(history[-1]) if history else {}
        if val_names:
            try:
                log('Evaluating best checkpoint...')
                final_metrics.update(evaluate_map(model, val_loader, device, len(classes)))
            except Exception as exc:  # noqa: BLE001 - keep the weights regardless
                log(f'Final evaluation failed: {exc}')

        # ── exports ──────────────────────────────────────────────────────
        # Only the state_dict is published. A torch.save(model) pickle would
        # be named .pt, which the model tester routes to ultralytics, so it
        # loaded as a YOLO checkpoint and failed with an opaque 500. It also
        # cannot be loaded without this exact source tree present.
        exported = {'pth': str(best_path)}

        onnx_path = weights_dir / 'best.onnx'
        needs_onnx = 'onnx' in export_formats or 'blob' in export_formats
        if needs_onnx:
            try:
                export_to_onnx(model, onnx_path, img_size, device)
                exported['onnx'] = str(onnx_path)
                log(f'  onnx -> {onnx_path}')
            except Exception as exc:  # noqa: BLE001
                exported['onnx'] = None
                log(f'  onnx FAILED: {exc}')

        if 'torchscript' in export_formats:
            try:
                ts_path = weights_dir / 'best.torchscript'
                export_to_torchscript(model, ts_path, img_size, device)
                exported['torchscript'] = str(ts_path)
                log(f'  torchscript -> {ts_path}')
            except Exception as exc:  # noqa: BLE001
                exported['torchscript'] = None
                log(f'  torchscript FAILED: {exc}')

        if 'blob' in export_formats:
            try:
                if not onnx_path.exists():
                    raise RuntimeError('ONNX export is required for .blob and it failed')
                blob_path = weights_dir / 'best.blob'
                export_to_blob(onnx_path, blob_path)
                exported['blob'] = str(blob_path)
                log(f'  blob -> {blob_path}')
            except Exception as exc:  # noqa: BLE001
                exported['blob'] = None
                log(f'  blob FAILED: {exc}')

        # evaluate_map returns per-class AP keyed by the 1-based label id used
        # inside the model; the UI wants class names.
        per_class = {}
        for label_id, ap in (final_metrics.pop('per_class_ap', {}) or {}).items():
            try:
                index = int(label_id) - 1
            except (TypeError, ValueError):
                continue
            name = classes[index] if 0 <= index < len(classes) else str(label_id)
            per_class[str(name)] = {'ap50': ap}

        finished_at = datetime.now().isoformat()
        final_status = 'stopped' if stopped_early else 'completed'
        update_status(config_file, {
            'status': final_status,
            'completed_at': finished_at,
            'best_model': str(best_path),
            'run_dir': str(results_dir / model_name),
            'exported_models': exported,
            'metrics': final_metrics,
            'per_class': per_class,
            'pid': None,
            'error': None,
        })

        current = read_config(config_file)
        append_history(project_path, {
            'model_name': model_name,
            'model_type': 'faster_rcnn',
            'status': final_status,
            'epochs': epochs,
            'completed_epochs': current.get('current_epoch', 0),
            'batch_size': batch_size,
            'img_size': img_size,
            'classes': classes,
            'train_images': len(train_names),
            'val_images': len(val_names),
            'metrics': final_metrics,
            'per_class': per_class,
            'best_model': str(best_path),
            'exported_models': exported,
            'started_at': config.get('started_at'),
            'completed_at': finished_at,
        })
        log(f'Training {final_status}.')

    except Exception as exc:  # noqa: BLE001 - every failure is reported to the UI
        message = str(exc) or exc.__class__.__name__
        log(f'ERROR: {message}')
        log(traceback.format_exc())
        update_status(config_file, {
            'status': 'failed', 'error': message,
            'completed_at': datetime.now().isoformat(), 'pid': None,
        })
        append_history(project_path, {
            'model_name': model_name, 'model_type': 'faster_rcnn',
            'status': 'failed', 'error': message, 'epochs': epochs,
            'started_at': config.get('started_at'),
            'completed_at': datetime.now().isoformat(),
        })
        sys.exit(1)


if __name__ == '__main__':
    main()
