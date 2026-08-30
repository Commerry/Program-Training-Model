#!/usr/bin/env python
"""
Ultralytics YOLO / RT-DETR training worker.

Run as: python yolo_worker.py <training_config.json>

Everything this process needs is in the config file the web process wrote,
including the path to the prepared dataset. It reports progress by updating
that same file after every epoch.
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.worker_common import (  # noqa: E402
    append_history, describe_device, load_config, make_logger, quiet_environment,
    read_config, self_check, stop_requested, update_status,
)

quiet_environment()

# Metric keys ultralytics reports, mapped to the short names the UI displays.
METRIC_ALIASES = {
    'metrics/precision(B)': 'precision',
    'metrics/recall(B)': 'recall',
    'metrics/mAP50(B)': 'mAP50',
    'metrics/mAP50-95(B)': 'mAP50_95',
    'val/box_loss': 'val_box_loss',
    'val/cls_loss': 'val_cls_loss',
    'val/dfl_loss': 'val_dfl_loss',
}


def normalise_metrics(raw):
    """Reduce a raw ultralytics metrics dict to plain JSON-safe numbers."""
    out = {}
    for key, value in (raw or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number:  # NaN
            continue
        out[METRIC_ALIASES.get(key, key.split('/')[-1])] = round(number, 5)
    return out


def per_class_metrics(results, classes):
    """
    Per-class AP from an ultralytics validation result.

    A single overall mAP hides the thing a user most needs to know: which
    class the model is bad at. On a ten-digit detector, "0.72 mAP" and "every
    digit is fine except 8" are very different situations.

    results.box.ap50 is ordered by results.box.ap_class_index, not by class id,
    so the two have to be zipped rather than assumed aligned.
    """
    out = {}
    box = getattr(results, 'box', None)
    if box is None:
        return out
    try:
        ap50 = list(getattr(box, 'ap50', []) or [])
        indices = list(getattr(box, 'ap_class_index', []) or [])
        precision = list(getattr(box, 'p', []) or [])
        recall = list(getattr(box, 'r', []) or [])
    except Exception:  # noqa: BLE001
        return out

    for position, class_index in enumerate(indices):
        try:
            class_index = int(class_index)
        except (TypeError, ValueError):
            continue
        name = classes[class_index] if 0 <= class_index < len(classes) else str(class_index)
        entry = {}
        if position < len(ap50):
            entry['ap50'] = round(float(ap50[position]), 5)
        if position < len(precision):
            entry['precision'] = round(float(precision[position]), 5)
        if position < len(recall):
            entry['recall'] = round(float(recall[position]), 5)
        out[str(name)] = entry
    return out


def main():
    config_file, config = load_config(sys.argv)
    log = make_logger(config_file.parent / 'training.log')

    model_type = config['model_type']
    model_name = config['model_name']
    epochs = int(config['epochs'])
    batch_size = int(config['batch_size'])
    img_size = int(config['img_size'])
    learning_rate = float(config.get('learning_rate') or 0.01)
    export_formats = [f for f in config.get('export_formats', ['pt']) if f != 'pt']
    weights = config.get('weights') or f'{model_type}.pt'
    data_yaml = Path(config['data_yaml'])
    results_dir = Path(config['results_dir'])
    project_path = config['project_path']

    try:
        if stop_requested(config_file):
            log('Cancelled before start')
            update_status(config_file, {'status': 'stopped', 'pid': None})
            return

        if not data_yaml.exists():
            raise FileNotFoundError(f'Prepared dataset is missing: {data_yaml}')

        device, device_label = describe_device()
        log(f'Device: {device_label}')
        base = config.get('base_model')
        if base:
            log(f'Continuing from a model this server trained: {base.get("name")}')
            comparison = config.get('base_classes') or {}
            if comparison.get('note'):
                log(f'  classes: {comparison["note"]}')
        else:
            log(f'Loading pretrained weights: {weights}')

        from ultralytics import YOLO, RTDETR
        loader = RTDETR if model_type.startswith('rtdetr') else YOLO
        model = loader(weights)

        update_status(config_file, {'status': 'running', 'current_epoch': 0,
                                    'device': device_label})

        history = list(config.get('metrics_history') or [])

        def on_epoch_end(trainer):
            # Clamped: ultralytics fires this hook once more for the final
            # validation pass after the last epoch, which otherwise reported
            # progress as "4/3".
            epoch = min(int(getattr(trainer, 'epoch', 0)) + 1, epochs)
            metrics = normalise_metrics(getattr(trainer, 'metrics', None))

            loss_items = getattr(trainer, 'loss_items', None)
            loss_names = getattr(trainer, 'loss_names', None) or []
            if loss_items is not None:
                for index, loss_name in enumerate(loss_names):
                    try:
                        metrics[f'train_{loss_name}'] = round(float(loss_items[index]), 5)
                    except (IndexError, TypeError, ValueError):
                        pass

            metrics['epoch'] = epoch
            history.append(metrics)
            update_status(config_file, {
                'current_epoch': epoch,
                'metrics': metrics,
                # Bounded so a 1000-epoch run does not grow the status file
                # into something the UI has to download on every poll.
                'metrics_history': history[-500:],
            })
            summary = ' '.join(
                f'{k}={v}' for k, v in metrics.items()
                if k in ('mAP50', 'mAP50_95', 'precision', 'recall')
            )
            log(f'Epoch {epoch}/{epochs} {summary}'.rstrip())

            if stop_requested(config_file):
                log('Stop requested — finishing after this epoch')
                # Ultralytics checks this flag at the top of each epoch and
                # exits the loop cleanly, keeping the weights written so far.
                trainer.stop = True
                trainer.stop_training = True

        # on_fit_epoch_end, not on_train_epoch_end: ultralytics assigns
        # trainer.metrics inside validate(), which runs *after* the training
        # hook fires. Reading them there recorded the previous epoch's numbers
        # against the current epoch, and left epoch 1 with none at all.
        model.add_callback('on_fit_epoch_end', on_epoch_end)

        log('Starting training...')
        # Augmentation is passed explicitly rather than left to ultralytics'
        # defaults. Those defaults mirror half of every epoch (fliplr=0.5),
        # which is right for most objects and wrong for any class that reads as
        # text: a mirrored 2 is not a 2, and its label still says it is. What
        # arrives here was decided from the project's own class names, and can
        # be overridden from the training screen.
        augmentation = config.get('augmentation') or {}
        log(f'augmentation: {augmentation}' if augmentation
            else 'augmentation: ultralytics defaults')

        model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=learning_rate,
            project=str(results_dir),
            name=model_name,
            exist_ok=True,
            device=device,
            # How many processes decode and augment images alongside the GPU.
            # This was pinned to 0 because ultralytics' dataloader workers used
            # to deadlock when this worker is itself a spawned subprocess on
            # Windows. Left configurable rather than pinned: on a fast card a
            # single decoding thread is the bottleneck, and the failure mode if
            # it returns is visible (a run that never reaches epoch 1) and
            # recoverable by setting it back to 0.
            workers=int(config.get('workers', 0) or 0),
            # Images held in memory across epochs. Worth it when the set fits;
            # pointless when decoding is not what the run is waiting on.
            cache=config.get('cache') or False,
            verbose=False,
            plots=True,
            patience=max(20, epochs // 5),
            seed=42,
            **augmentation,
        )

        run_dir = results_dir / model_name
        best_pt = run_dir / 'weights' / 'best.pt'
        last_pt = run_dir / 'weights' / 'last.pt'
        if not best_pt.exists() and last_pt.exists():
            best_pt = last_pt

        stopped_early = stop_requested(config_file)
        per_class = {}

        if not best_pt.exists():
            raise RuntimeError(
                'Training finished but no weights file was produced. '
                f'Expected {best_pt}'
            )
        log(f'Best weights: {best_pt}')

        # ── Final validation on the held-out split ────────────────────────
        final_metrics = dict(read_config(config_file).get('metrics') or {})
        try:
            log('Running final validation...')
            validator = loader(str(best_pt))
            # project/name are given explicitly; without them ultralytics
            # writes validation output to its own ./runs/detect/val folder.
            results = validator.val(data=str(data_yaml), imgsz=img_size,
                                    device=device, workers=0, verbose=False,
                                    split='val', project=str(results_dir),
                                    name=f'{model_name}_val', exist_ok=True)
            final_metrics.update(normalise_metrics(getattr(results, 'results_dict', None)))
            per_class = per_class_metrics(results, config.get('classes', []))
            log('Validation: ' + ', '.join(
                f'{k}={final_metrics[k]}' for k in
                ('mAP50', 'mAP50_95', 'precision', 'recall') if k in final_metrics
            ))
            if per_class:
                weakest = sorted(per_class.items(),
                                 key=lambda kv: kv[1].get('ap50', 0))[:3]
                log('Weakest classes: ' + ', '.join(
                    f'{name} AP50={stats.get("ap50")}' for name, stats in weakest))
        except Exception as exc:  # noqa: BLE001 - a failed val must not lose the weights
            log(f'Final validation failed (weights are still usable): {exc}')

        # ── Does it actually detect anything? ────────────────────────────
        # The metrics above can look perfect while the weights detect nothing;
        # see worker_common.self_check for the run where that happened.
        check = self_check(best_pt, config.get('dataset_path'), img_size, log)

        # ultralytics picks best.pt by fitness, which is mostly mAP50-95 --
        # a ranking measure. An epoch whose precision has collapsed to 0.008
        # can still rank well and win, and then best.pt detects nothing while
        # the final epoch's weights work fine. Seen on an ordinary 24-image
        # run: best 0/5 at any usable threshold, last 2/5 at 0.40.
        #
        # best.pt is not silently redefined -- quietly handing back different
        # weights than the ones named is its own kind of dishonesty -- but the
        # alternative is checked and reported, because a run that produced a
        # working model and hands over a broken one is worth a sentence.
        if check and not check.get('usable') and last_pt.exists() and last_pt != best_pt:
            other = self_check(last_pt, config.get('dataset_path'), img_size, log)
            if other and other.get('usable'):
                check['alternative'] = {'path': str(last_pt), 'name': last_pt.name,
                                        **other}
                log(f'But {last_pt.name} from the same run does detect '
                    f'({other["images_with_detections"]}/'
                    f'{other["images_checked"]} images, best '
                    f'{other["best_score"]:.2f}). ultralytics chose best.pt by '
                    'a ranking score; use the last-epoch weights instead.')

        # ── Exports ──────────────────────────────────────────────────────
        exported = {'pt': str(best_pt)}
        if export_formats:
            log(f'Exporting to: {", ".join(export_formats)}')
            exporter = loader(str(best_pt))
            for fmt in export_formats:
                try:
                    if fmt == 'blob':
                        exported['blob'] = export_blob(exporter, best_pt, img_size, log)
                    else:
                        out = exporter.export(format=fmt, imgsz=img_size)
                        exported[fmt] = str(out) if out else None
                    log(f'  {fmt} -> {exported.get(fmt)}')
                except Exception as exc:  # noqa: BLE001 - one format failing is not fatal
                    exported[fmt] = None
                    log(f'  {fmt} FAILED: {exc}')

        finished_at = datetime.now().isoformat()
        final_status = 'stopped' if stopped_early else 'completed'
        update_status(config_file, {
            'status': final_status,
            'completed_at': finished_at,
            'best_model': str(best_pt),
            'run_dir': str(run_dir),
            'exported_models': exported,
            'metrics': final_metrics,
            'per_class': per_class,
            'self_check': check,
            'pid': None,
            'error': None,
        })

        current = read_config(config_file)
        append_history(project_path, {
            'model_name': model_name,
            'model_type': model_type,
            'status': final_status,
            'epochs': epochs,
            'completed_epochs': current.get('current_epoch', 0),
            'batch_size': batch_size,
            'img_size': img_size,
            'classes': config.get('classes', []),
            'base_model': config.get('base_model'),
            'train_images': config.get('train_images'),
            'val_images': config.get('val_images'),
            'metrics': final_metrics,
            'per_class': per_class,
            'self_check': check,
            'best_model': str(best_pt),
            'exported_models': exported,
            'started_at': config.get('started_at'),
            'completed_at': finished_at,
        })

        log(f'Training {final_status}.')

    except Exception as exc:  # noqa: BLE001 - report every failure through the config file
        message = str(exc) or exc.__class__.__name__
        log(f'ERROR: {message}')
        log(traceback.format_exc())
        update_status(config_file, {
            'status': 'failed',
            'error': message,
            'completed_at': datetime.now().isoformat(),
            'pid': None,
        })
        append_history(project_path, {
            'model_name': model_name,
            'model_type': model_type,
            'status': 'failed',
            'error': message,
            'epochs': epochs,
            'started_at': config.get('started_at'),
            'completed_at': datetime.now().isoformat(),
        })
        sys.exit(1)


def export_blob(exporter, best_pt, img_size, log):
    """
    Compile the model for a Luxonis OAK device.

    blobconverter needs an ONNX source and uploads it to Intel's conversion
    service, so it is done as a two-step export rather than a direct one.
    """
    try:
        import blobconverter
    except ImportError:
        raise RuntimeError(
            'blobconverter is not installed. Run: pip install blobconverter'
        )

    onnx_path = Path(exporter.export(format='onnx', imgsz=img_size, opset=12,
                                     simplify=True))
    log(f'  blob: converting {onnx_path.name} via blobconverter...')
    produced = blobconverter.from_onnx(model=str(onnx_path), data_type='FP16', shaves=6)

    target = best_pt.with_suffix('.blob')
    import shutil
    shutil.move(str(produced), str(target))
    return str(target)


if __name__ == '__main__':
    main()
