"""
Say what is actually inside a detector ONNX.

When a model built by other tooling will not run, the useful question is not
what the error said but what shape the file hands back, and that is two lines
of onnxruntime nobody should have to write from memory:

    python backend/tools/inspect_onnx.py path/to/model.onnx

It prints the inputs, the outputs, the metadata, and which of the layouts the
application knows how to read this one looks like. Reporting that is enough to
settle a model that still will not run; guessing from the error message is not.

It does not need the server, or a project, or an image.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def shape_of(node):
    return [d if isinstance(d, int) else str(d) for d in (node.shape or [])]


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip())
        return 2

    path = Path(argv[1])
    if not path.is_file():
        print(f'No such file: {path}')
        return 2

    try:
        import onnxruntime as ort
    except ImportError:
        print('onnxruntime is not installed. pip install onnxruntime')
        return 2

    try:
        session = ort.InferenceSession(str(path),
                                       providers=['CPUExecutionProvider'])
    except Exception as exc:  # noqa: BLE001
        print(f'{path.name} could not be opened: {exc}')
        return 1

    print(f'{path.name}  ({path.stat().st_size / 1e6:.1f} MB)')

    print('\ninputs')
    for node in session.get_inputs():
        print(f'  {node.name:20} {shape_of(node)}  {node.type}')

    print('\noutputs')
    for node in session.get_outputs():
        print(f'  {node.name:20} {shape_of(node)}  {node.type}')

    meta = session.get_modelmeta()
    custom = dict(meta.custom_metadata_map or {})
    print('\nmetadata')
    if custom:
        for key in sorted(custom):
            print(f'  {key:20} {custom[key][:160]}')
    else:
        print('  (none -- this is why ultralytics cannot name the classes)')

    # Metadata is what ultralytics trusts and what trips it up, so the two
    # fields it reads are worth calling out by name.
    for key in ('imgsz', 'names'):
        if key in custom:
            value = custom[key]
            print(f'\n{key} reads as: {value[:200]}')

    print('\nrun it, and see what comes back')
    try:
        import numpy as np
        from services.onnxrunner import OnnxDetector

        detector = OnnxDetector(path, display_name=path.name)
        frame = np.zeros((480, 640, 3), np.uint8)
        found = detector.predict(frame, threshold=0.01)
        print(f'  read as the {detector.layout} layout, '
              f'{len(found)} detection(s) on a blank frame')
        print('  this file will run in the Test Model page.')
    except Exception as exc:  # noqa: BLE001
        print(f'  {type(exc).__name__}: {exc}')
        print('  send this whole output along and the layout can be added.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
