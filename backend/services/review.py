"""
What a person changed about a model's predictions, and what that says.

A pre-labelled picture that somebody has corrected carries more information
than either half alone. The boxes they kept say the model was right; the ones
they moved say it was close; the ones they deleted are false positives, and
the ones they drew themselves are objects it missed entirely. Those four are
the whole of what there is to learn from, and until now none of it was
written down -- the correction landed on top of the prediction and the
prediction was gone.

This records the comparison at the moment of saving, when both versions exist.
It writes no judgement of its own: it counts, and the counting is arithmetic
rather than inference, so it can be trusted in a way a model's opinion of its
own work cannot.

That distinction is the point. A system that learns from what it predicted
gets more certain with every round and no more correct -- it grades itself
against its own answers. A system that learns from what was corrected is
being told, by somebody who looked, where it was wrong.
"""

# Two boxes over the same object, when a person has nudged one. Above this the
# correction is not worth calling a correction.
SAME_BOX = 0.9

# Below this they are not the same object at all, and the prediction counts as
# a false positive rather than a badly-placed true one.
#
# Two thresholds rather than one. Detection scoring uses 0.5 and that is right
# when the classes disagree: a different label with only partial overlap
# really is two different things. It is wrong when they agree -- a person
# dragging a box properly onto its object lands around 0.45, and calling that
# a deletion plus a fresh drawing turns one correction into two facts that are
# both false: a false positive the model never made, and a miss it did not
# have. The looser number applies only where the label already matches, so
# nothing gets paired up that was not the same object to begin with.
SAME_OBJECT = 0.5
SAME_OBJECT_SAME_TAG = 0.3


def _iou(a, b):
    ax2, ay2 = a['x'] + a['width'], a['y'] + a['height']
    bx2, by2 = b['x'] + b['width'], b['y'] + b['height']
    left, top = max(a['x'], b['x']), max(a['y'], b['y'])
    right, bottom = min(ax2, bx2), min(ay2, by2)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    union = a['width'] * a['height'] + b['width'] * b['height'] - overlap
    return overlap / union if union > 0 else 0.0


def compare(predicted, saved):
    """
    Match what the model drew against what was left behind, and name the
    differences.

    Greedy by overlap, best pair first. Greedy is the right shape here rather
    than an approximation of something better: the boxes on one picture are
    few, and a person correcting them works one object at a time, so the best
    available pairing is the pairing they meant.
    """
    predicted = list(predicted or [])
    saved = list(saved or [])

    pairs = sorted(
        ((_iou(p, s), pi, si)
         for pi, p in enumerate(predicted)
         for si, s in enumerate(saved)),
        key=lambda item: -item[0])

    taken_p, taken_s, matched = set(), set(), []
    for overlap, pi, si in pairs:
        floor = (SAME_OBJECT_SAME_TAG if predicted[pi]['tag'] == saved[si]['tag']
                 else SAME_OBJECT)
        if overlap < floor or pi in taken_p or si in taken_s:
            continue
        taken_p.add(pi)
        taken_s.add(si)
        matched.append((overlap, predicted[pi], saved[si]))

    changes = {'kept': [], 'moved': [], 'relabelled': [],
               'deleted': [], 'added': []}

    for overlap, prediction, keeper in matched:
        entry = {
            'tag': keeper['tag'],
            'predicted_tag': prediction['tag'],
            'iou': round(float(overlap), 4),
            'score': prediction.get('score'),
        }
        if prediction['tag'] != keeper['tag']:
            changes['relabelled'].append(entry)
        elif overlap >= SAME_BOX:
            changes['kept'].append(entry)
        else:
            changes['moved'].append(entry)

    for index, prediction in enumerate(predicted):
        if index not in taken_p:
            changes['deleted'].append({'tag': prediction['tag'],
                                       'predicted_tag': prediction['tag'],
                                       'iou': 0.0,
                                       'score': prediction.get('score')})
    for index, keeper in enumerate(saved):
        if index not in taken_s:
            changes['added'].append({'tag': keeper['tag'],
                                     'predicted_tag': None,
                                     'iou': 0.0, 'score': None})

    counts = {kind: len(items) for kind, items in changes.items()}
    total = sum(counts.values())
    # Of everything the model put on the picture, how much survived untouched.
    predicted_count = len(predicted)
    counts['agreement'] = (round(len(changes['kept']) / predicted_count, 4)
                           if predicted_count else None)
    return {'counts': counts, 'changes': changes, 'total': total}


def record_for(existing, saved_regions):
    """
    The review entry to store alongside a save, or None if this is not one.

    Only a picture the model labelled and a person has now saved counts. A
    picture drawn from scratch has nothing to compare against, and one saved
    twice is not corrected twice -- the second save is compared against the
    first person's work, not the model's, and would report every box as kept.
    """
    if not existing or not existing.get('auto_labelled'):
        return None
    if existing.get('review'):
        return None

    from datetime import datetime
    result = compare(existing.get('regions') or [], saved_regions or [])
    return {
        'reviewed_at': datetime.now().isoformat(),
        'model': (existing.get('auto_label') or {}).get('model'),
        'score_threshold': (existing.get('auto_label') or {}).get('score_threshold'),
        'counts': result['counts'],
        'changes': result['changes'],
    }


def summarise(entries):
    """
    Every review in a project, added up per class.

    What this answers is which classes the model is actually weak on, and in
    which way -- a class that is mostly relabelled is being confused with
    another, one that is mostly added is being missed, one that is mostly
    deleted is being imagined. Those are three different problems with three
    different fixes, and a single accuracy number tells them apart from none
    of the others.
    """
    per_class = {}
    totals = {'kept': 0, 'moved': 0, 'relabelled': 0, 'deleted': 0,
              'added': 0, 'images': 0}

    for review in entries:
        if not review:
            continue
        totals['images'] += 1
        for kind, items in (review.get('changes') or {}).items():
            if kind not in totals:
                continue
            for item in items:
                totals[kind] += 1
                # A relabelled box is a fact about the class the model chose,
                # since that is the mistake; the others are about the class the
                # box ended up as.
                tag = (item.get('predicted_tag') if kind in ('relabelled', 'deleted')
                       else item.get('tag'))
                if not tag:
                    continue
                bucket = per_class.setdefault(tag, {
                    'kept': 0, 'moved': 0, 'relabelled': 0,
                    'deleted': 0, 'added': 0})
                bucket[kind] += 1

    ranked = []
    for tag, bucket in per_class.items():
        seen = sum(bucket.values())
        wrong = bucket['relabelled'] + bucket['deleted'] + bucket['added']
        ranked.append({
            'tag': tag,
            **bucket,
            'reviewed': seen,
            # Plain arithmetic over what people actually changed. Not a
            # confidence, not a metric the model had any hand in.
            'correction_rate': round(wrong / seen, 4) if seen else 0.0,
        })
    ranked.sort(key=lambda row: (-row['correction_rate'], -row['reviewed']))

    reviewed = sum(totals[k] for k in ('kept', 'moved', 'relabelled', 'deleted'))
    return {
        'totals': totals,
        'per_class': ranked,
        'agreement': (round(totals['kept'] / reviewed, 4) if reviewed else None),
    }


def queue(name, limit=50):
    """
    The pre-labelled pictures nobody has checked yet, most useful first.

    Ordered by how unsure the model was, because a picture it was confident
    about teaches nothing whether it was right or wrong. Reviewing thirty of
    these does more than reviewing three hundred taken in the order they were
    uploaded.
    """
    from services import projects

    waiting = []
    for entry in projects.list_images(name):
        if entry.get('augmented'):
            continue
        stored = projects.read_annotation(name, entry['filename'])
        if not stored or not stored.get('auto_labelled') or stored.get('review'):
            continue
        scores = [r.get('score') for r in (stored.get('regions') or [])
                  if isinstance(r.get('score'), (int, float))]
        waiting.append({
            'filename': entry['filename'],
            'original_name': stored.get('original_name'),
            'batch': stored.get('batch'),
            'boxes': len(stored.get('regions') or []),
            'lowest_score': round(min(scores), 4) if scores else None,
            'tags': sorted({r['tag'] for r in (stored.get('regions') or [])}),
            'informativeness': informativeness(stored),
        })

    waiting.sort(key=lambda item: -item['informativeness'])
    return {'waiting': len(waiting), 'images': waiting[:max(1, int(limit))]}


def project_summary(name):
    """Every correction made in a project, added up."""
    from services import projects

    entries = []
    for entry in projects.list_images(name):
        if entry.get('augmented'):
            continue
        stored = projects.read_annotation(name, entry['filename'])
        if stored and stored.get('review'):
            entries.append(stored['review'])

    result = summarise(entries)
    result['pending'] = queue(name, limit=1)['waiting']
    return result


def informativeness(entry):
    """
    How much reviewing this picture would teach, highest first.

    The pictures worth a person's time are the ones the model was unsure
    about. A box it scored 0.95 is one it already knows; a box at 0.45 is the
    boundary it has not learned yet, and correcting that is what moves it.
    Ordering a queue by this is the difference between reviewing thirty
    pictures and reviewing three hundred.

    A picture the model found nothing in scores highest of all -- either there
    is nothing there, which takes a second to confirm, or it missed something,
    which is the most useful correction there is.
    """
    scores = [r.get('score') for r in (entry.get('regions') or [])
              if isinstance(r.get('score'), (int, float))]
    if not scores:
        return 1.0
    # Distance from certainty, taking the least certain box on the picture:
    # one doubtful box is reason enough to look.
    return round(1.0 - min(scores), 4)
