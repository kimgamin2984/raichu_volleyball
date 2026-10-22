"""Publish local training files without losing snapshots to Windows locks."""

import glob
import os
import tempfile
import time


def _replace_with_retry(source, destination):
    """Retry access/sharing violations while leaving other I/O errors visible."""
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.5))


def resolve_saved_path(path):
    """Find the newest complete publication, including a deferred replacement."""
    path = os.path.abspath(path)
    candidates = [path] + glob.glob(glob.escape(path) + '.pending-*')
    available = []
    for candidate in candidates:
        try:
            available.append((os.stat(candidate).st_mtime_ns, candidate))
        except FileNotFoundError:
            continue
    return max(available)[1] if available else path


def atomic_write(path, writer):
    """Write privately, then publish or retain a complete resumable snapshot.

    A ``.pending-*`` name is assigned only after the writer has closed the
    completed file. Readers can therefore select it if Windows prevents
    replacing the usual filename. An incomplete ``.writing-*`` is never read.
    """
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    suffix = os.path.splitext(path)[1]
    fd, writing_path = tempfile.mkstemp(
        prefix='.' + os.path.basename(path) + '.writing-', suffix=suffix,
        dir=parent,
    )
    os.close(fd)
    pending_path = path + '.pending-' + os.path.basename(writing_path)
    completed = False
    try:
        writer(writing_path)
        completed = True
        # The destination is unique, so it cannot be held by a prior reader.
        _replace_with_retry(writing_path, pending_path)
        try:
            _replace_with_retry(pending_path, path)
        except PermissionError:
            print(f'Could not replace locked file {path}; kept complete recovery '
                  f'file {pending_path}.', flush=True)
            return pending_path
        return path
    except BaseException:
        if completed and os.path.exists(writing_path):
            print(f'Completed file retained at {writing_path}.', flush=True)
        raise
    finally:
        if not completed:
            try:
                os.remove(writing_path)
            except OSError:
                pass
