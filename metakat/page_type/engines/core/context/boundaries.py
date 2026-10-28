"""Five-bit scan and optional complete document metadata contract."""
import numpy as np


def encode_boundary_features(n, scan_start=False, scan_end=False,
                             document_starts=None, document_ends=None):
    """[scan start, scan end, metadata provided, document start, document end].

    Absent metadata is never inferred from training targets. Supplied metadata
    must describe both tasks on every page; partial metadata is unsupported.
    """
    if n < 1:
        raise ValueError('nonempty page sequence required')
    values = np.zeros((n, 5), np.float32)
    values[0, 0], values[-1, 1] = scan_start, scan_end
    if (document_starts is None) != (document_ends is None):
        raise ValueError('complete start and end metadata must be supplied together')
    if document_starts is not None:
        starts, ends = np.asarray(document_starts), np.asarray(document_ends)
        if (starts.shape != (n,) or ends.shape != (n,)
                or not np.isin(starts, [0, 1]).all() or not np.isin(ends, [0, 1]).all()):
            raise ValueError('binary document metadata required for every page')
        values[:, 2], values[:, 3], values[:, 4] = 1, starts, ends
    return values

