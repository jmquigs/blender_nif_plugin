"""Compare the numpy tangent space against pyffi's update_tangent_space.

Runs on a normal system Python with numpy; pyffi is imported from the
submodule in this repository. For every NiTriShape with normals and UVs
in the given nifs, the tangent space is computed both ways from the same
data and compared. The tangents stored in the file are not used: they
may have been written by another tool (e.g. NifSkope) with a different
algorithm.

Usage:
    python tools/compare_tangents.py file.nif [file.nif ...]
    python tools/compare_tangents.py --two-files old.nif new.nif

The second form compares the tangents stored in two nifs, shape by shape
(e.g. two exports of the same scene with USE_NUMPY_TANGENTS off and on).

Exits non-zero on failure.
"""

import argparse
import os
import sys
import time

# the pinned pyffi uses time.clock, removed in Python 3.8
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pyffi'))
# import the module directly: importing the io_scene_nif package needs bpy
sys.path.insert(0, os.path.join(ROOT, 'io_scene_nif', 'utility'))

import numpy as np  # noqa: E402
from pyffi.formats.nif import NifFormat  # noqa: E402
import nif_tangent_space as nts  # noqa: E402

TOLERANCE = 1e-5


def read_nif(path):
    data = NifFormat.Data()
    with open(path, 'rb') as stream:
        data.read(stream)
    return data


def iter_shapes(data):
    for block in data.blocks:
        if isinstance(block, NifFormat.NiTriShape):
            yield block


def vec_array(vecs):
    return np.array([v.as_tuple() for v in vecs],
                    dtype=np.float64).reshape(-1, 3)


def angle_deg(a, b):
    """Angle between rows of a and b, in degrees; nan where either is 0."""
    na = np.sqrt(np.einsum('ij,ij->i', a, a))
    nb = np.sqrt(np.einsum('ij,ij->i', b, b))
    with np.errstate(all='ignore'):
        cos = np.einsum('ij,ij->i', a, b) / (na * nb)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def nanmax(values):
    values = values[~np.isnan(values)]
    return float(values.max()) if values.size else 0.0


def is_zero(vecs):
    return ~np.any(vecs != 0, axis=1)


def compare_shape(shape, label):
    """Compare old and new on one shape. Returns True on pass."""
    shape_data = shape.data
    if not shape_data.has_normals or len(shape_data.uv_sets) == 0:
        print('%s: skipped (no normals or no uvs)' % label)
        return True

    geom = nts.read_geometry(shape_data)
    stats = {}
    t0 = time.perf_counter()
    new_tan, new_bin, new_norms = nts.compute_tangent_space(
        geom['verts'].copy(), geom['norms'].copy(), geom['uvs'].copy(),
        geom['tris'].copy(), hash_uvs=geom['hash_uvs'], vcols=geom['vcols'],
        stats=stats)
    new_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    shape.update_tangent_space(as_extra=False)
    old_time = time.perf_counter() - t0
    old_tan = vec_array(shape_data.tangents)
    old_bin = vec_array(shape_data.bitangents)
    old_norms = vec_array(shape_data.normals)

    # now overwrite with the numpy result through apply_tangent_space (the
    # pyffi write stage, which the exporter pays too) and check that it
    # stores exactly what was computed
    t0 = time.perf_counter()
    nts.apply_tangent_space(shape, new_tan, new_bin, as_extra=False)
    write_time = time.perf_counter() - t0
    written_ok = (np.array_equal(vec_array(shape_data.tangents), new_tan)
                  and np.array_equal(vec_array(shape_data.bitangents),
                                     new_bin))

    degen = stats['degenerate_vertices']
    ok_rows = ~degen
    err_tan = np.abs(new_tan - old_tan)
    err_bin = np.abs(new_bin - old_bin)
    err_norm = np.abs(new_norms - old_norms)
    max_tan = float(err_tan[ok_rows].max()) if ok_rows.any() else 0.0
    max_bin = float(err_bin[ok_rows].max()) if ok_rows.any() else 0.0
    max_tan_degen = float(err_tan[degen].max()) if degen.any() else 0.0
    max_bin_degen = float(err_bin[degen].max()) if degen.any() else 0.0
    max_norm = float(err_norm.max()) if err_norm.size else 0.0
    exact_tan = int(np.count_nonzero(np.all(new_tan == old_tan, axis=1)))
    exact_bin = int(np.count_nonzero(np.all(new_bin == old_bin, axis=1)))
    zero_counts = (int(is_zero(old_tan).sum()), int(is_zero(new_tan).sum()),
                   int(is_zero(old_bin).sum()), int(is_zero(new_bin).sum()))

    passed = (max_tan <= TOLERANCE and max_bin <= TOLERANCE
              and max_tan_degen <= TOLERANCE and max_bin_degen <= TOLERANCE
              and max_norm <= TOLERANCE
              and zero_counts[0] == zero_counts[1]
              and zero_counts[2] == zero_counts[3]
              and written_ok)

    num_verts = geom['verts'].shape[0]
    print('%s: %s' % (label, 'PASS' if passed else 'FAIL'))
    print('  verts %d, tris %d, groups %d, degenerate groups %d '
          '(%d verts)' % (num_verts, geom['tris'].shape[0],
                          stats['num_groups'],
                          stats['num_degenerate_groups'],
                          int(degen.sum())))
    print('  tangents:   max err %.3g, max angle %.3g deg, '
          'bit-exact %d/%d, zero old/new %d/%d'
          % (max_tan, nanmax(angle_deg(new_tan, old_tan)[ok_rows]),
             exact_tan, num_verts, zero_counts[0], zero_counts[1]))
    print('  bitangents: max err %.3g, max angle %.3g deg, '
          'bit-exact %d/%d, zero old/new %d/%d'
          % (max_bin, nanmax(angle_deg(new_bin, old_bin)[ok_rows]),
             exact_bin, num_verts, zero_counts[2], zero_counts[3]))
    print('  degenerate verts: max err tan %.3g, bin %.3g'
          % (max_tan_degen, max_bin_degen))
    print('  normals: max err %.3g' % max_norm)
    print('  time: pyffi %.4f s, numpy %.4f s + write %.4f s (%.1fx)'
          % (old_time, new_time, write_time,
             old_time / max(new_time + write_time, 1e-9)))
    if not written_ok:
        print('  apply_tangent_space wrote different values')
    return passed, old_time, new_time + write_time


def compare_files(paths):
    all_passed = True
    total_old = total_new = 0.0
    for path in paths:
        data = read_nif(path)
        for i, shape in enumerate(iter_shapes(data)):
            label = '%s [%d] %s' % (os.path.basename(path), i,
                                    shape.name.decode('ascii', 'replace'))
            result = compare_shape(shape, label)
            if result is True:
                continue
            passed, old_time, new_time = result
            all_passed &= passed
            total_old += old_time
            total_new += new_time
    print('total time: pyffi %.3f s, numpy %.3f s' % (total_old, total_new))
    print('OVERALL: %s' % ('PASS' if all_passed else 'FAIL'))
    return all_passed


def compare_two_files(path_a, path_b):
    """Compare stored tangents/bitangents of two nifs shape by shape."""
    shapes_a = list(iter_shapes(read_nif(path_a)))
    shapes_b = list(iter_shapes(read_nif(path_b)))
    if len(shapes_a) != len(shapes_b):
        print('FAIL: shape count differs (%d vs %d)'
              % (len(shapes_a), len(shapes_b)))
        return False
    all_passed = True
    for i, (a, b) in enumerate(zip(shapes_a, shapes_b)):
        label = '[%d] %s' % (i, a.name.decode('ascii', 'replace'))
        if a.data.num_vertices != b.data.num_vertices:
            print('%s: FAIL vertex count differs' % label)
            all_passed = False
            continue
        passed = True
        for attr in ('tangents', 'bitangents', 'normals'):
            va = vec_array(getattr(a.data, attr))
            vb = vec_array(getattr(b.data, attr))
            if va.shape != vb.shape:
                print('%s: %s FAIL shape %s vs %s'
                      % (label, attr, va.shape, vb.shape))
                passed = False
                continue
            err = float(np.abs(va - vb).max()) if va.size else 0.0
            zeros = (int(is_zero(va).sum()), int(is_zero(vb).sum()))
            ok = err <= TOLERANCE and zeros[0] == zeros[1]
            passed &= ok
            print('%s: %s max err %.3g, zero %d/%d %s'
                  % (label, attr, err, zeros[0], zeros[1],
                     'ok' if ok else 'FAIL'))
        flags = (a.data.num_uv_sets, b.data.num_uv_sets,
                 a.data.bs_num_uv_sets, b.data.bs_num_uv_sets)
        if flags[0] != flags[1] or flags[2] != flags[3]:
            print('%s: uv set flags differ %r' % (label, flags))
            passed = False
        all_passed &= passed
    print('OVERALL: %s' % ('PASS' if all_passed else 'FAIL'))
    return all_passed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('nifs', nargs='+')
    parser.add_argument('--two-files', action='store_true',
                        help='compare stored tangents of two nifs')
    args = parser.parse_args(argv)
    if args.two_files:
        if len(args.nifs) != 2:
            parser.error('--two-files takes exactly two nifs')
        ok = compare_two_files(*args.nifs)
    else:
        ok = compare_files(args.nifs)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
