"""Check that two pyffi checkouts compute identical skin bounding spheres.

For each nif, with each pyffi checkout: load the file, zero every skinned
bone's bounding_sphere_offset and bounding_sphere_radius, call
NiGeometry.update_skin_center_radius() on every skinned shape (timed),
and write the result. The outputs of the two checkouts must be
byte-identical. The offsets and radii are also read back from both
outputs, and the largest absolute difference is printed (it should be
0.0; it helps diagnose a failing byte comparison).

Each pyffi checkout runs in its own subprocess (both are imported as
"pyffi"), with PYTHONHASHSEED=0 so that the header string table is
written in the same order by both runs.

Usage:
    python tools/compare_skin_center_radius.py \\
        --baseline /path/to/pyffi-old --patched /path/to/pyffi-new \\
        [--out DIR] [--repeat N] file.nif [...]

Worker mode (used internally, one pyffi checkout per process):
    python tools/compare_skin_center_radius.py --run PYFFI_DIR --out DIR \\
        --results results.json [--repeat N] file.nif [...]

Exits non-zero if any output differs.
"""

import argparse
import filecmp
import json
import os
import subprocess
import sys
import tempfile
import time

# the pinned pyffi uses time.clock, removed in Python 3.8
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

NifFormat = None


def import_pyffi(pyffi_dir):
    global NifFormat
    sys.path.insert(0, os.path.abspath(pyffi_dir))
    from pyffi.formats.nif import NifFormat as nif_format
    NifFormat = nif_format
    import pyffi
    return os.path.dirname(os.path.abspath(pyffi.__file__))


def read_nif(path):
    data = NifFormat.Data()
    with open(path, 'rb') as stream:
        data.read(stream)
    return data


def skinned_shapes(data):
    for block in data.blocks:
        if isinstance(block, NifFormat.NiGeometry) and block.skin_instance:
            yield block


def zero_spheres(shapes):
    for shape in shapes:
        for bone in shape.skin_instance.data.bone_list:
            bone.bounding_sphere_offset.x = 0.0
            bone.bounding_sphere_offset.y = 0.0
            bone.bounding_sphere_offset.z = 0.0
            bone.bounding_sphere_radius = 0.0


def spheres(data):
    """[(offset x, y, z, radius)] for every bone of every skinned shape."""
    return [(bone.bounding_sphere_offset.x, bone.bounding_sphere_offset.y,
             bone.bounding_sphere_offset.z, bone.bounding_sphere_radius)
            for shape in skinned_shapes(data)
            for bone in shape.skin_instance.data.bone_list]


def run_worker(pyffi_dir, nifs, out_dir, results_path, repeat):
    pyffi_path = import_pyffi(pyffi_dir)
    results = {'pyffi': pyffi_path, 'files': {}}
    for nif in nifs:
        stem = os.path.splitext(os.path.basename(nif))[0]
        data = read_nif(nif)
        shapes = list(skinned_shapes(data))
        # best of several runs; every run starts from zeroed spheres and
        # computes the same values
        best = None
        for _ in range(repeat):
            zero_spheres(shapes)
            total = 0.0
            for shape in shapes:
                start = time.perf_counter()
                shape.update_skin_center_radius()
                total += time.perf_counter() - start
            best = total if best is None else min(best, total)
        out = os.path.join(out_dir, stem + '.nif')
        with open(out, 'wb') as stream:
            data.write(stream)
        results['files'][stem] = {
            'output': out,
            'time': best,
            'shapes': len(shapes),
            'bones': sum(shape.skin_instance.data.num_bones
                         for shape in shapes),
            # read back from the written file
            'spheres': spheres(read_nif(out)),
        }
        print('  %s: %.3f s' % (stem, best))
        sys.stdout.flush()
    with open(results_path, 'w') as stream:
        json.dump(results, stream, indent=1)


def run_subprocess(args):
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = '0'
    cmd = [sys.executable, '-W', 'ignore', os.path.abspath(__file__)] + args
    subprocess.check_call(cmd, env=env)


def run_driver(opts):
    out_dir = opts.out or tempfile.mkdtemp(
        prefix='compare_skin_center_radius.')
    nifs = [os.path.abspath(n) for n in opts.nifs]
    if not nifs:
        sys.exit('no input nifs')
    results = {}
    for label, pyffi_dir in (('baseline', opts.baseline),
                             ('patched', opts.patched)):
        run_out = os.path.join(out_dir, label)
        if not os.path.isdir(run_out):
            os.makedirs(run_out)
        results_path = os.path.join(out_dir, label + '.json')
        print('%s: %s' % (label, pyffi_dir))
        sys.stdout.flush()
        run_subprocess(['--run', pyffi_dir, '--out', run_out,
                        '--results', results_path,
                        '--repeat', str(opts.repeat)] + nifs)
        with open(results_path) as stream:
            results[label] = json.load(stream)

    print()
    print('pyffi baseline: %s' % results['baseline']['pyffi'])
    print('pyffi patched:  %s' % results['patched']['pyffi'])
    print('times: best of %d' % opts.repeat)
    print()
    print('%-14s %6s %5s %9s %9s %7s %10s %10s  %s' % (
        'file', 'shapes', 'bones', 'old (s)', 'new (s)', 'speedup',
        'max d off', 'max d rad', 'result'))
    failures = 0
    for stem, old in results['baseline']['files'].items():
        new = results['patched']['files'][stem]
        problems = []
        if not filecmp.cmp(old['output'], new['output'], shallow=False):
            problems.append('nif differs')
        if len(old['spheres']) != len(new['spheres']):
            problems.append('bone count differs')
        max_off = max_rad = 0.0
        for a, b in zip(old['spheres'], new['spheres']):
            max_off = max(max_off, max(abs(x - y) for x, y in
                                       zip(a[:3], b[:3])))
            max_rad = max(max_rad, abs(a[3] - b[3]))
        failures += bool(problems)
        print('%-14s %6d %5d %9.3f %9.3f %6.1fx %10.3g %10.3g  %s' % (
            stem, new['shapes'], new['bones'], old['time'], new['time'],
            old['time'] / max(new['time'], 1e-9), max_off, max_rad,
            '; '.join(problems) or 'identical'))
    print()
    print('outputs in %s' % out_dir)
    if failures:
        print('FAIL: %d output(s) differ' % failures)
        sys.exit(1)
    print('PASS: all outputs byte-identical')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0])
    parser.add_argument('--baseline', help='baseline pyffi checkout')
    parser.add_argument('--patched', help='patched pyffi checkout')
    parser.add_argument('--out', help='output directory')
    parser.add_argument('--repeat', type=int, default=3,
                        help='time the best of this many runs (default 3)')
    parser.add_argument('--run', metavar='PYFFI_DIR',
                        help='worker: run with this pyffi checkout')
    parser.add_argument('--results', help='worker: results json path')
    parser.add_argument('nifs', nargs='*')
    opts = parser.parse_args()
    if opts.run:
        run_worker(opts.run, opts.nifs, opts.out, opts.results, opts.repeat)
    else:
        if not (opts.baseline and opts.patched):
            parser.error('--baseline and --patched are required')
        run_driver(opts)


if __name__ == '__main__':
    main()
