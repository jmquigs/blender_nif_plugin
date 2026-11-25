"""Check that two pyffi checkouts produce identical skin partitions.

Runs NiTriBasedGeom.update_skin_partition() on every skinned NiTriShape
of the given nifs, once with each pyffi checkout, writes the resulting
nifs, and checks that they are byte-identical and that the returned
lostweight values are equal. Prints update_skin_partition timings,
baseline vs patched.

Each pyffi checkout runs in its own subprocess (both are imported as
"pyffi"), with PYTHONHASHSEED=0 so that the header string table is
written in the same order by both runs.

Usage:
    python tools/compare_skin_partition.py \\
        --baseline /path/to/pyffi-old --patched /path/to/pyffi-new \\
        [--out DIR] [--no-synthetic] [--argsets NAME,...] file.nif [...]

A baseline checkout can be made from the pyffi submodule with e.g.
    git -C pyffi worktree add --detach /tmp/pyffi-baseline 887fe13
    git -C /tmp/pyffi-baseline submodule update --init pyffi/formats/nif/nifxml

By default a small synthetic nif is also generated (with the baseline
pyffi) and checked. It has one NiSkinInstance shape and one
BSDismemberSkinInstance shape, vertices with up to 6 bone influences and
many tied weights.

Worker mode (used internally, one pyffi checkout per process):
    python tools/compare_skin_partition.py --run PYFFI_DIR --out DIR \\
        --results results.json [--argsets ...] file.nif [...]
    python tools/compare_skin_partition.py --make-synthetic PYFFI_DIR out.nif

Exits non-zero if any output differs.
"""

import argparse
import filecmp
import json
import os
import random
import subprocess
import sys
import tempfile
import time

# the pinned pyffi uses time.clock, removed in Python 3.8
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

# the exporter's Fallout 3 arguments, see object_export.py
FO3_ARGS = dict(
    maxbonesperpartition=18, maxbonespervertex=4, stripify=False,
    stitchstrips=True, padbones=False, maximize_bone_sharing=True)


def _args(**kwargs):
    args = dict(FO3_ARGS)
    args.update(kwargs)
    return args


# name -> (update_skin_partition keyword arguments, use trianglepartmap)
ARGSETS = [
    ('fo3_map', FO3_ARGS, True),
    ('fo3_nomap', FO3_ARGS, False),
    ('stripify_true', _args(stripify=True), True),
    ('stripify_none', _args(stripify=None), True),
    ('padbones4', _args(maxbonesperpartition=4, maxbonespervertex=4,
                        padbones=True), True),
]
ARGSET_NAMES = [name for name, _, _ in ARGSETS]

# body parts for synthetic triangles that have no existing partition
SYNTHETIC_BODY_PARTS = [0, 7, 10]

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
        if isinstance(block, NifFormat.NiTriShape) and block.skin_instance:
            yield block


def build_partmap(shape, triangles):
    """Map each triangle to the body part (or partition index) of the
    existing partition that holds it.

    Triangles are keyed by their sorted vertex tuple. This only needs to
    be deterministic, so both pyffi versions get the same input.
    """
    skininst = shape.skin_instance
    skinpart = skininst.skin_partition
    if skinpart is None:
        skinpart = skininst.data.skin_partition
    dismember = isinstance(skininst, NifFormat.BSDismemberSkinInstance)
    key_to_part = {}
    if skinpart is not None:
        for i, block in enumerate(skinpart.skin_partition_blocks):
            part = skininst.partitions[i].body_part if dismember else i
            for tri in block.get_mapped_triangles():
                key_to_part.setdefault(tuple(sorted(tri)), part)
    partmap = []
    for i, tri in enumerate(triangles):
        key = tuple(sorted(tri))
        if key in key_to_part:
            partmap.append(key_to_part[key])
        else:
            # no existing partition: bands of triangles
            partmap.append(SYNTHETIC_BODY_PARTS[
                (i * len(SYNTHETIC_BODY_PARTS)) // len(triangles)])
    return partmap


def run_worker(pyffi_dir, nifs, out_dir, results_path, argset_names):
    pyffi_path = import_pyffi(pyffi_dir)
    results = {'pyffi': pyffi_path, 'files': {}}
    for nif in nifs:
        stem = os.path.splitext(os.path.basename(nif))[0]
        file_results = results['files'].setdefault(stem, {})
        for name, kwargs, use_map in ARGSETS:
            if name not in argset_names:
                continue
            # load fresh for every argument set
            data = read_nif(nif)
            shapes = []
            total = 0.0
            for shape in skinned_shapes(data):
                triangles = list(shape.data.get_triangles())
                partmap = (build_partmap(shape, triangles)
                           if use_map else None)
                start = time.perf_counter()
                lostweight = shape.update_skin_partition(
                    triangles=triangles, trianglepartmap=partmap, **kwargs)
                elapsed = time.perf_counter() - start
                total += elapsed
                skinpart = shape.skin_instance.skin_partition
                shapes.append({
                    'name': shape.name.decode('latin-1'),
                    'lostweight': repr(lostweight),
                    'partitions': skinpart.num_skin_partition_blocks,
                    'time': elapsed,
                })
            out = os.path.join(out_dir, '%s.%s.nif' % (stem, name))
            with open(out, 'wb') as stream:
                data.write(stream)
            file_results[name] = {
                'output': out, 'shapes': shapes, 'time': total}
            print('  %s %s: %.2f s' % (stem, name, total))
            sys.stdout.flush()
    with open(results_path, 'w') as stream:
        json.dump(results, stream, indent=1)


def make_synthetic(pyffi_dir, out, size=60, num_bones=24, seed=1234):
    """Write a skinned grid with one NiSkinInstance shape and one
    BSDismemberSkinInstance shape, and no skin partitions."""
    import_pyffi(pyffi_dir)
    rng = random.Random(seed)
    data = NifFormat.Data(version=0x14020007, user_version=11,
                          user_version_2=34)
    root = NifFormat.NiNode()
    root.name = b'Scene Root'
    data.roots = [root]
    bones = []
    for b in range(num_bones):
        bone = NifFormat.NiNode()
        bone.name = ('Bone%02d' % b).encode('ascii')
        root.add_child(bone)
        bones.append(bone)
    tris = []
    for y in range(size - 1):
        for x in range(size - 1):
            v = y * size + x
            tris.append((v, v + 1, v + size))
            tris.append((v + 1, v + size + 1, v + size))
    for kind in ('NiSkinInstance', 'BSDismemberSkinInstance'):
        shape = NifFormat.NiTriShape()
        shape.name = ('grid_' + kind).encode('ascii')
        root.add_child(shape)
        shapedata = NifFormat.NiTriShapeData()
        shape.data = shapedata
        shapedata.num_vertices = size * size
        shapedata.has_vertices = True
        shapedata.vertices.update_size()
        for i, vert in enumerate(shapedata.vertices):
            vert.x = float(i % size)
            vert.y = float(i // size)
            vert.z = 0.0
        shapedata.set_triangles(tris)
        skininst = getattr(NifFormat, kind)()
        skininst.data = NifFormat.NiSkinData()
        skininst.skeleton_root = root
        shape.skin_instance = skininst
        bone_weights = [{} for _ in bones]
        for v in range(size * size):
            # bones near this vertex's column, so neighbouring triangles
            # share bones
            base = ((v % size) * num_bones) // size
            pool = [b for b in range(base - 3, base + 4)
                    if 0 <= b < num_bones]
            count = min(len(pool), rng.choice([1, 1, 2, 2, 3, 4, 5, 6]))
            # quantized weights, so many vertices have tied weights
            raw = [rng.randint(1, 4) for _ in range(count)]
            for b, w in zip(rng.sample(pool, count), raw):
                bone_weights[b][v] = float(w) / sum(raw)
        for bone, vert_weights in zip(bones, bone_weights):
            if vert_weights:
                shape.add_bone(bone, vert_weights)
    with open(out, 'wb') as stream:
        data.write(stream)


def run_subprocess(args):
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = '0'
    cmd = [sys.executable, '-W', 'ignore', os.path.abspath(__file__)] + args
    subprocess.check_call(cmd, env=env)


def run_driver(opts):
    out_dir = opts.out or tempfile.mkdtemp(prefix='compare_skin_partition.')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    nifs = [os.path.abspath(n) for n in opts.nifs]
    if not opts.no_synthetic:
        synthetic = os.path.join(out_dir, 'synthetic.nif')
        run_subprocess(['--make-synthetic', opts.baseline, synthetic])
        nifs.append(synthetic)
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
                        '--argsets', opts.argsets] + nifs)
        with open(results_path) as stream:
            results[label] = json.load(stream)

    print()
    print('pyffi baseline: %s' % results['baseline']['pyffi'])
    print('pyffi patched:  %s' % results['patched']['pyffi'])
    print()
    print('%-14s %-14s %5s %9s %9s %7s  %s' % (
        'file', 'argset', 'parts', 'old (s)', 'new (s)', 'speedup',
        'result'))
    failures = 0
    old_files = results['baseline']['files']
    new_files = results['patched']['files']
    for stem in old_files:
        old_total = new_total = 0.0
        for name in ARGSET_NAMES:
            if name not in old_files[stem]:
                continue
            old = old_files[stem][name]
            new = new_files[stem][name]
            problems = []
            if not filecmp.cmp(old['output'], new['output'], shallow=False):
                problems.append('nif differs')
            old_lost = [s['lostweight'] for s in old['shapes']]
            new_lost = [s['lostweight'] for s in new['shapes']]
            if old_lost != new_lost:
                problems.append('lostweight differs: %s vs %s'
                                % (old_lost, new_lost))
            failures += bool(problems)
            parts = sum(s['partitions'] for s in new['shapes'])
            old_total += old['time']
            new_total += new['time']
            print('%-14s %-14s %5d %9.2f %9.2f %6.1fx  %s' % (
                stem, name, parts, old['time'], new['time'],
                old['time'] / max(new['time'], 1e-9),
                '; '.join(problems) or 'identical'))
        print('%-14s %-14s %5s %9.2f %9.2f %6.1fx' % (
            stem, 'total', '', old_total, new_total,
            old_total / max(new_total, 1e-9)))
    print()
    print('outputs in %s' % out_dir)
    if failures:
        print('FAIL: %d output(s) differ' % failures)
        sys.exit(1)
    print('PASS: all outputs byte-identical, lostweight equal')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0])
    parser.add_argument('--baseline', help='baseline pyffi checkout')
    parser.add_argument('--patched', help='patched pyffi checkout')
    parser.add_argument('--out', help='output directory')
    parser.add_argument('--no-synthetic', action='store_true',
                        help='do not generate and check a synthetic nif')
    parser.add_argument('--argsets', default=','.join(ARGSET_NAMES),
                        help='comma separated argument sets, from: %s'
                        % ', '.join(ARGSET_NAMES))
    parser.add_argument('--run', metavar='PYFFI_DIR',
                        help='worker: run with this pyffi checkout')
    parser.add_argument('--results', help='worker: results json path')
    parser.add_argument('--make-synthetic', nargs=2,
                        metavar=('PYFFI_DIR', 'OUT'),
                        help='worker: write the synthetic nif')
    parser.add_argument('nifs', nargs='*')
    opts = parser.parse_args()
    argset_names = opts.argsets.split(',')
    for name in argset_names:
        if name not in ARGSET_NAMES:
            parser.error('unknown argset %r' % name)
    if opts.make_synthetic:
        make_synthetic(*opts.make_synthetic)
    elif opts.run:
        run_worker(opts.run, opts.nifs, opts.out, opts.results, argset_names)
    else:
        if not (opts.baseline and opts.patched):
            parser.error('--baseline and --patched are required')
        run_driver(opts)


if __name__ == '__main__':
    main()
