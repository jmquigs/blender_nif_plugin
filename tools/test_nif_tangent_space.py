"""Unit tests for io_scene_nif/utility/nif_tangent_space.py.

Run on a normal system Python with numpy, from the repository root:

    python -m unittest tools/test_nif_tangent_space.py

The tests that compare against pyffi's update_tangent_space import pyffi
from the submodule (git submodule update --init --recursive); they are
skipped if it is not available.
"""

import os
import struct
import sys
import time
import unittest

# the pinned pyffi uses time.clock, removed in Python 3.8
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pyffi'))
# import the module directly: importing the io_scene_nif package needs bpy
sys.path.insert(0, os.path.join(ROOT, 'io_scene_nif', 'utility'))

import numpy as np  # noqa: E402
import nif_tangent_space as nts  # noqa: E402

try:
    from pyffi.formats.nif import NifFormat
except ImportError:
    NifFormat = None

needs_pyffi = unittest.skipIf(NifFormat is None, 'pyffi not available')


def make_shape(verts, norms, uvs, tris, vcols=None):
    """Build a pyffi NiTriShape (Fallout 3 style) from arrays."""
    shape = NifFormat.NiTriShape()
    data = NifFormat.NiTriShapeData()
    shape.data = data
    data.num_vertices = len(verts)
    data.has_vertices = True
    data.vertices.update_size()
    for vec, (x, y, z) in zip(data.vertices, np.asarray(verts).tolist()):
        vec.x, vec.y, vec.z = x, y, z
    data.has_normals = True
    data.normals.update_size()
    for vec, (x, y, z) in zip(data.normals, np.asarray(norms).tolist()):
        vec.x, vec.y, vec.z = x, y, z
    if uvs is not None:
        data.num_uv_sets = 1
        data.bs_num_uv_sets = 1
        data.has_uv = True
        data.uv_sets.update_size()
        for uv, (u, v) in zip(data.uv_sets[0], np.asarray(uvs).tolist()):
            uv.u, uv.v = u, v
    if vcols is not None:
        data.has_vertex_colors = True
        data.vertex_colors.update_size()
        for col, (r, g, b, a) in zip(data.vertex_colors,
                                     np.asarray(vcols).tolist()):
            col.r, col.g, col.b, col.a = r, g, b, a
    data.set_triangles([tuple(t) for t in np.asarray(tris).tolist()])
    return shape


def vec_array(vecs):
    return np.array([v.as_tuple() for v in vecs],
                    dtype=np.float64).reshape(-1, 3)


class TestComputeTangentSpace(unittest.TestCase):

    def test_single_triangle(self):
        verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        norms = [(0, 0, 1)] * 3
        uvs = [(0, 0), (1, 0), (0, 1)]
        tan, bin_, nrm = nts.compute_tangent_space(
            verts, norms, uvs, [(0, 1, 2)])
        np.testing.assert_allclose(tan, [(0, 1, 0)] * 3, atol=1e-12)
        np.testing.assert_allclose(bin_, [(1, 0, 0)] * 3, atol=1e-12)
        np.testing.assert_allclose(nrm, norms, atol=1e-12)

    def test_uv_seam_shares_frame(self):
        # a quad in the xy plane, split along the diagonal 0-2, with the
        # two triangles in different (mirrored) uv islands, so the
        # diagonal vertices are duplicated (4, 5) with different uvs
        verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                 (0, 0, 0), (1, 1, 0)]
        norms = [(0, 0, 1)] * 6
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1),
               (5, 5), (5.5, 4.0)]
        tris = [(0, 1, 2), (4, 5, 3)]
        stats = {}
        tan, bin_, _ = nts.compute_tangent_space(
            verts, norms, uvs, tris, stats=stats)
        self.assertEqual(stats['num_groups'], 4)
        # duplicated vertices share the frame of their twin
        np.testing.assert_array_equal(tan[0], tan[4])
        np.testing.assert_array_equal(tan[2], tan[5])
        np.testing.assert_array_equal(bin_[0], bin_[4])
        np.testing.assert_array_equal(bin_[2], bin_[5])
        # and the seam vertices see both triangles, the others only one
        self.assertFalse(np.allclose(tan[0], tan[1]))
        # all frames orthonormal and in the plane
        for vecs in (tan, bin_):
            np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0)
            np.testing.assert_allclose(vecs[:, 2], 0.0, atol=1e-12)

    def test_zero_area_uv_triangle_skipped(self):
        norms = [(0, 0, 1)] * 5
        good = nts.compute_tangent_space(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)], norms[:3],
            [(0, 0), (1, 0), (0, 1)], [(0, 1, 2)])
        # add triangle (1, 3, 4) whose three uvs coincide: zero area in
        # uv space, sdir and tdir are zero, so pyffi skips it
        # (note: a triangle with collinear but distinct uvs also has zero
        # area, but its sdir/tdir are not zero and pyffi does use it)
        verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 2, 0), (2, 0, 0)]
        uvs = [(0, 0), (1, 0), (0, 1), (1, 0), (1, 0)]
        stats = {}
        tan, bin_, _ = nts.compute_tangent_space(
            verts, norms, uvs, [(0, 1, 2), (1, 3, 4)], stats=stats)
        # the vertices of the good triangle are unaffected
        np.testing.assert_array_equal(tan[:3], good[0])
        np.testing.assert_array_equal(bin_[:3], good[1])
        # vertices 3, 4 only have the skipped triangle: no contributions
        np.testing.assert_array_equal(tan[3:], 0.0)
        np.testing.assert_array_equal(bin_[3:], 0.0)
        self.assertEqual(stats['num_degenerate_groups'], 2)

    def test_collinear_uv_triangle_is_used(self):
        # pyffi quirk: r == 0 gives sign +1 and the triangle is kept
        tan, bin_, _ = nts.compute_tangent_space(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 0, 1)] * 3,
            [(0, 0), (1, 1), (2, 2)], [(0, 1, 2)])
        self.assertTrue(np.all(np.any(tan != 0, axis=1)))
        self.assertTrue(np.all(np.any(bin_ != 0, axis=1)))

    def test_zero_normal(self):
        verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        norms = [(0, 0, 0), (0, 0, 2), (0, 0, 1)]
        uvs = [(0, 0), (1, 0), (0, 1)]
        _, _, nrm = nts.compute_tangent_space(verts, norms, uvs, [(0, 1, 2)])
        np.testing.assert_array_equal(nrm, [(0, 0, 0), (0, 0, 1), (0, 0, 1)])

    def test_float_to_int(self):
        values = np.array([0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 2.5, -2.5,
                           np.nan, np.inf, -np.inf])
        expected = [0, 0, 1, -1, 1, -1, 3, -3, 0, 2 ** 31, -2 ** 31]
        np.testing.assert_array_equal(nts._float_to_int(values), expected)

    def test_group_rows(self):
        keys = np.array([[1, 2], [0, 0], [1, 2], [0, 1], [0, 0]], float)
        inverse, num = nts._group_rows(keys)
        self.assertEqual(num, 3)
        self.assertEqual(inverse[0], inverse[2])
        self.assertEqual(inverse[1], inverse[4])
        self.assertEqual(len(set(inverse[[0, 1, 3]])), 3)


@needs_pyffi
class TestAgainstPyffi(unittest.TestCase):

    def check(self, verts, norms, uvs, tris, vcols=None):
        shape = make_shape(verts, norms, uvs, tris, vcols)
        geom = nts.read_geometry(shape.data)
        stats = {}
        tan, bin_, nrm = nts.compute_tangent_space(
            geom['verts'], geom['norms'], geom['uvs'], geom['tris'],
            hash_uvs=geom['hash_uvs'], vcols=geom['vcols'], stats=stats)
        shape.update_tangent_space(as_extra=False)
        old_tan = vec_array(shape.data.tangents)
        old_bin = vec_array(shape.data.bitangents)
        old_nrm = vec_array(shape.data.normals)
        np.testing.assert_array_equal(tan, old_tan)
        np.testing.assert_array_equal(bin_, old_bin)
        np.testing.assert_array_equal(nrm, old_nrm)
        return stats

    def test_single_triangle(self):
        self.check([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 0, 1)] * 3,
                   [(0, 0), (1, 0), (0, 1)], [(0, 1, 2)])

    def test_random_meshes_with_degenerate_cases(self):
        rng = np.random.RandomState(1234)
        num_degenerate = 0
        for trial in range(40):
            num = rng.randint(3, 120)
            # coarse coordinates so that many vertices hash together
            verts = rng.randint(-3, 4, size=(num, 3)) * 0.5
            norms = rng.randn(num, 3)
            norms[rng.rand(num) < 0.3] = (0, 0, 1)
            norms[rng.rand(num) < 0.05] = 0.0  # zero normals
            # some duplicates whose normal differs below the hash
            # precision, so a group mixes slightly different normals
            dup = rng.rand(num) < 0.3
            src = rng.randint(0, num, size=num)
            verts[dup] = verts[src[dup]]
            norms[dup] = norms[src[dup]] + rng.uniform(
                -4e-4, 4e-4, size=(int(dup.sum()), 3))
            uvs = rng.randint(0, 4, size=(num, 2)) * 0.25
            uvs[rng.rand(num) < 0.1] = (0.5, 0.5)
            num_tris = rng.randint(1, 3 * num)
            tris = rng.randint(0, num, size=(num_tris, 3))
            vcols = rng.rand(num, 4) if trial % 2 else None
            stats = self.check(verts, norms, uvs, tris, vcols)
            num_degenerate += stats['num_degenerate_groups']
        # make sure the degenerate code paths were exercised
        self.assertGreater(num_degenerate, 0)

    def test_tangent_degenerate_keeps_raw_tangent(self):
        # every contributing uv direction maps to the normal direction:
        # the triangle lies in the xz plane but the normal says +x, so
        # the bitangent projects to zero and the tangent stays raw
        verts = [(0, 0, 0), (1, 0, 0), (0, 0, 1)]
        norms = [(1, 0, 0)] * 3
        uvs = [(0, 0), (1, 0), (0, 1)]
        stats = self.check(verts, norms, uvs, [(0, 1, 2)])
        self.assertEqual(stats['num_degenerate_groups'], 3)
        tan, bin_, _ = nts.compute_tangent_space(
            verts, norms, uvs, [(0, 1, 2)])
        np.testing.assert_array_equal(bin_, 0.0)
        np.testing.assert_array_equal(tan, [(0, 0, 1)] * 3)

    def test_apply_extra_data(self):
        verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        norms = [(0, 0, 1)] * 3
        uvs = [(0, 0), (1, 0), (0.3, 1)]
        for as_extra in (True, None):
            old = make_shape(verts, norms, uvs, [(0, 1, 2)])
            new = make_shape(verts, norms, uvs, [(0, 1, 2)])
            if as_extra is None:
                # autodetect: existing extra data means oblivion style
                for shape in (old, new):
                    extra = NifFormat.NiBinaryExtraData()
                    extra.name = nts.TANGENT_SPACE_EXTRA_NAME
                    shape.add_extra_data(extra)
            old.update_tangent_space(as_extra=as_extra)
            nts.update_tangent_space(new, as_extra=as_extra)
            old_extra = nts.find_tangent_space_extra(old)
            new_extra = nts.find_tangent_space_extra(new)
            self.assertEqual(len(old.get_extra_datas()),
                             len(new.get_extra_datas()))
            self.assertEqual(old_extra.binary_data, new_extra.binary_data)
            self.assertEqual(len(new_extra.binary_data), 2 * 3 * 12)
            self.assertEqual(old.data.num_uv_sets, new.data.num_uv_sets)
            # data tangents untouched
            self.assertEqual(new.data.num_uv_sets & 4096, 0)
            values = struct.unpack('<18f', new_extra.binary_data)
            self.assertAlmostEqual(values[2], 0.0)

    def test_apply_fallout3_flags(self):
        verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        norms = [(0, 0, 1)] * 3
        uvs = [(0, 0), (1, 0), (0, 1)]
        old = make_shape(verts, norms, uvs, [(0, 1, 2)])
        new = make_shape(verts, norms, uvs, [(0, 1, 2)])
        old.update_tangent_space()
        nts.update_tangent_space(new)
        self.assertEqual(old.data.num_uv_sets, new.data.num_uv_sets)
        self.assertEqual(old.data.bs_num_uv_sets, new.data.bs_num_uv_sets)
        self.assertTrue(new.data.bs_num_uv_sets & 4096)
        np.testing.assert_array_equal(vec_array(old.data.tangents),
                                      vec_array(new.data.tangents))

    def test_no_uv_sets_clears_flag(self):
        verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        shape = make_shape(verts, [(0, 0, 1)] * 3, None, [(0, 1, 2)])
        # tangent flag set, but no uv sets (bs_num_uv_sets & 1 == 0)
        shape.data.num_uv_sets = 4096
        shape.data.bs_num_uv_sets = 4096
        self.assertEqual(len(shape.data.uv_sets), 0)
        self.assertIsNone(nts.update_tangent_space(shape))
        self.assertEqual(shape.data.num_uv_sets, 0)
        self.assertEqual(shape.data.bs_num_uv_sets, 0)


if __name__ == '__main__':
    unittest.main()
