"""Unit tests for pyffi's NiGeometry.update_skin_center_radius().

Checks that the numpy implementation gives bit-identical results to the
original pure-Python code (which pyffi runs when numpy cannot be
imported), and checks a few results against hand-computed values.

Run on a normal system Python with numpy, from the repository root:

    python -m unittest tools/test_skin_center_radius.py

pyffi is imported from the submodule (git submodule update --init
--recursive); the tests are skipped if it is not available.
"""

import math
import os
import struct
import sys
import time
import unittest
from unittest import mock

# the pinned pyffi uses time.clock, removed in Python 3.8
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pyffi'))

try:
    from pyffi.formats.nif import NifFormat
except ImportError:
    NifFormat = None

needs_pyffi = unittest.skipIf(NifFormat is None, 'pyffi not available')

IDENTITY = (0.0, 1.0, (0.0, 0.0, 0.0))
# rotation about z, scale and translation
TRANSFORM = (0.7, 1.5, (2.0, -3.0, 0.25))


def make_shape(verts, bones):
    """Build a skinned NiTriShape. Returns the shape and its skeleton root
    (the skin instance only holds a weak reference to the root).

    :param verts: list of (x, y, z)
    :param bones: list of (vertex indices, transform), where transform is
        (angle about z, scale, translation). Indices are stored as given,
        duplicates included.
    """
    shape = NifFormat.NiTriShape()
    data = NifFormat.NiTriShapeData()
    shape.data = data
    data.num_vertices = len(verts)
    data.has_vertices = True
    data.vertices.update_size()
    for vert, (x, y, z) in zip(data.vertices, verts):
        vert.x, vert.y, vert.z = x, y, z
    root = NifFormat.NiNode()
    skininst = NifFormat.NiSkinInstance()
    skindata = NifFormat.NiSkinData()
    skininst.data = skindata
    skininst.skeleton_root = root
    shape.skin_instance = skininst
    skininst.num_bones = len(bones)
    skininst.bones.update_size()
    skindata.num_bones = len(bones)
    skindata.bone_list.update_size()
    for i, (indices, (angle, scale, translation)) in enumerate(bones):
        bone = NifFormat.NiNode()
        root.add_child(bone)
        skininst.bones[i] = bone
        block = skindata.bone_list[i]
        block.num_vertices = len(indices)
        block.vertex_weights.update_size()
        for weight, index in zip(block.vertex_weights, indices):
            weight.index = index
            weight.weight = 1.0
        transform = block.skin_transform
        cos, sin = math.cos(angle), math.sin(angle)
        rot = transform.rotation
        rot.m_11, rot.m_12, rot.m_13 = cos, sin, 0.0
        # 0.0 - sin rather than -sin, so angle 0 gives +0.0, not -0.0
        rot.m_21, rot.m_22, rot.m_23 = 0.0 - sin, cos, 0.0
        rot.m_31, rot.m_32, rot.m_33 = 0.0, 0.0, 1.0
        transform.translation.x, transform.translation.y, \
            transform.translation.z = translation
        transform.scale = scale
    return shape, root


def run(verts, bones, use_numpy=True):
    """Return [(offset x, y, z, radius)] per bone."""
    shape, root = make_shape(verts, bones)
    if use_numpy:
        shape.update_skin_center_radius()
    else:
        # importing numpy inside the function raises ImportError
        with mock.patch.dict(sys.modules, {'numpy': None}):
            shape.update_skin_center_radius()
    return [(block.bounding_sphere_offset.x, block.bounding_sphere_offset.y,
             block.bounding_sphere_offset.z, block.bounding_sphere_radius)
            for block in shape.skin_instance.data.bone_list]


def bits(values):
    """Bit patterns, so 0.0 and -0.0 (and nan) compare exactly."""
    return [struct.pack('<4d', *v) for v in values]


VERTS = [
    (0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0),
    (-4.5, 0.125, 7.0),
    (3.3, -1.7, -2.2),
    (0.1, 0.2, 0.3),
    (10.0, -10.0, 5.5),
    (0.0, -0.0, 1.0),
    (-0.0, 0.0, 2.0),
]


@needs_pyffi
class TestSkinCenterRadius(unittest.TestCase):

    def assert_same_as_original(self, verts, bones):
        new = run(verts, bones, use_numpy=True)
        old = run(verts, bones, use_numpy=False)
        self.assertEqual(bits(new), bits(old))
        return new

    def test_single_vertex(self):
        result = self.assert_same_as_original(
            VERTS, [([3], IDENTITY), ([3], TRANSFORM)])
        # identity transform: the center is the vertex, radius 0
        self.assertEqual(result[0], VERTS[3] + (0.0,))
        self.assertEqual(result[1][3], 0.0)

    def test_duplicate_indices(self):
        result = self.assert_same_as_original(
            VERTS, [([1, 2, 1, 3, 2, 2], IDENTITY),
                    ([1, 2, 3], IDENTITY),
                    ([5, 0, 5, 4, 0], TRANSFORM)])
        # duplicates do not change the bounding sphere
        self.assertEqual(result[0], result[1])
        x, y, z, r = result[0]
        self.assertEqual((x, y, z), ((-4.5 + 3.3) * 0.5,
                                     (-1.7 + 2.0) * 0.5,
                                     (-2.2 + 7.0) * 0.5))
        self.assertGreater(r, 0.0)

    def test_non_identity_transform(self):
        bones = [([0, 1, 2, 3, 4, 5], IDENTITY),
                 ([0, 1, 2, 3, 4, 5], TRANSFORM)]
        result = self.assert_same_as_original(VERTS, bones)
        # the radius is not transformed
        self.assertEqual(result[0][3], result[1][3])
        # the center is transformed: scale, rotation, then translation
        angle, scale, translation = TRANSFORM
        cx, cy, cz = result[0][:3]
        cos, sin = math.cos(angle), math.sin(angle)
        expected = (scale * (cx * cos - cy * sin) + translation[0],
                    scale * (cx * sin + cy * cos) + translation[1],
                    scale * cz + translation[2])
        for got, want in zip(result[1][:3], expected):
            self.assertAlmostEqual(got, want, places=9)
        self.assertNotEqual(result[1][:3], result[0][:3])

    def test_all_bones_many_vertices(self):
        # a larger shape with a regular structure and many ties
        verts = [((i % 7) * 0.5 - 1.5, ((i // 7) % 5) * -0.25,
                  (i % 3) * 1.25 - 1.0) for i in range(200)]
        bones = [(list(range(i, 200, 9)), TRANSFORM if i % 2 else IDENTITY)
                 for i in range(9)]
        self.assert_same_as_original(verts, bones)

    def test_signed_zero_extremes(self):
        # x is +-0.0 for every vertex of the bone: min and max are zero,
        # and their sign must match the original code
        for indices in ([6, 7], [7, 6], [0, 6, 7]):
            self.assert_same_as_original(
                VERTS, [(indices, IDENTITY), (indices, TRANSFORM)])

    def test_signed_zero_reaches_offset(self):
        # with a -0.0 translation, the sign of a zero center component
        # survives the transform, so a different zero would change the
        # stored offset
        verts = [(0.0, -1.0, -1.0), (-0.0, -2.0, -3.0)]
        transform = (0.0, 1.0, (-0.0, -0.0, -0.0))
        result = self.assert_same_as_original(verts, [([0, 1], transform)])
        self.assertEqual(struct.pack('<d', result[0][0]),
                         struct.pack('<d', 0.0))

    def test_nan_vertex(self):
        verts = list(VERTS) + [(float('nan'), 1.0, 1.0)]
        new = run(verts, [([1, 8, 2], IDENTITY)], use_numpy=True)
        old = run(verts, [([1, 8, 2], IDENTITY)], use_numpy=False)
        self.assertEqual(bits(new), bits(old))

    def test_empty_bone_raises(self):
        for use_numpy in (True, False):
            with self.assertRaises(ValueError):
                run(VERTS, [([1, 2], IDENTITY), ([], IDENTITY)],
                    use_numpy=use_numpy)

    def test_no_skin(self):
        shape = NifFormat.NiTriShape()
        self.assertIsNone(shape.update_skin_center_radius())


if __name__ == '__main__':
    unittest.main()
