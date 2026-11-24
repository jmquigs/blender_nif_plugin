"""Vectorized (numpy) tangent space calculation for nif geometry.

This is a drop-in replacement for pyffi's
``NifFormat.NiTriBasedGeom.update_tangent_space``. It reproduces that
function's output, including its quirks, but avoids pyffi's per-element
object overhead.

The module is split in two parts:

* :func:`compute_tangent_space` is pure numpy. It does not import bpy or
  pyffi, so it can be unit tested and compared on a plain Python install.
* :func:`apply_tangent_space` writes the result into a pyffi trishape.

The code must run on the Python 3.5 and the old numpy bundled with
Blender 2.79b, so it avoids f-strings and newer numpy APIs such as
``np.unique(..., axis=0)``.
"""

# ***** BEGIN LICENSE BLOCK *****
#
# Copyright © 2005-2015, NIF File Format Library and Tools contributors.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#
#    * Neither the name of the NIF File Format Library and Tools
#      project nor the names of its contributors may be used to endorse
#      or promote products derived from this software without specific
#      prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# ***** END LICENSE BLOCK *****

import numpy as np

# name of the Oblivion style tangent space extra data block
TANGENT_SPACE_EXTRA_NAME = b'Tangent space (binormal & tangent vectors)'

# hash precisions, as passed by pyffi's update_tangent_space to
# get_vertex_hash_generator
VERTEX_FACTOR = 10 ** 3
NORMAL_FACTOR = 10 ** 3
UV_FACTOR = 10 ** -2
VCOL_FACTOR = 10 ** -2

# pyffi.utils.mathutils.float_to_int maps infinities to these values
_INT_INF = 2147483648.0


def _float_to_int(values):
    """Vectorized pyffi.utils.mathutils.float_to_int.

    Returns float64 values holding integers. Python's ``int(x)`` of a
    float is exact, so comparing the truncated floats is the same as
    comparing pyffi's ints. NaN maps to 0, +-inf to +-2**31.
    """
    with np.errstate(invalid='ignore'):
        result = np.trunc(np.where(values > 0, values + 0.5, values - 0.5))
    result[np.isnan(result)] = 0.0
    result[result == np.inf] = _INT_INF
    result[result == -np.inf] = -_INT_INF
    # fold -0.0 into 0.0
    result += 0.0
    return result


def _group_rows(keys):
    """Map each row of the (N, K) array *keys* to a group id.

    Rows that compare equal get the same id. Equivalent to
    ``np.unique(keys, axis=0, return_inverse=True)[1]`` (up to the
    numbering of the groups) but works on numpy < 1.13.

    :return: (inverse, num_groups)
    """
    num_rows = keys.shape[0]
    if num_rows == 0:
        return np.zeros(0, dtype=np.intp), 0
    order = np.lexsort(keys.T)
    sorted_keys = keys[order]
    is_new = np.empty(num_rows, dtype=bool)
    is_new[0] = True
    is_new[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    group_sorted = np.cumsum(is_new) - 1
    inverse = np.empty(num_rows, dtype=np.intp)
    inverse[order] = group_sorted
    return inverse, int(group_sorted[-1]) + 1


def _dot(a, b):
    """Row-wise dot product, summed in the same order as pyffi."""
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def _normalize_rows(vecs):
    """Normalize rows in place like pyffi's Vector3.normalize.

    pyffi computes ``factor = 1.0 / sqrt(x*x + y*y + z*z)`` and then
    multiplies each component by it. A row whose squared length is
    exactly zero raises ZeroDivisionError in pyffi; such rows are left
    as they are here.

    :return: Boolean mask of the rows that were normalized.
    """
    sqlen = _dot(vecs, vecs)
    ok = sqlen != 0
    vecs[ok] *= (1.0 / np.sqrt(sqlen[ok]))[:, None]
    return ok


def normalize_normals(norms):
    """Return a float64 copy of *norms* with pyffi's normalization applied.

    Zero length normals are left alone, as pyffi does.
    """
    norms = np.array(norms, dtype=np.float64).reshape(-1, 3)
    with np.errstate(all='ignore'):
        _normalize_rows(norms)
    return norms


def compute_tangent_space(verts, norms, uvs, tris,
                          hash_uvs=None, vcols=None, stats=None):
    """Compute the tangent space as pyffi's update_tangent_space does.

    :param verts: (N, 3) vertex positions.
    :param norms: (N, 3) vertex normals (need not be normalized).
    :param uvs: (N, 2) first UV set, as stored in the nif (V flipped).
    :param tris: (T, 3) triangle vertex indices.
    :param hash_uvs: Optional list of (N, 2) arrays, all the UV sets of
        the shape. pyffi puts every UV set into the vertex hash. Defaults
        to ``[uvs]``. The UV hash factor is 0.01, so this only matters for
        UVs of magnitude 50 or more.
    :param vcols: Optional (N, 4) vertex colors, for the vertex hash.
        These only matter for colors of magnitude 50 or more.
    :param stats: Optional dict, filled in with ``num_groups``,
        ``num_degenerate_groups`` and ``degenerate_vertices`` (a boolean
        (N,) mask of the vertices in degenerate groups).
    :return: ``(tangents, bitangents, normalized_norms)``, (N, 3) float64
        arrays. ``normalized_norms`` is what pyffi leaves in
        ``data.normals`` after its in-place normalization.
    """
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    norms = np.array(norms, dtype=np.float64).reshape(-1, 3)
    uvs = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    tris = np.asarray(tris, dtype=np.intp).reshape(-1, 3)
    num_verts = verts.shape[0]
    if hash_uvs is None:
        hash_uvs = [uvs]

    with np.errstate(all='ignore'):
        # step 1: vertex hash, to merge vertices split along uv seams
        key_columns = [verts * VERTEX_FACTOR, norms * NORMAL_FACTOR]
        for uvset in hash_uvs:
            key_columns.append(
                np.asarray(uvset, dtype=np.float64).reshape(-1, 2) * UV_FACTOR)
        if vcols is not None:
            key_columns.append(
                np.asarray(vcols, dtype=np.float64).reshape(-1, 4) * VCOL_FACTOR)
        keys = _float_to_int(np.hstack(key_columns))
        group, num_groups = _group_rows(keys)

        # step 2: per triangle contributions
        t1 = tris[:, 0]
        t2 = tris[:, 1]
        t3 = tris[:, 2]
        h1 = group[t1]
        h2 = group[t2]
        h3 = group[t3]
        keep = (h1 != h2) & (h2 != h3) & (h3 != h1)

        v_2v_1 = verts[t2] - verts[t1]
        v_3v_1 = verts[t3] - verts[t1]
        w2w1 = uvs[t2] - uvs[t1]
        w3w1 = uvs[t3] - uvs[t1]
        w2w1_u = w2w1[:, 0:1]
        w2w1_v = w2w1[:, 1:2]
        w3w1_u = w3w1[:, 0:1]
        w3w1_v = w3w1[:, 1:2]

        # surface of triangle in texture space, and its sign (nan -> -1)
        r = w2w1_u * w3w1_v - w3w1_u * w2w1_v
        r_sign = np.where(r >= 0, 1.0, -1.0)

        sdir = (w3w1_v * v_2v_1 - w2w1_v * v_3v_1) * r_sign
        tdir = (w2w1_u * v_3v_1 - w3w1_u * v_2v_1) * r_sign
        keep &= _normalize_rows(sdir)
        keep &= _normalize_rows(tdir)

        # accumulate; sdir goes into the bitangents and tdir into the
        # tangents (the nif convention, keep as is)
        # bincount adds the weights in input order, which with the
        # (triangle, corner) row major layout below is the same order in
        # which pyffi adds them, and it is much faster than np.add.at on
        # old numpy versions
        hashes = np.column_stack((h1, h2, h3))[keep].ravel()
        tan = np.empty((num_groups, 3), dtype=np.float64)
        bin_ = np.empty((num_groups, 3), dtype=np.float64)
        tdir = tdir[keep]
        sdir = sdir[keep]
        # (old numpy rejects minlength=0)
        minlength = max(num_groups, 1)
        for k in range(3):
            tan[:, k] = np.bincount(
                hashes, weights=np.repeat(tdir[:, k], 3),
                minlength=minlength)[:num_groups]
            bin_[:, k] = np.bincount(
                hashes, weights=np.repeat(sdir[:, k], 3),
                minlength=minlength)[:num_groups]

        # step 3: Gram-Schmidt per group
        # pyffi normalizes data.normals in place; a zero normal is left
        # alone and (0, 1, 0) is used for the math instead
        normalized_norms = norms
        normal_ok = _normalize_rows(normalized_norms)
        math_norms = normalized_norms.copy()
        math_norms[~normal_ok] = (0.0, 1.0, 0.0)

        # pyffi loops over all vertices in index order and updates the
        # shared per group vectors in place, so a group with k vertices
        # is processed k times. To reproduce that exactly, process the
        # vertices in passes: pass r handles the r-th vertex (in index
        # order) of every group with more than r vertices. Within one
        # pass all groups are distinct, so it vectorizes.
        degenerate = np.zeros(num_groups, dtype=bool)
        if num_verts:
            by_group = np.argsort(group, kind='mergesort')  # stable
            counts = np.bincount(group, minlength=num_groups)
            starts = np.zeros(num_groups, dtype=np.intp)
            starts[1:] = np.cumsum(counts)[:-1]
            max_count = int(counts.max())
        else:
            max_count = 0
        for rank in range(max_count):
            active = counts > rank
            vert_idx = by_group[starts[active] + rank]
            g = group[vert_idx]
            n = math_norms[vert_idx]

            # bin[h] -= n * (n * bin[h]), then normalize
            b = bin_[g]
            b -= n * _dot(n, b)[:, None]
            b_ok = _normalize_rows(b)
            bin_[g] = b
            # on failure pyffi leaves the tangent untouched
            degenerate[g[~b_ok]] = True

            # tan[h] -= n * (n * tan[h])
            # tan[h] -= bin[h] * (bin[h] * tan[h]), then normalize
            g = g[b_ok]
            n = n[b_ok]
            b = b[b_ok]
            t = tan[g]
            t -= n * _dot(n, t)[:, None]
            t -= b * _dot(b, t)[:, None]
            t_ok = _normalize_rows(t)
            tan[g] = t
            degenerate[g[~t_ok]] = True

        # pyffi's ZeroDivisionError fallback builds a frame from cross
        # products but only assigns it to locals, so it has no effect.
        # Degenerate groups therefore keep the zero or raw vectors above.

    if stats is not None:
        stats['num_groups'] = num_groups
        stats['num_degenerate_groups'] = int(np.count_nonzero(degenerate))
        stats['degenerate_vertices'] = degenerate[group]

    # step 4: back to per vertex arrays
    return tan[group], bin_[group], normalized_norms


def uv_sets_from_export_list(uvlist, num_uv_sets):
    """Build the nif UV sets from the exporter's per-vertex uv list.

    Matches what the exporter writes: ``(u, 1.0 - v)``, and ``(0, 0)``
    (pyffi's default) for vertices whose uv list is empty.

    :param uvlist: For each vertex, a sequence of (u, v) per uv layer, or
        an empty sequence.
    :param num_uv_sets: Number of uv layers.
    :return: List of (N, 2) float64 arrays, one per uv set.
    """
    num_verts = len(uvlist)
    has_uv = np.array([len(uv) != 0 for uv in uvlist], dtype=bool)
    uv_sets = []
    for j in range(num_uv_sets):
        uvs = np.zeros((num_verts, 2), dtype=np.float64)
        if has_uv.any():
            values = np.array([(uv[j][0], uv[j][1])
                               for uv in uvlist if len(uv) != 0],
                              dtype=np.float64).reshape(-1, 2)
            values[:, 1] = 1.0 - values[:, 1]  # opengl standard
            uvs[has_uv] = values
        uv_sets.append(uvs)
    return uv_sets


def _to_bytes(array):
    if hasattr(array, 'tobytes'):
        return array.tobytes()
    return array.tostring()  # numpy < 1.9


def find_tangent_space_extra(trishape):
    """Return the Oblivion style tangent space extra data block, or None."""
    from pyffi.formats.nif import NifFormat
    for extra in trishape.get_extra_datas():
        if isinstance(extra, NifFormat.NiBinaryExtraData):
            if extra.name == TANGENT_SPACE_EXTRA_NAME:
                return extra
    return None


def apply_tangent_space(trishape, tangents, bitangents, as_extra=None):
    """Store tangent space data in a pyffi trishape.

    This is the output stage of pyffi's update_tangent_space.

    :param trishape: A NifFormat.NiTriBasedGeom.
    :param tangents: (N, 3) tangents.
    :param bitangents: (N, 3) bitangents.
    :param as_extra: True to store as NiBinaryExtraData (Oblivion), False
        to store in the geometry data (Fallout 3, Skyrim), None to
        autodetect: True only if the extra data block already exists.
    """
    from pyffi.formats.nif import NifFormat
    data = trishape.data
    extra = find_tangent_space_extra(trishape)
    if as_extra is None:
        as_extra = bool(extra)

    if as_extra:
        if not extra:
            extra = NifFormat.NiBinaryExtraData()
            extra.name = TANGENT_SPACE_EXTRA_NAME
            trishape.add_extra_data(extra)
        packed = np.concatenate(
            (np.asarray(tangents, dtype=np.float64).reshape(-1, 3),
             np.asarray(bitangents, dtype=np.float64).reshape(-1, 3)))
        extra.binary_data = _to_bytes(packed.astype('<f4'))
    else:
        data.num_uv_sets |= 4096
        data.bs_num_uv_sets |= 4096
        data.tangents.update_size()
        data.bitangents.update_size()
        for values, vecs in ((tangents, data.tangents),
                             (bitangents, data.bitangents)):
            # tolist gives python floats, much faster to iterate
            for (x, y, z), vec in zip(np.asarray(values).tolist(), vecs):
                vec.x = x
                vec.y = y
                vec.z = z


def clear_tangent_space_flag(data):
    """What pyffi does when the shape has no UV sets."""
    data.num_uv_sets &= ~4096
    data.bs_num_uv_sets &= ~4096


def read_geometry(data):
    """Read the tangent space inputs out of pyffi geometry data.

    This goes through pyffi's objects and is slow. The exporter builds the
    arrays from its own lists instead; this is for tools and tests.

    :return: dict with verts, norms, uvs, tris, hash_uvs, vcols.
    """
    num_verts = data.num_vertices
    verts = np.array([v.as_tuple() for v in data.vertices],
                     dtype=np.float64).reshape(-1, 3)
    norms = np.array([v.as_tuple() for v in data.normals],
                     dtype=np.float64).reshape(-1, 3)
    hash_uvs = [np.array([(uv.u, uv.v) for uv in uvset],
                         dtype=np.float64).reshape(-1, 2)
                for uvset in data.uv_sets]
    vcols = None
    if data.has_vertex_colors:
        vcols = np.array([(c.r, c.g, c.b, c.a) for c in data.vertex_colors],
                         dtype=np.float64).reshape(-1, 4)
    tris = np.array(list(data.get_triangles()), dtype=np.intp).reshape(-1, 3)
    return dict(verts=verts, norms=norms,
                uvs=hash_uvs[0] if hash_uvs else np.zeros((num_verts, 2)),
                tris=tris, hash_uvs=hash_uvs, vcols=vcols)


def update_tangent_space(trishape, as_extra=None, stats=None):
    """Numpy equivalent of trishape.update_tangent_space(as_extra).

    Reads its inputs out of the pyffi data (slow); see read_geometry.
    Includes pyffi's early exits and in-place normal normalization.

    :return: (tangents, bitangents) or None on early exit.
    """
    from pyffi.formats.nif import NifFormat
    data = trishape.data
    if not isinstance(data, NifFormat.NiTriBasedGeomData):
        raise ValueError(
            'cannot update tangent space of a geometry with %s data'
            % (data.__class__ if data else 'no'))
    if len(data.uv_sets) == 0:
        clear_tangent_space_flag(data)
        return None
    if len(data.uv_sets[0]) == 0 or len(data.normals) == 0:
        return None
    geom = read_geometry(data)
    tangents, bitangents, norms = compute_tangent_space(
        geom['verts'], geom['norms'], geom['uvs'], geom['tris'],
        hash_uvs=geom['hash_uvs'], vcols=geom['vcols'], stats=stats)
    for (x, y, z), vec in zip(norms.tolist(), data.normals):
        vec.x = x
        vec.y = y
        vec.z = z
    apply_tangent_space(trishape, tangents, bitangents, as_extra)
    return tangents, bitangents
