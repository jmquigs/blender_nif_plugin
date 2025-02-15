"""This script contains classes to import collision objects."""

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

import bpy
import mathutils

from functools import reduce
import operator

from pyffi.formats.nif import NifFormat
from pyffi.utils.quickhull import qhull3d
from io_scene_nif.modules.object.object_import import NiObject
from io_scene_nif.utility.nif_logging import NifLog
from io_scene_nif.utility.nif_global import NifOp


def box_from_extents(b_name, minx, maxx, miny, maxy, minz, maxz):
    verts = []
    for x in [minx, maxx]:
        for y in [miny, maxy]:
            for z in [minz, maxz]:
                verts.append( (x,y,z) )
    faces = [[0,1,3,2],[6,7,5,4],[0,2,6,4],[3,1,5,7],[4,5,1,0],[7,6,2,3]]
    return mesh_from_data(b_name, verts, faces)
    
def create_ob(ob_name, ob_data):
    ob = bpy.data.objects.new(ob_name, ob_data)
    bpy.context.scene.objects.link(ob)
    bpy.context.scene.objects.active = ob
    return ob

def mesh_from_data(name, verts, faces, wireframe = True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = create_ob(name, me)
    if wireframe:
        ob.draw_type = 'WIRE'
    return ob

class bhkshape_import():
    """Import basic and Havok Collision Shapes"""

    def __init__(self, parent):
        self.nif_import = parent
        self.HAVOK_SCALE = parent.HAVOK_SCALE

    def get_havok_objects(self):
        return self.nif_import.dict_havok_objects

    def import_bhk_shape(self, bhkshape):
        """Imports any supported collision shape as list of blender meshes."""

        if self.nif_import.data._user_version_value_._value == 12:
            if self.nif_import.data._user_version_2_value_._value == 83:
                self.HAVOK_SCALE = self.nif_import.HAVOK_SCALE * 10
            else:
                self.HAVOK_SCALE = self.nif_import.HAVOK_SCALE

        if isinstance(bhkshape, NifFormat.bhkTransformShape):
            return self.import_bhktransform(bhkshape)

        elif isinstance(bhkshape, NifFormat.bhkRigidBody):
            return self.import_bhkridgidbody(bhkshape)

        elif isinstance(bhkshape, NifFormat.bhkBoxShape):
            return self.import_bhkbox_shape(bhkshape)

        elif isinstance(bhkshape, NifFormat.bhkSphereShape):
            return self.import_bhksphere_shape(bhkshape)

        elif isinstance(bhkshape, NifFormat.bhkCapsuleShape):
            return self.import_bhkcapsule_shape(bhkshape)

        elif isinstance(bhkshape, NifFormat.bhkConvexVerticesShape):
            return self.import_bhkconvex_vertices_shape(bhkshape)

        elif isinstance(bhkshape, NifFormat.bhkPackedNiTriStripsShape):
            return self.import_bhkpackednitristrips_shape(bhkshape)

        elif isinstance(bhkshape, NifFormat.bhkNiTriStripsShape):
            self.havok_mat = bhkshape.material
            return reduce(operator.add,
                          (self.import_bhk_shape(strips)
                           for strips in bhkshape.strips_data))

        elif isinstance(bhkshape, NifFormat.NiTriStripsData):
            return self.import_nitristrips(bhkshape)

        elif isinstance(bhkshape, NifFormat.bhkMoppBvTreeShape):
            return self.import_bhk_shape(bhkshape.shape)

        elif isinstance(bhkshape, NifFormat.bhkListShape):
            return reduce(operator.add, ( self.import_bhk_shape(subshape)
                                          for subshape in bhkshape.sub_shapes ))

        NifLog.warn("Unsupported bhk shape {0}".format(bhkshape.__class__.__name__))
        return []


    def import_bhktransform(self, bhkshape):
        """Imports a BhkTransform block and applies the transform to the collision object"""

        # import shapes
        collision_objs = self.import_bhk_shape(bhkshape.shape)
        # find transformation matrix
        transform = mathutils.Matrix(bhkshape.transform.as_list())

        # fix scale
        transform.translation = transform.translation * self.HAVOK_SCALE

        # apply transform
        for b_col_obj in collision_objs:
            b_col_obj.matrix_local = b_col_obj.matrix_local * transform
            # b_col_obj.nifcollision.havok_material = NifFormat.HavokMaterial._enumkeys[bhkshape.material]
            # and return a list of transformed collision shapes
        return collision_objs


    def import_bhkridgidbody(self, bhkshape):
        """Imports a BhkRigidBody block and applies the transform to the collision objects"""

        # import shapes
        collision_objs = self.import_bhk_shape(bhkshape.shape)

        # find transformation matrix in case of the T version
        if isinstance(bhkshape, NifFormat.bhkRigidBodyT):
            # set rotation
            transform = mathutils.Quaternion([
                bhkshape.rotation.w, bhkshape.rotation.x,
                bhkshape.rotation.y, bhkshape.rotation.z]).to_matrix()
            transform = transform.to_4x4()

            # set translation
            transform.translation = mathutils.Vector(
                    (bhkshape.translation.x * self.HAVOK_SCALE,
                     bhkshape.translation.y * self.HAVOK_SCALE,
                     bhkshape.translation.z * self.HAVOK_SCALE))

            # apply transform
            for b_col_obj in collision_objs:
                b_col_obj.matrix_local = b_col_obj.matrix_local * transform

        # set physics flags and mass
        for b_col_obj in collision_objs:
            scn = bpy.context.scene
            scn.objects.active = b_col_obj
            bpy.ops.rigidbody.object_add(type='ACTIVE')
            b_col_obj.rigid_body.enabled = True
            
            if bhkshape.mass > 0.0001:
                # for physics emulation
                # (mass 0 results in issues with simulation)
                b_col_obj.rigid_body.mass = bhkshape.mass / len(collision_objs)


            b_col_obj.nifcollision.deactivator_type = NifFormat.DeactivatorType._enumkeys[bhkshape.deactivator_type]
            b_col_obj.nifcollision.solver_deactivation = NifFormat.SolverDeactivation._enumkeys[bhkshape.solver_deactivation]
            b_col_obj.nifcollision.oblivion_layer = NifFormat.OblivionLayer._enumkeys[bhkshape.layer]
            b_col_obj.nifcollision.quality_type = NifFormat.MotionQuality._enumkeys[bhkshape.quality_type]
            b_col_obj.nifcollision.motion_system = NifFormat.MotionSystem._enumkeys[bhkshape.motion_system]
            
            b_col_obj.niftools.bsxflags = self.nif_import.bsxflags
            b_col_obj.niftools.objectflags = self.nif_import.objectflags
            b_col_obj.niftools.upb = self.nif_import.upbflags
            
            b_col_obj.rigid_body.mass = bhkshape.mass / len(collision_objs)
            
            b_col_obj.rigid_body.use_deactivation = True
            b_col_obj.rigid_body.friction = bhkshape.friction
            b_col_obj.rigid_body.restitution = bhkshape.restitution
            #b_col_obj.rigid_body. = bhkshape.
            b_col_obj.rigid_body.linear_damping = bhkshape.linear_damping
            b_col_obj.rigid_body.angular_damping = bhkshape.angular_damping
            b_col_obj.rigid_body.deactivate_linear_velocity = mathutils.Vector([
                                            bhkshape.linear_velocity.w,
                                            bhkshape.linear_velocity.x, 
                                            bhkshape.linear_velocity.y, 
                                            bhkshape.linear_velocity.z]).magnitude
            b_col_obj.rigid_body.deactivate_angular_velocity = mathutils.Vector([
                                            bhkshape.angular_velocity.w,
                                            bhkshape.angular_velocity.x, 
                                            bhkshape.angular_velocity.y, 
                                            bhkshape.angular_velocity.z]).magnitude
            
            b_col_obj.collision.permeability = bhkshape.penetration_depth

            b_col_obj.nifcollision.max_linear_velocity = bhkshape.max_linear_velocity
            b_col_obj.nifcollision.max_angular_velocity = bhkshape.max_angular_velocity
                        
            b_col_obj.nifcollision.col_filter = bhkshape.col_filter

        # import constraints
        # this is done once all objects are imported
        # for now, store all imported havok shapes with object lists
        self.nif_import.dict_havok_objects[bhkshape] = collision_objs

        # and return a list of transformed collision shapes
        return collision_objs


    def import_bhkbox_shape(self, bhkshape):
        """Import a BhkBox block as a simple Box collision object"""
        # create box
        minx = -bhkshape.dimensions.x * self.HAVOK_SCALE
        maxx = +bhkshape.dimensions.x * self.HAVOK_SCALE
        miny = -bhkshape.dimensions.y * self.HAVOK_SCALE
        maxy = +bhkshape.dimensions.y * self.HAVOK_SCALE
        minz = -bhkshape.dimensions.z * self.HAVOK_SCALE
        maxz = +bhkshape.dimensions.z * self.HAVOK_SCALE

        #create blender object
        b_obj = box_from_extents("box", minx, maxx, miny, maxy, minz, maxz)

        # set bounds type
        b_obj.draw_type = 'WIRE'
        b_obj.draw_bounds_type = 'BOX'
        b_obj.game.use_collision_bounds = True
        b_obj.game.collision_bounds_type = 'BOX'
        b_obj.game.radius = bhkshape.radius
        b_obj.nifcollision.havok_material = NifFormat.HavokMaterial._enumkeys[bhkshape.material]
        
        return [ b_obj ]


    def import_bhksphere_shape(self, bhkshape):
        """Import a BhkSphere block as a simple uv-sphere collision object"""
        # create sphere
        b_radius = bhkshape.radius * self.HAVOK_SCALE
        
        b_obj = box_from_extents("sphere", -b_radius, b_radius, -b_radius, b_radius, -b_radius, b_radius)

        # set bounds type
        b_obj.draw_type = 'WIRE'
        b_obj.draw_bounds_type = 'SPHERE'
        b_obj.game.use_collision_bounds = True
        b_obj.game.collision_bounds_type = 'SPHERE'
        b_obj.game.radius = bhkshape.radius
        b_obj.nifcollision.havok_material = NifFormat.HavokMaterial._enumkeys[bhkshape.material]

        return [ b_obj ]


    def import_bhkcapsule_shape(self, bhkshape):
        """Import a BhkCapsule block as a simple cylinder collision object"""
        b_radius = bhkshape.radius
        # create capsule mesh
        length = (bhkshape.first_point - bhkshape.second_point).norm()
        minx = miny = -b_radius * self.HAVOK_SCALE
        maxx = maxy = +b_radius * self.HAVOK_SCALE
        minz = -(length + 2*b_radius) * (self.HAVOK_SCALE / 2)
        maxz = +(length + 2*b_radius) * (self.HAVOK_SCALE / 2)

        #create blender object
        b_obj = box_from_extents("capsule", minx, maxx, miny, maxy, minz, maxz)

        # set bounds type
        b_obj.draw_type = 'BOUNDS'
        b_obj.draw_bounds_type = 'CAPSULE'
        b_obj.game.use_collision_bounds = True
        b_obj.game.collision_bounds_type = 'CAPSULE'
        b_obj.game.radius = bhkshape.radius*self.HAVOK_SCALE
        b_obj.nifcollision.havok_material = NifFormat.HavokMaterial._enumkeys[bhkshape.material]
        
        # center around middle; will acount for bone length once it is parented
        b_obj.location.y = length / 2 * self.HAVOK_SCALE
        return [ b_obj ]


    def import_bhkconvex_vertices_shape(self, bhkshape):
        """Import a BhkConvexVertex block as a convex hull collision object"""

        # find vertices (and fix scale)
        verts, faces = qhull3d(   [ (self.HAVOK_SCALE * n_vert.x,
                                     self.HAVOK_SCALE * n_vert.y,
                                     self.HAVOK_SCALE * n_vert.z)
                                     for n_vert in bhkshape.vertices ])

        b_obj = mesh_from_data("convexpoly", verts, faces)

        b_obj.show_wire = True
        b_obj.draw_type = 'WIRE'
        b_obj.draw_bounds_type = 'BOX'
        b_obj.game.use_collision_bounds = True
        b_obj.game.collision_bounds_type = 'CONVEX_HULL'

        # radius: quick estimate
        b_obj.game.radius = bhkshape.radius
        b_obj.nifcollision.havok_material = NifFormat.HavokMaterial._enumkeys[bhkshape.material]

        return [ b_obj ]


    def import_nitristrips(self, bhkshape):
        """Import a NiTriStrips block as a Triangle-Mesh collision object"""
        # no factor 7 correction!!!
        verts = [ (v.x, v.y, v.z) for v in bhkshape.vertices ]
        faces = list(bhkshape.get_triangles())
        b_obj = mesh_from_data("poly", verts, faces)
        
        # set bounds type
        b_obj.draw_type = 'WIRE'
        b_obj.draw_bounds_type = 'BOX'
        b_obj.game.use_collision_bounds = True
        b_obj.game.collision_bounds_type = 'TRIANGLE_MESH'
        # radius: quick estimate
        b_obj.game.radius = bhkshape.radius
        b_obj.nifcollision.havok_material = NifFormat.HavokMaterial._enumkeys[self.havok_mat]

        return [ b_obj ]

    def import_bhkpackednitristrips_shape(self, bhkshape):
        """Import a BhkPackedNiTriStrips block as a Triangle-Mesh collision object"""

        # create mesh for each sub shape
        hk_objects = []
        vertex_offset = 0
        subshapes = bhkshape.sub_shapes

        if not subshapes:
            # fallout 3 stores them in the data
            subshapes = bhkshape.data.sub_shapes

        for subshape_num, subshape in enumerate(subshapes):
            verts = []
            faces = []
            for vert_index in range(vertex_offset, vertex_offset + subshape.num_vertices):
                n_vert = bhkshape.data.vertices[vert_index]
                verts.append( (n_vert.x * self.HAVOK_SCALE,
                               n_vert.y * self.HAVOK_SCALE,
                               n_vert.z * self.HAVOK_SCALE) )

            for hktriangle in bhkshape.data.triangles:
                if ((vertex_offset <= hktriangle.triangle.v_1)
                    and (hktriangle.triangle.v_1
                         < vertex_offset + subshape.num_vertices)):
                    faces.append((hktriangle.triangle.v_1 - vertex_offset,
                                  hktriangle.triangle.v_2 - vertex_offset,
                                  hktriangle.triangle.v_3 - vertex_offset) )
                else:
                    continue
            # todo: face normals are ignored here - are they even relevant?
            # could just run the recalc normals operator if they are
            # old solution was rather hacky anyway
            
            b_obj = mesh_from_data('poly%i' % subshape_num, verts, faces)

            # set bounds type
            b_obj.draw_type = 'WIRE'
            b_obj.draw_bounds_type = 'BOX'
            b_obj.game.use_collision_bounds = True
            b_obj.game.collision_bounds_type = 'TRIANGLE_MESH'
            # radius: quick estimate
            b_obj.game.radius = min(vert.co.length for vert in b_mesh.vertices)
            # set material
            b_obj.nifcollision.havok_material = NifFormat.HavokMaterial._enumkeys[subshape.material]

            vertex_offset += subshape.num_vertices
            hk_objects.append(b_obj)

        return hk_objects

class bound_import():
    """Import a bound box shape"""

    def __init__(self, parent):
        self.nif_import = parent

    def import_bounding_box(self, bbox):
        """Import a bounding box (BSBound, or NiNode with bounding box)."""

        # calculate bounds
        if isinstance(bbox, NifFormat.BSBound):
            b_name = 'BSBound'
            minx = bbox.center.x - bbox.dimensions.x
            miny = bbox.center.y - bbox.dimensions.y
            minz = bbox.center.z - bbox.dimensions.z
            maxx = bbox.center.x + bbox.dimensions.x
            maxy = bbox.center.y + bbox.dimensions.y
            maxz = bbox.center.z + bbox.dimensions.z
            n_bbox_center = bbox.center.as_list()
            
        elif isinstance(bbox, NifFormat.NiNode):
            if not bbox.has_bounding_box:
                raise ValueError("Expected NiNode with bounding box.")
            b_name = 'Bounding Box'

            # Ninode's(bbox) behaves like a seperate mesh.
            # bounding_box center(bbox.bounding_box.translation) is relative to the bound_box
            minx = bbox.bounding_box.translation.x - bbox.translation.x - bbox.bounding_box.radius.x
            miny = bbox.bounding_box.translation.y - bbox.translation.y - bbox.bounding_box.radius.y
            minz = bbox.bounding_box.translation.z - bbox.translation.z - bbox.bounding_box.radius.z
            maxx = bbox.bounding_box.translation.x - bbox.translation.x + bbox.bounding_box.radius.x
            maxy = bbox.bounding_box.translation.y - bbox.translation.y + bbox.bounding_box.radius.y
            maxz = bbox.bounding_box.translation.z - bbox.translation.z + bbox.bounding_box.radius.z
            n_bbox_center = bbox.bounding_box.translation.as_list()

        else:
            raise TypeError("Expected BSBound or NiNode but got %s."
                            % bbox.__class__.__name__)

        #create blender object
        b_obj = box_from_extents(b_name, minx, maxx, miny, maxy, minz, maxz)
        # link box to scene and set transform
        # XXX this is set in the import_branch() method
        # ob.matrix_local = mathutils.Matrix(
        #    *bbox.bounding_box.rotation.as_list())
        # ob.setLocation(
        #    *bbox.bounding_box.translation.as_list())
        
        # TODO b_obj.niftools.objectflags = self.nif_import.objectflags
        b_obj.location = n_bbox_center

        # set bounds type
        b_obj.show_bounds = True
        b_obj.draw_type = 'BOUNDS'
        b_obj.draw_bounds_type = 'BOX'
        b_obj.game.use_collision_bounds = True
        b_obj.game.collision_bounds_type = 'BOX'
        # quick radius estimate
        b_obj.game.radius = max(maxx, maxy, maxz)
        
        return b_obj
