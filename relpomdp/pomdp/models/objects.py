from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.arrays import *
from abc import ABC, abstractmethod
import moos3d.util as util
import numpy as np

class GObjManager:
    def __init__(self):
        self._cls_map = {}
        self._int_to_cls = {}
        self._str_to_cls = {}

    def register(self, objcls, typeint, typestr):
        """objcls should be a child class of GridWorldObject"""
        if objcls in self._cls_map:
            raise ValueError("%s is already registered!" % objcls.__name__)
        self._cls_map[objcls] = (objcls, typeint, typestr)
        if typeint in self._int_to_cls:
            raise ValueError("Object %d is already registered!" % typeint)
        self._int_to_cls[typeint] = objcls
        if typestr in self._str_to_cls:
            raise ValueError("Object %s is already registered!" % typestr)
        self._str_to_cls[typestr] = objcls

    def register_all(self, objclses):
        for objcls, typeint, typestr in objclses:
            self.register(objcls, typeint, typestr)

    def is_known(self, obj):
        """Given an obj, return True if this obj is considered known.
        
        obj can be:
          An object instance of a subclass of GridWorldObject
          A string or integer representation of the object class
          An object class
        """
        if isinstance(obj, GridWorldObject):
            return obj.__class__ in self._cls_map
        elif type(obj) == int:
            return obj in self._int_to_cls
        elif type(obj) == str:
            return obj in self._str_to_cls
        else:
            return obj in self._cls_map

    def all_object_types(self, get_str=True):
        if get_str:
            return list(self._str_to_cls.keys())
        else:
            return list(self._cls_map.keys())

    def info(self, obj):
        if not self.is_known(obj):
            return None
        if isinstance(obj, GridWorldObject):
            return self._cls_map[obj.__class__]
        elif type(obj) == int:
            return self._cls_map[self._int_to_cls[obj]]
        elif type(obj) == str:
            return self._cls_map[self._str_to_cls[obj]]
        else:
            return self._cls_map[obj]
    


class GridWorldObject(ABC):
    """`x,y,z` are coordinates relative to the world frame."""
    def __init__(self, id_, objtype="object"):
        self._objtype = objtype
        self._id = id_  # id of the object, int.
    @property
    def objtype(self):
        return self._objtype
    @property
    def id(self):
        return self._id
    @abstractmethod
    def render(self):
        pass
    @abstractmethod
    def cleanup(self):
        pass

class TetrisObject(GridWorldObject):
    """A tetris object is an object made up of one or several connected cubes.
    Note that `coords` are coordinates relative to the tetris object's frame.
    And `x,y,z` are coordinates relative to the world frame."""
    def __init__(self, id_, coords, objtype="tetris_block", color=(0, 1, 0.5)):
        super().__init__(id_, objtype=objtype)
        self._color = color
        self._coords = coords
        self._coords_to_index = {self._coords[i]:i for i in range(len(self._coords))}
        

    def cube_poses(self, x, y, z):
        poses = [
            np.array([x + c[0],
                      y + c[1],
                      z + c[2]])
            for c in self._coords
        ]
        return poses

    def cube_index(self, cube_coords):
        return self._coords_to_index[cube_coords]

    @property
    def object_id_attrb(self):
        return self._object_id_attrb
        
    def init_render(self):
        vertices, indices, colors, bcolors = util.cube(color=self._color,
                                                       boundary_color=list(i/2 for i in self._color))
        self._vertex_vbo, self._index_vbo\
            = util.generate_vbo_elements(vertices, indices)
        self._color_vbo, self._bcolor_vbo\
            = util.generate_vbo_arrays([colors, bcolors])
        self._num_indices = len(indices)

        # We want to have this data available for every vertex rendered for this object.
        self._object_id_attrb = [self._id] * 8 * len(self._coords)

    def _draw_cube(self):
        util.draw_quads(self._num_indices,
                        self._vertex_vbo, self._index_vbo, self._color_vbo,
                        self._bcolor_vbo, 3, 3)

    def render(self):
        for x, y, z in self._coords:
            glPushMatrix()
            glTranslatef(x, y, z)
            glTranslatef(0.5, 0.5, 0.5)  # align the cube with the grid lines; 0.5 is half of cube length.
            self._draw_cube()
            glPopMatrix()

    def render_cube(self, i):
        x, y, z = self._coords[i]
        glPushMatrix()
        glTranslatef(x, y, z)
        glTranslatef(0.5, 0.5, 0.5)  # align the cube with the grid lines; 0.5 is half of cube length.
        self._draw_cube()
        glPopMatrix()

    def cleanup(self):
        glDeleteBuffers(4, np.array([self._vertex_vbo, self._index_vbo, self._color_vbo, self._bcolor_vbo]))

class Cube(TetrisObject):
    def __init__(self, id_, objtype="cube", color=(0,1,1)):
        coords = [(0,0,0)]
        super().__init__(id_, coords, objtype=objtype, color=color)

class OrangeRicky(TetrisObject):
    """
    Corresponds to
  
           ^ x-axis
           b
         aCd  -> y-axis
    """
    def __init__(self, id_, objtype="orange_ricky", color=(0.9, 0.5, 0)):
        coords = [(0,0,0), # center "C"
                  (0,1,0),  # d
                  (1,1,0),  # b
                  (0,-1,0)] # a
        super().__init__(id_, coords, objtype=objtype, color=color)


class Hero(TetrisObject):
    """
    Corresponds to
         aCbd  -> y-axis
    """
    def __init__(self, id_, objtype="hero", color=(0, 1, 1)):
        coords = [(0,0,0),  # center "C"
                  (0,-1,0), # a
                  (0,1,0),  # b
                  (0,2,0)]  # d
        super().__init__(id_, coords, objtype=objtype, color=color)        


class Teewee(TetrisObject):
    """
    Corresponds to

        bCd  -> y-axis
         a
    """
    def __init__(self, id_, objtype="hero", color=(0.7, 0, 0.9)):
        coords = [(0,0,0),  # center "C"
                  (-1,0,0), # a
                  (0,-1,0),  # b
                  (0,1,0)]  # d
        super().__init__(id_, coords, objtype=objtype, color=color)

        
class Smashboy(TetrisObject):
    """
    Corresponds to

        ^ z-axis
        ab
        Cd  -> y-axis
    """
    def __init__(self, id_, objtype="hero", color=(1, 1, 0.2)):
        coords = [(0,0,0),  # center "C"
                  (0,0,1), # a
                  (0,1,1), # b
                  (0,1,0)]  # d
        super().__init__(id_, coords, objtype=objtype, color=color)


class Robot(GridWorldObject):
    # By default, the robot has a camera, and
    # it looks into the direction (-1, 0, 0)
    def __init__(self, id_,
                 camera_pose,  # 6D camera pose relative to the robot
                 camera_model,
                 objtype="robot"):
                 
        super().__init__(id_, objtype=objtype)
        self._camera_pose = camera_pose
        self._camera_model = camera_model
        sx, sy, sz, sthx, sthy, sthz = self._camera_pose
        self._camera_model.transform_camera(self._camera_pose, permanent=True) #)sx, sy, sz, sthx, sthy, sthz, permanent=True)
                
    def init_render(self):
        vertices, indices, colors = util.cube(color=(1,0,0))
        self._vertex_vbo, self._index_vbo, self._color_vbo\
            = util.generate_vbo_elements(vertices, indices, colors)
        self._num_indices = len(indices)

        # vertices for axes
        axes_vertices = np.array([0,0,0,
                                  0,0,0,
                                  0,0,0,
                                  2,0,0,
                                  0,2,0,
                                  0,0,2])
        axes_colors = np.array([0.8,0.2,0.2,  # origin - red
                                0.2,0.8,0.2,  # origin - green
                                0.2,0.2,0.8,  # origin - blue
                                0.8,0.2,0.2,  # Red
                                0.2,0.8,0.2,  # Green
                                0.2,0.2,0.8]) # Blue
        axes_indices = np.array([0,3,1,4,2,5])
        self._axes_vertex_vbo, self._axes_index_vbo, self._axes_color_vbo \
            = util.generate_vbo_elements(axes_vertices, axes_indices, axes_colors)

        self._camera_model.init_render()

    def _render_axes(self):
        # Draw axes
        glEnableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, self._axes_vertex_vbo);
        glVertexPointer(3, GL_FLOAT, 0, None);
        glBindBuffer(GL_ARRAY_BUFFER, self._axes_color_vbo);
        glColorPointer(3, GL_FLOAT, 0, None);        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._axes_index_vbo);
        glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, None)
        glDisableClientState(GL_COLOR_ARRAY)

    def render(self, render_fov=False):
        if render_fov:
            glPushMatrix()
            sx, sy, sz, sthx, sthy, sthz = self._camera_pose
            glTranslatef(sx, sy, sz)
            glRotatef(sthz, 0, 0, 1)
            glRotatef(sthy, 0, 1, 0)
            glRotatef(sthx, 1, 0, 0)
            self._camera_model.render()
            glPopMatrix()

        # render robot cube
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, self._vertex_vbo);
        glVertexPointer(3, GL_FLOAT, 0, None);
        glBindBuffer(GL_ARRAY_BUFFER, self._color_vbo);
        glColorPointer(3, GL_FLOAT, 0, None);        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._index_vbo);
        glDrawElements(GL_QUADS, self._num_indices, GL_UNSIGNED_INT, None)
        glDisableClientState(GL_COLOR_ARRAY)

        # render axis
        self._render_axes()
        

    def cleanup(self):
        glDeleteBuffers(3, np.array([self._vertex_vbo, self._index_vbo, self._color_vbo]))

    @property
    def camera_model(self):
        return self._camera_model

    def camera_pose(self, robot_pose):
        """
        Returns world-frame camera pose, with rotation in quaternion.
        """
        # robot pose with respect to world frame
        rx, ry, rz, qx, qy, qz, qw = robot_pose
        rthx, rthy, rthz = util.quat_to_euler(qx, qy, qz, qw)
        # camera pose with respect to robot frame
        sx, sy, sz, sthx, sthy, sthz = self._camera_pose
        # camera pose with respect to world frame
        cam_pose_world = [rx + sx, ry + sy, rz + sz]\
            + util.R_to_quat(util.R_quat(qx, qy, qz, qw) * util.R_euler(sthx, sthy, sthz)).tolist()
        return cam_pose_world
        
