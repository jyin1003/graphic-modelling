#--- Imports ---#
#region
from OpenGL.GL import *

import math
import numpy as np
import time
from ObjModel import ObjModel
import imgui

import magic
# We import the 'lab_utils' module as 'lu' to save a bit of typing while still clearly marking where the code came from.
import lab_utils as lu
#endregion
#--- Globals ---#
#region
g_wrenchModel = None
g_groundModel = None
g_crocodileModel = None

g_worldSpaceLightDirection = [-1, -1, -1]
g_cameraDistance = 200.0
g_cameraYaw = 45.0
g_cameraPitch = 40.0
g_lookTargetHeight = 6.0

#endregion
#--- Callbacks ---#
#region
def renderFrame(width: int, height: int) -> None:
    """
        Draws to draw a frame.
        
        Parameters:
            width, height: size of the frame buffer, or window
    """
    global g_wrenchModel
    global g_groundModel
    global g_crocodileModel
    
    global g_worldSpaceLightDirection
    global g_cameraDistance
    global g_cameraYaw
    global g_cameraPitch
    global g_lookTargetHeight
    """
        This configures the fixed-function transformation from 
        Normalized Device Coordinates (NDC) to the screen 
        (pixels - called 'window coordinates' in OpenGL documentation).
        See: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glViewport.xhtml
    """
    glViewport(0, 0, width, height)
    # Set the colour we want the frame buffer cleared to, 
    glClearColor(0.2, 0.3, 0.1, 1.0)
    """
        Tell OpenGL to clear the render target to the clear values 
        for both depth and colour buffers (depth uses the default)
    """
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)
    """
        Use the camera parameters to calculate the position of the 
        'eye' or camera, the viewer location
    """
    eye = lu.Mat3(lu.make_rotation_y(math.radians(g_cameraYaw)) * lu.make_rotation_x(-math.radians(g_cameraPitch))) * [0.0, 0.0, g_cameraDistance]
    target = [0, g_lookTargetHeight, 0]
    
    worldToViewTfm = magic.make_lookAt(eye, target, [0, 1, 0])
    viewToClipTfm = magic.make_perspective(45.0, width / height, 0.1, 1000.0)
    """
        rendering each of the parts of the crane in turn, 
        note however that the order of drawing is irrelevant, 
        what matteris is how the transforms are combined.
    """    
    ground_scale_factor = 15.0
    scaling_matrix = lu.make_scale(ground_scale_factor, ground_scale_factor, ground_scale_factor)

    
    wrenchModelToWorldTransform = lu.Mat4()
    drawObjModel(viewToClipTfm, worldToViewTfm, wrenchModelToWorldTransform, g_wrenchModel)
    
    groundModelToWorldTransform = lu.Mat4()
    groundModelToWorldTransform = scaling_matrix * groundModelToWorldTransform
    drawObjModel(viewToClipTfm, worldToViewTfm, groundModelToWorldTransform, g_groundModel)
    
    crocodileModelToWorldTransform = lu.Mat4()
    # -, y, x
    crocodileModelToWorldTransform = lu.make_translation(0.0, 0.0, 50.0) * crocodileModelToWorldTransform
    drawObjModel(viewToClipTfm, worldToViewTfm, crocodileModelToWorldTransform, g_crocodileModel)
    
def initResources() -> None:
    global g_wrenchModel
    global g_groundModel
    global g_crocodileModel
    
    g_wrenchModel = ObjModel("10299_Monkey_Wrench_v1.obj")
    g_groundModel = ObjModel("ground.obj")
    g_crocodileModel = ObjModel("12262_Crocodile_v1.obj")
    """
        the basic magic setup turns off backface culling, 
        here we turn it back in again.
    """
    glEnable(GL_CULL_FACE)

def drawUi() -> None:
    """
        Called by the magic at a good time to do UI stuff, 
        you can actually add UI code whenever, but it is generally 
        best to keep it to itself, from a design standpoint - 
        avoids cluttering up the rendering code.
        You can safely ignore the details of this bit of code 
        for this lab, it uses ImGui.
    """
    global g_cameraDistance
    global g_cameraYaw
    global g_cameraPitch
    global g_lookTargetHeight

    if imgui.tree_node("Camera Controls", imgui.TREE_NODE_DEFAULT_OPEN):
        _,g_cameraDistance = imgui.slider_float("CameraDistance", g_cameraDistance, 1.0, 250.0)
        _,g_cameraYaw = imgui.slider_float("CameraYaw", g_cameraYaw, 0.0, 360.0)
        _,g_cameraPitch = imgui.slider_float("CameraPitch", g_cameraPitch, -89.0, 89.0)
        _,g_lookTargetHeight = imgui.slider_float("LookTargetHeight", g_lookTargetHeight, 0.0, 25.0)
        imgui.tree_pop()
    
#--- Other Functions ---#
#region
def drawObjModel(viewToClipTfm: lu.Mat4, worldToViewTfm: lu.Mat4, 
                    modelToWorldTfm: lu.Mat4, model: ObjModel) -> None:
    """
        Set the parameters that the ObjModel implementation expects.
        Most of what happens here is beyond the scope of this lab!

        Parameters:

            viewToClipTfm: perspective projection matrix

            worldToViewTfm: lookat matrix

            modelToWorldTfm: model matrix (commonly rotation/translation)

            model: the objModel to draw
    """
    
    """
        Lighting/Shading is very often done in view space, which is 
        why a transformation that lands positions in this space is 
        needed
    """
    modelToViewTransform = worldToViewTfm * modelToWorldTfm
    
    """
        This is a special transform that ensures that normal vectors 
        remain orthogonal to the surface they are supposed to be 
        even in the presence of non-uniform scaling.
        
        It is a 3x3 matrix as vectors don't need translation anyway 
        and this transform is only for vectors, not points. 
        If there is no non-uniform scaling this is just the same as 
        Mat3(modelToViewTransform)
    """
    modelToViewNormalTransform = lu.inverse(lu.transpose(lu.Mat3(modelToViewTransform)))

    """
        Bind the shader program such that we can set the uniforms 
        (model.render sets it again)
    """
    glUseProgram(model.defaultShader)

    """
        transform (rotate) light direction into view space 
        (as this is what the ObjModel shader wants)
    """
    viewSpaceLightDirection = lu.normalize(lu.Mat3(worldToViewTfm) * g_worldSpaceLightDirection)
    glUniform3fv(glGetUniformLocation(model.defaultShader, "viewSpaceLightDirection"), 1, viewSpaceLightDirection)

    """
        This dictionary contains a few transforms that are needed 
        to render the ObjModel using the default shader. It would be 
        possible to just set the modelToWorld transform, as this is 
        the only thing that changes between the objects, and compute 
        the other matrices in the vertex shader. However, this would 
        push a lot of redundant computation to the vertex shader and 
        makes the code less self contained, in this way we set all 
        the required parameters explicitly.
    """
    transforms = {
        "modelToClipTransform" : viewToClipTfm * worldToViewTfm * modelToWorldTfm,
        "modelToViewTransform" : modelToViewTransform,
        "modelToViewNormalTransform" : modelToViewNormalTransform,
    }
    
    model.render(None, None, transforms)
#endregion
"""
    This does all the openGL setup and window creation needed.
    It hides a lot of things that we will want to get a handle on 
    as time goes by.
"""
magic.run_program(
    "Wrench", 1280, 800, 
    renderFrame, initResources, drawUi)

