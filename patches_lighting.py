#--- Imports ---#
#region
from OpenGL.GL import *
import math
import numpy as np
import time
import imgui

import magic2 as magic
"""
    We import the 'lab_utils' module as 'lu' to save a bit of 
    typing while still clearly marking where the code came from.
"""
import lab_utils as lu
#endregion
#--- Globals ---#
#region
g_cameraDistance = 2.0
g_cameraYawDeg = 0.0
g_cameraPitchDeg = 0.0
g_yFovDeg = 45.0

g_triangleVerts1 = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]

g_triangleVerts2 = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -1],
]

g_triangleVerts3 = [
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
]

g_triangleVerts4 = [
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 1],
]



#endregion
#--- Callbacks ---#
#region
def renderFrame(width: int, height: int) -> None:
    """
        Callback which draws a frame.
        
        Parameters:
            
            width, height: size of the frame buffer, or window
    """
    global g_triangleVerts1
    global g_triangleVerts2
    global g_triangleVerts3
    global g_triangleVerts4
    global g_cameraDistance
    global g_cameraYawDeg
    global g_cameraPitchDeg
    global g_yFovDeg

    """
        This configures the fixed-function transformation from 
        Normalized Device Coordinates (NDC) to the screen 
        (pixels - called 'window coordinates' in OpenGL documentation).
        See: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glViewport.xhtml
    """
    glViewport(0, 0, width, height)

    # Set the colour we want the frame buffer cleared to, 
    glClearColor(0.8, 0.9, 1.0, 1.0); # Equal amounts of red, green, and blue for a grey color

    """
        Tell OpenGL to clear the render target to the clear values 
        for both depth and colour buffers (depth uses the default)
    """
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)

    #worldToViewTransform = lu.Mat4()
    #worldToViewTransform = magic.make_lookAt([0,0,g_cameraDistance], [0,0,0], [0,1,0])
    
    #eyePos = lu.Mat3(lu.make_rotation_y(math.radians(g_cameraYawDeg)) * lu.make_rotation_x(-math.radians(g_cameraPitchDeg))) * [0.0, 0.0, g_cameraDistance]
    #lookTarget = [0,0,0] 
    #up = [0,1,0] 
    #worldToViewTransform =  magic.make_lookAt(eyePos, lookTarget, up)
    
    eyePos = lu.Mat3(lu.make_rotation_y(math.radians(g_cameraYawDeg)) * lu.make_rotation_x(-math.radians(g_cameraPitchDeg))) * [0.0, 0.0, g_cameraDistance]
    worldToViewTransform = magic.make_lookAt(eyePos, [0,0,0], [0,1,0])
    #viewToClipTransform = lu.Mat4()
    viewToClipTransform = magic.make_perspective(g_yFovDeg, width / height, 0.01, 50.0)


    worldToClipTransform = viewToClipTransform * worldToViewTransform

    magic.draw_vertex_data_as_triangles(
        g_triangleVerts1, worldToClipTransform)
    magic.draw_vertex_data_as_triangles(
        g_triangleVerts2, worldToClipTransform)
    magic.draw_vertex_data_as_triangles(
        g_triangleVerts3, worldToClipTransform)
    magic.draw_vertex_data_as_triangles(
        g_triangleVerts4, worldToClipTransform)

    """
        Draw UI display for the render data here to save having 
        to copy transforms and data. ImGui can be used everywhere 
        in the draw function, it is just usually neater to keep UI 
        separate. But, not always!
    """
    """
    if imgui.tree_node("World Space", imgui.TREE_NODE_DEFAULT_OPEN):
        for i,(x,y,z) in enumerate(g_triangleVerts1):
            imgui.input_float3(f"v{i}", x,y,z, flags = 2)
        imgui.tree_pop()
    if imgui.tree_node("View Space", imgui.TREE_NODE_DEFAULT_OPEN):
        for i,(x,y,z) in enumerate(g_triangleVerts1):
            hx,hy,hz,hw = worldToViewTransform * [x,y,z,1]
            imgui.input_float3(f"v{i}", hx, hy, hz, flags=2)
        imgui.tree_pop()
    if imgui.tree_node("Clip Space", imgui.TREE_NODE_DEFAULT_OPEN):
        for i,(x,y,z) in enumerate(g_triangleVerts1):
            hx,hy,hz,hw = worldToClipTransform * [x,y,z,1]
            imgui.input_float4(f"v{i}", hx, hy, hz, hw, flags = 2)
        imgui.tree_pop()
    if imgui.tree_node("NDC", imgui.TREE_NODE_DEFAULT_OPEN):
        for i,(x,y,z) in enumerate(g_triangleVerts1):
            hx,hy,hz,hw = worldToClipTransform * [x,y,z,1]
            imgui.input_float3(f"v{i}", hx/hw, hy/hw, hz/hw, flags = 2)
        imgui.tree_pop()
    """

    magic.draw_coordinate_system(viewToClipTransform, worldToViewTransform)

def drawUi(width: int, height: int) -> None:
    """
        Callback to draw the UI for the frame.

        Parameters:

            width, height: the size of the framebuffer, or window.
    """
    global g_triangleVerts1
    global g_triangleVerts2
    global g_triangleVerts3
    global g_triangleVerts4
    
    global g_cameraDistance
    global g_cameraYawDeg
    global g_cameraPitchDeg
    global g_yFovDeg

    imgui.push_item_width(125)
    _,g_cameraDistance = imgui.slider_float("CameraDistance", g_cameraDistance, 1.00, 20.0)
    _,g_yFovDeg = imgui.slider_float("Y-Fov (Deg)", g_yFovDeg, 1.00, 90.0)
    _,g_cameraYawDeg = imgui.slider_float("Camera Yaw (Deg)", g_cameraYawDeg, -180.00, 180.0)
    _,g_cameraPitchDeg = imgui.slider_float("Camera Pitch (Deg)", g_cameraPitchDeg, -89.00, 89.0)
    imgui.pop_item_width()
#endregion
"""
    This does all the openGL setup and window creation needed. It 
    hides a lot of things that we will want to get a handle on as 
    time goes by.
"""
magic.run_program("COSC3000 - Computer Graphics Lab 2", 640, 480, renderFrame, None, drawUi)

#        vec2 uv = abs(2.0 * mod(v2f_modelSpaceXy.xy * 10.0, 1.0) - 1.0);
#       fragmentColor = vec4(1.0 - vec3(pow(max(uv.x, uv.y), 21.0)), 1.0);
