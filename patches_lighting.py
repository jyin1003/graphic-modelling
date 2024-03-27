#--- Imports ---#
#region
from OpenGL.GL import *
import math
import numpy as np
import time
import imgui

import magic2 as magic
from ObjModel import ObjModel
import lab_utils as lu
#endregion
#--- Globals ---#
#region
g_cameraDistance = 6.0
g_cameraYawDeg = -90.0
g_cameraPitchDeg = 0.0
g_yFovDeg = 45.0

g_cameraModel = None
g_worldSpaceLightDirection = [-1, -1, -1]

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

    global g_cameraModel
    global g_worldSpaceLightDirection

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
    cameraModelToWorldTransform = lu.Mat4()
    camera_scale_factor = 0.01
    camera_scaling_matrix = lu.make_scale(camera_scale_factor, camera_scale_factor, camera_scale_factor)
    cameraModelToWorldTransform = camera_scaling_matrix * cameraModelToWorldTransform
    cameraModelToWorldTransform = lu.make_translation(0.0, 3.0, -0.5) * cameraModelToWorldTransform
    cameraModelToWorldTransform = lu.make_rotation_x(math.radians(-50.0)) * cameraModelToWorldTransform
    drawObjModel(viewToClipTransform, worldToViewTransform, cameraModelToWorldTransform, g_cameraModel)

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
    _,g_cameraDistance = imgui.slider_float("CameraDistance", g_cameraDistance, 1.00, 30.0)
    _,g_yFovDeg = imgui.slider_float("Y-Fov (Deg)", g_yFovDeg, 1.00, 90.0)
    _,g_cameraYawDeg = imgui.slider_float("Camera Yaw (Deg)", g_cameraYawDeg, -180.00, 180.0)
    _,g_cameraPitchDeg = imgui.slider_float("Camera Pitch (Deg)", g_cameraPitchDeg, -89.00, 89.0)
    imgui.pop_item_width()
#endregion

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
    
def initResources() -> None:
    global g_cameraModel

    g_cameraModel = ObjModel("objects/10124_SLR_Camera_SG_V1_Iteration2.obj")
#endregion

magic.run_program("COSC3000 - Computer Graphics Lab 2", 640, 480, renderFrame, initResources, drawUi)

#        vec2 uv = abs(2.0 * mod(v2f_modelSpaceXy.xy * 10.0, 1.0) - 1.0);
#       fragmentColor = vec4(1.0 - vec3(pow(max(uv.x, uv.y), 21.0)), 1.0);
