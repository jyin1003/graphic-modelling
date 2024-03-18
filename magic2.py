#region
from OpenGL.GL import *
import glfw

import numpy as np
from ctypes import sizeof, c_float, c_void_p, c_uint, string_at
import math
import sys

import imgui

from collections.abc import Callable

# we use 'warnings' to remove this warning that ImGui[glfw] gives
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from imgui.integrations.glfw import GlfwRenderer as ImGuiGlfwRenderer

from lab_utils import Vec3, Mat3, Mat4, make_translation, normalize

from ObjModel import ObjModel
import lab_utils as lu
#endregion
#region
g_mousePos = [0.0, 0.0]

VAL_Position = 0
g_vertexDataBuffer = 0
g_vertexArrayObject = 0
g_simpleShader = 0

g_screenCaptureTexture = None

g_coordinateSystemModel = None

g_glfwMouseMap = {
    "MOUSE_BUTTON_LEFT" : glfw.MOUSE_BUTTON_LEFT,
    "MOUSE_BUTTON_RIGHT" : glfw.MOUSE_BUTTON_RIGHT,
    "MOUSE_BUTTON_MIDDLE" : glfw.MOUSE_BUTTON_MIDDLE,
}

g_glfwKeymap = {
    "SPACE" : glfw.KEY_SPACE,
    "APOSTROPHE" : glfw.KEY_APOSTROPHE,
    "COMMA" : glfw.KEY_COMMA,
    "MINUS" : glfw.KEY_MINUS,
    "PERIOD" : glfw.KEY_PERIOD,
    "SLASH" : glfw.KEY_SLASH,
    "0" : glfw.KEY_0,
    "1" : glfw.KEY_1,
    "2" : glfw.KEY_2,
    "3" : glfw.KEY_3,
    "4" : glfw.KEY_4,
    "5" : glfw.KEY_5,
    "6" : glfw.KEY_6,
    "7" : glfw.KEY_7,
    "8" : glfw.KEY_8,
    "9" : glfw.KEY_9,
    "SEMICOLON" : glfw.KEY_SEMICOLON,
    "EQUAL" : glfw.KEY_EQUAL,
    "A" : glfw.KEY_A,
    "B" : glfw.KEY_B,
    "C" : glfw.KEY_C,
    "D" : glfw.KEY_D,
    "E" : glfw.KEY_E,
    "F" : glfw.KEY_F,
    "G" : glfw.KEY_G,
    "H" : glfw.KEY_H,
    "I" : glfw.KEY_I,
    "J" : glfw.KEY_J,
    "K" : glfw.KEY_K,
    "L" : glfw.KEY_L,
    "M" : glfw.KEY_M,
    "N" : glfw.KEY_N,
    "O" : glfw.KEY_O,
    "P" : glfw.KEY_P,
    "Q" : glfw.KEY_Q,
    "R" : glfw.KEY_R,
    "S" : glfw.KEY_S,
    "T" : glfw.KEY_T,
    "U" : glfw.KEY_U,
    "V" : glfw.KEY_V,
    "W" : glfw.KEY_W,
    "X" : glfw.KEY_X,
    "Y" : glfw.KEY_Y,
    "Z" : glfw.KEY_Z,
    "LEFT_BRACKET" : glfw.KEY_LEFT_BRACKET,
    "BACKSLASH" : glfw.KEY_BACKSLASH,
    "RIGHT_BRACKET" : glfw.KEY_RIGHT_BRACKET,
    "GRAVE_ACCENT" : glfw.KEY_GRAVE_ACCENT,
    "WORLD_1" : glfw.KEY_WORLD_1,
    "WORLD_2" : glfw.KEY_WORLD_2,
    "ESCAPE" : glfw.KEY_ESCAPE,
    "ENTER" : glfw.KEY_ENTER,
    "TAB" : glfw.KEY_TAB,
    "BACKSPACE" : glfw.KEY_BACKSPACE,
    "INSERT" : glfw.KEY_INSERT,
    "DELETE" : glfw.KEY_DELETE,
    "RIGHT" : glfw.KEY_RIGHT,
    "LEFT" : glfw.KEY_LEFT,
    "DOWN" : glfw.KEY_DOWN,
    "UP" : glfw.KEY_UP,
    "PAGE_UP" : glfw.KEY_PAGE_UP,
    "PAGE_DOWN" : glfw.KEY_PAGE_DOWN,
    "HOME" : glfw.KEY_HOME,
    "END" : glfw.KEY_END,
    "CAPS_LOCK" : glfw.KEY_CAPS_LOCK,
    "SCROLL_LOCK" : glfw.KEY_SCROLL_LOCK,
    "NUM_LOCK" : glfw.KEY_NUM_LOCK,
    "PRINT_SCREEN" : glfw.KEY_PRINT_SCREEN,
    "PAUSE" : glfw.KEY_PAUSE,
    "F1" : glfw.KEY_F1,
    "F2" : glfw.KEY_F2,
    "F3" : glfw.KEY_F3,
    "F4" : glfw.KEY_F4,
    "F5" : glfw.KEY_F5,
    "F6" : glfw.KEY_F6,
    "F7" : glfw.KEY_F7,
    "F8" : glfw.KEY_F8,
    "F9" : glfw.KEY_F9,
    "F10" : glfw.KEY_F10,
    "F11" : glfw.KEY_F11,
    "F12" : glfw.KEY_F12,
    "F13" : glfw.KEY_F13,
    "F14" : glfw.KEY_F14,
    "F15" : glfw.KEY_F15,
    "F16" : glfw.KEY_F16,
    "F17" : glfw.KEY_F17,
    "F18" : glfw.KEY_F18,
    "F19" : glfw.KEY_F19,
    "F20" : glfw.KEY_F20,
    "F21" : glfw.KEY_F21,
    "F22" : glfw.KEY_F22,
    "F23" : glfw.KEY_F23,
    "F24" : glfw.KEY_F24,
    "F25" : glfw.KEY_F25,
    "KP_0" : glfw.KEY_KP_0,
    "KP_1" : glfw.KEY_KP_1,
    "KP_2" : glfw.KEY_KP_2,
    "KP_3" : glfw.KEY_KP_3,
    "KP_4" : glfw.KEY_KP_4,
    "KP_5" : glfw.KEY_KP_5,
    "KP_6" : glfw.KEY_KP_6,
    "KP_7" : glfw.KEY_KP_7,
    "KP_8" : glfw.KEY_KP_8,
    "KP_9" : glfw.KEY_KP_9,
    "KP_DECIMAL" : glfw.KEY_KP_DECIMAL,
    "KP_DIVIDE" : glfw.KEY_KP_DIVIDE,
    "KP_MULTIPLY" : glfw.KEY_KP_MULTIPLY,
    "KP_SUBTRACT" : glfw.KEY_KP_SUBTRACT,
    "KP_ADD" : glfw.KEY_KP_ADD,
    "KP_ENTER" : glfw.KEY_KP_ENTER,
    "KP_EQUAL" : glfw.KEY_KP_EQUAL,
    "LEFT_SHIFT" : glfw.KEY_LEFT_SHIFT,
    "LEFT_CONTROL" : glfw.KEY_LEFT_CONTROL,
    "LEFT_ALT" : glfw.KEY_LEFT_ALT,
    "LEFT_SUPER" : glfw.KEY_LEFT_SUPER,
    "RIGHT_SHIFT" : glfw.KEY_RIGHT_SHIFT,
    "RIGHT_CONTROL" : glfw.KEY_RIGHT_CONTROL,
    "RIGHT_ALT" : glfw.KEY_RIGHT_ALT,
    "RIGHT_SUPER" : glfw.KEY_RIGHT_SUPER,
    "MENU" : glfw.KEY_MENU,
}
#endregion
#region
def get_shader_info_log(obj: int) -> str:
    """
        Get the current error message from the shader.

        Parameters:

            obj: integer handle to the shader object being
            compiled or linked.
        
        Returns:

            The current error message, or an empty string
            if the operation failed silently (GOOD LUCK)
    """
    logLength = glGetShaderiv(obj, GL_INFO_LOG_LENGTH)

    if logLength > 0:
        return glGetShaderInfoLog(obj).decode()

    return ""

def compile_and_attach_shader_module(shader_program: int, 
    shader_stage: int, source_code: str) -> bool:
    """
        Compile the source code for a shader module 
        (e.g., vertex / fragment) and attaches it to the
        given shader program.

        Parameters:

            shader_program: the program to attach the compiled
            module to.

            shader_stage: the stage which the module is meant for

            source_code: the source code to be compiled.
        
        Returns:

            Whether the source code was successfully compiled.
    """

    # Create the opengl shader module object
    module = glCreateShader(shader_stage)
    # upload the source code for the shader
    # Note the function takes an array of source strings and lengths.
    glShaderSource(module, [source_code])
    glCompileShader(module)

    """
        If there is a syntax or other compiler error during shader 
        compilation, we'd like to know
    """
    compile_ok = glGetShaderiv(module, GL_COMPILE_STATUS)

    if not compile_ok:
        err = get_shader_info_log(module)
        print("SHADER COMPILE ERROR: '%s'" % err);
        return False

    glAttachShader(shader_program, module)
    glDeleteShader(module)
    return True

def build_shader(
    vertex_shader_source: str, fragment_shader_source: str, 
    attrib_locations: dict[str, int], 
    frag_data_locations: dict[str, int] = {}) -> int:
    """
        Creates a more general shader that binds a map of attribute 
        streams to the shader and the also any number of output 
        shader variables.
        The fragDataLocs can be left out for programs that don't 
        use multiple render targets as the default for any 
        output variable is zero.

        Parameters:

            vertex_shader_source, fragment_shader_source: source code
                for the vertex and fragment modules.
            
            attrib_locations: location of each attribute.
                eg. {"position": 0, "colour": 1, ...}
            
            frag_data_locations: optional, describes each colour
                attachment.

                eg. {"albedo": 0, "normal": 1, ...}
    """
    shader = glCreateProgram()

    if compile_and_attach_shader_module(
        shader, GL_VERTEX_SHADER, vertex_shader_source) \
        and compile_and_attach_shader_module(
            shader, GL_FRAGMENT_SHADER, fragment_shader_source):
	    # Link the attribute names we used in the vertex shader to the integer index
        for name, location in attrib_locations.items():
            glBindAttribLocation(shader, location, name)

        """
	        If we have multiple images bound as render targets, 
            we need to specify which 'out' variable in the fragment 
            shader goes where in this case it is totally redundant 
            as we only have one (the default render target, 
            or frame buffer) and the default binding is always zero.
        """
        for name, location in frag_data_locations.items():
            glBindFragDataLocation(shader, location, name)

        """
            once the bindings are done we can link the program stages 
            to get a complete shader pipeline. This can yield errors,
            for example if the vertex and fragment shaders don't have 
            compatible out and in variables (e.g., the fragment 
            shader expects some data that the vertex shader is not 
            outputting).
        """
        glLinkProgram(shader)
        linkStatus = glGetProgramiv(shader, GL_LINK_STATUS)
        if not linkStatus:
            err = glGetProgramInfoLog(shader)
            print("SHADER LINKER ERROR: '%s'" % err)
            sys.exit(1)
    return shader

def debug_message_callback(
    source: int, type: int, id: int, 
    severity: int, length: int, message: str, 
    userParam: c_void_p) -> None:
    print(message)

def build_basic_shader(
    vertex_shader_source: str, fragment_shader_source: int) -> int:
    """
        creates a basic shader that binds the 0th attribute stream 
        to the shader attribute "positionIn" and the output shader 
        variable 'fragmentColor' to the 0th render target 
        (the default)

        Parameters:

            vertex_shader_source, fragment_shader_source: source code
                for the vertex and fragment shader modules.
            
        Returns:

            Integer handle to the created shader program.
    """
    
    shader = glCreateProgram()

    if compile_and_attach_shader_module(
        shader, GL_VERTEX_SHADER, vertex_shader_source) \
        and compile_and_attach_shader_module(
            shader, GL_FRAGMENT_SHADER, fragment_shader_source):
        """
	        Link the name we used in the vertex shader 'positionIn' 
            to the integer index 0. This ensures that when the shader 
            executes, data fed into 'positionIn' will be sourced from 
            the 0'th generic attribute stream. This seemingly 
            backwards way of telling the shader where to look allows 
            OpenGL programs to swap vertex buffers without needing 
            to do any string lookups at run time.
        """
        glBindAttribLocation(shader, 0, "positionIn")

        """
	        If we have multiple images bound as render targets, 
            we need to specify which 'out' variable in the fragment 
            shader goes wher. In this case it is totally redundant 
            as we only have one (the default render target, or 
            frame buffer) and the default binding is always zero.
        """
        glBindFragDataLocation(shader, 0, "fragmentColor")

        """
            once the bindings are done we can link the program stages
            to get a complete shader pipeline. This can yield errors, 
            for example if the vertex and fragment modules don't have 
            compatible out and in variables (e.g., the fragment 
            module expects some data that the vertex module is not 
            outputting).
        """
        glLinkProgram(shader)
        linkStatus = glGetProgramiv(shader, GL_LINK_STATUS)
        if not linkStatus:
            err = glGetProgramInfoLog(shader)
            print("SHADER LINKER ERROR: '%s'" % err)
            sys.exit(1)
    return shader

def flatten(array: list[list[float]]) -> np.ndarray:
    """
        Turns a multidimensional array into a 1D array

        Parameters:

            array: the array to be flattened

        Returns:

            a one dimensional numpy array holding the
            original data.
    """
    data_array = np.array(array, dtype = np.float32)
    length = data_array.nbytes // data_array.itemsize
    return data_array.reshape(length)

def upload_float_data(
    buffer_object: int, float_data: list[list[float]]) -> None:
    """
        Uploads the given set of floats to the given buffer object.

        Parameters:

            buffer_object: integer handle to the buffer which
                will be written to.
            
            float_data: the data to be uploaded.
    """
    flat_data = flatten(float_data)
    """
        Upload data to the currently bound GL_ARRAY_BUFFER, 
        note that this is completely anonymous binary data, 
        no type information is retained (we'll supply that 
        later in glVertexAttribPointer)
    """
    glBindBuffer(GL_ARRAY_BUFFER, buffer_object)
    glBufferData(GL_ARRAY_BUFFER, flat_data.nbytes, flat_data, GL_STATIC_DRAW)

def create_vertex_array_object(vertex_positions: list[Vec3]) -> tuple[int]:
    """
        Create a vertex array object (mesh) from the given set of
        points.

        Parameters:

            vertex_positions: the list of points with which
                construct the mesh. Interpreted as positions.
        
        Returns:

            integer handles to the created vertex buffer and 
            vertex array objects.

            eg. vao, vbo = create_vertex_array_object(my_positions)
    """

    """
        GLuint &positionBuffer, GLuint &vertexArrayObject
        glGenX(<count>, <array of GLuint>) 
        is the typical pattern for creating objects in OpenGL.  
        Do pay attention to this idiosyncrasy as the first parameter 
        indicates the number of objects we want created.  
        Usually this is just one, but if you were to change the below
        code to '2' OpenGL would happily overwrite whatever is after
        'positionBuffer' on the stack (this leads to nasty bugs 
        that are sometimes very hard to detect 
        - i.e., this was a poor design choice!)
    """
    positionBuffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer)

    # re-package python data into something we can send to OpenGL
    flat_data = flatten(vertex_positions)
    """
        Upload data to the currently bound GL_ARRAY_BUFFER, 
        note that this is completely anonymous binary data, 
        no type information is retained 
        (we'll supply that later in glVertexAttribPointer)
    """
    glBufferData(GL_ARRAY_BUFFER, flat_data.nbytes, flat_data, GL_STATIC_DRAW)

    vertexArrayObject = glGenVertexArrays(1)
    glBindVertexArray(vertexArrayObject)
    """
        The positionBuffer is already bound to the GL_ARRAY_BUFFER 
        location. This is typical OpenGL style - you bind the buffer 
        to GL_ARRAY_BUFFER, and the vertex array object using 
        'glBindVertexArray', and then glVertexAttribPointer 
        implicitly uses both of these.  You often need to read the 
        manual or find example code. 'VAL_Position' is an integer, 
        which tells it which attribute we want to attach this data to, 
        this must be the same that we set up our shader using 
        glBindAttribLocation.  Next provide we type information 
        about the data in the buffer: 
            there are three components (x,y,z) per element (position)
            
            and they are of type 'float', and are not normalized
            
            The last arguments can be used to describe the layout 
            in more detail (stride & offset).
    
            Note: The last argument is 'pointer' and has type 
            'const void *', however, in modern OpenGL, 
            the data ALWAYS comes from the current GL_ARRAY_BUFFER 
            object, and 'pointer' is interpreted as an offset 
            (which is somewhat clumsy).
    """
    attribute_index = VAL_Position
    element_count = 3
    element_format = GL_FLOAT
    normalized = GL_FALSE
    stride = 12 #alternatively 0 can be used to indicate tightly packed data
    offset = c_void_p(0)
    glVertexAttribPointer(
        attribute_index, element_count, element_format, 
        normalized, stride, offset)
    """
        For the currently bound vertex array object, 
        enable the VAL_Position'th vertex array 
        (otherwise the data is not fed to the shader)
    """
    glEnableVertexAttribArray(attribute_index)

    """
        Unbind the buffers again to avoid unintentianal 
        GL state corruption (this is something that can 
        be rather inconventient to debug)
    """
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return (positionBuffer, vertexArrayObject)

def draw_vertex_data_as_coloured_triangles(
    triangle_vertices: list[Vec3], transform: Mat4,
    colour: np.ndarray) -> None:
    """
        Draw the given set of positions.

        Parameters:

            triangle_vertices: set of x,y,z positions to draw.

            transform: 4x4 transform matrix to apply

            colour: r,g,b,a colour to use
    """

    draw_vertex_data_as_triangles_with_shaders(triangle_vertices, 
        transform, """
    #version 330
    in vec3 positionIn;
    uniform mat4 transformationMatrix;
    void main() 
    {
	    gl_Position = transformationMatrix * vec4(positionIn, 1.0);
    }
    """,
    """
    #version 330
    uniform vec4 colour;
    out vec4 fragmentColor;
    void main() 
    {
	    fragmentColor = vec4(colour);
    }
    """, lambda shader: set_colour(shader, colour))

def set_colour(shader: int, colour: np.ndarray) -> None:
    """
        Set the "colour" uniform of the given shader.

        Parameters:

            shader: integer handle to the shader program

            colour: colour to be passed in, stored in RGBA form
    """
    glUniform4fv(glGetUniformLocation(shader, "colour"), 1, colour);

def draw_vertex_data_as_triangles(
    triangle_vertices: list[Vec3], transform: Mat4) -> None:
    """
        Draw the given set of positions.

        Parameters:

            triangle_vertices: set of x,y,z positions to draw

            transform: 4x4 transform matrix to apply
    """
    draw_vertex_data_as_triangles_with_shaders(
        triangle_vertices, transform, """
    #version 330
    in vec3 positionIn;
    uniform mat4 transformationMatrix;
    out vec3 v2f_worldSpacePosition;
    void main() 
    {
        v2f_worldSpacePosition = positionIn;
	    gl_Position = transformationMatrix * vec4(positionIn, 1.0);
    }
    """,
    """
    #version 330
    in  vec3 v2f_worldSpacePosition;
    out vec4 fragmentColor;
    void main() 
    {
	    fragmentColor = vec4(v2f_worldSpacePosition, 1.0);
    }
    """)

def draw_vertex_data_as_triangles_part3(
    triangle_vertices: list[Vec3], transform: Mat4, 
    pattern_threshold: float) -> None:
    """
        Draw the given points (part 3!)

        Parameters:

            triangle_vertices: the set of x,y,z points to draw

            transform: 4x4 transformation matrix to apply

            pattern_threshold: the pattern threshold value to use
    """

    draw_vertex_data_as_triangles_with_shaders(
        triangle_vertices, transform, """
    #version 330
    in vec3 positionIn;
    uniform mat4 transformationMatrix;
    out vec3 v2f_worldSpacePosition;
    void main() 
    {
        v2f_worldSpacePosition = positionIn;
	    gl_Position = transformationMatrix * vec4(positionIn, 1.0);
    }
    """,
    """
    #version 330
    in  vec3 v2f_worldSpacePosition;
    uniform float patternThreshold;
    out vec4 fragmentColor;
    void main() 
    {
        vec2 uv = abs(2.0 * mod(v2f_worldSpacePosition.xy * 10.0, 1.0) - 1.0);
        if (uv.x > patternThreshold || uv.y > patternThreshold)
    	    fragmentColor = vec4(vec3(0.0), 1.0);
        else
    	    fragmentColor = vec4(1.0);
    }
    """, lambda shader: set_pattern_threshold(shader, pattern_threshold))

g_userShader = 0
g_vertexShaderSourceCode = ""

def set_pattern_threshold(shader: int, pattern_threshold: float) -> None:
    """
        Set the "patternThreshold" uniform to use in rendering.

        Parameters:

            shader: integer handle to the shader program to use.

            pattern_threshold: value to set.
    """
    glUniform1f(
        glGetUniformLocation(shader, "patternThreshold"), 
        pattern_threshold)

def draw_vertex_data_as_triangles_with_shaders(
    triangle_vertices: list[Vec3], transform: Mat4, 
    vertex_shader_source: str, fragment_shader_source: str, 
    set_uniforms: Callable[[int], None] | None = None) -> None:
    """
        Draw the given set of points.

        Parameters:

            triangle_vertices: set of x,y,z points to draw

            transform: 4x4 transform matrix to apply

            vertex_shader_source, fragment_shader_source: source code
                for the vertex and fragment modules
            
            set_uniforms: optional callback to set uniform value(s).
    """
    global g_userShader
    global g_vertexArrayObject
    global g_vertexDataBuffer
    global g_vertexShaderSourceCode

    """
        We re-compile and create the shader program if the source code
        has changed. This is useful for debugging, for example to 
        re-load shaders from files without re-starting, but care 
        must be taken that all uniforms etc are set!
    """
    if g_vertexShaderSourceCode != vertex_shader_source\
        or not g_userShader:
        
        if g_userShader != 0:
            glDeleteProgram(g_userShader)
        g_vertexShaderSourceCode = vertex_shader_source
        g_userShader = build_basic_shader(vertex_shader_source, fragment_shader_source)

    """
        Fetch the location of the uniform from the shader program 
        object - usually this is done once and then the integer 
        index can be used to set the value of the uniform in the 
        rendering loop, here we re-do this every iteration 
        for illustration.
        https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGetUniformLocation.xhtml
    """
    transformUniformIndex = glGetUniformLocation(g_userShader, "transformationMatrix")

    """
        Bind ('use') current shader program
        NOTE: glGetUniformLocation takes the shader program as an 
            argument, so it does not require the shader to be bound.
            However, glUniformMatrix4fv DOES NOT take the shader 
            program as an argument and therefore it works on the 
            one that is currently bound using glUseProgram. 
            This is easy to mess up, so be careful!
    """
    glUseProgram(g_userShader)
    
    if set_uniforms:
        set_uniforms(g_userShader)

    """
        Now we set the uniform to the value of the transform matrix, 
        there are functions for different types.
        NOTE: the third argument tells OpenGL whether it should 
            transpose the matrix, OpenGL expects matrices in column 
            major-order. This is one of those pesky details that is 
            easy to get wrong and somewhat hard to debug.
        https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glUniform.xhtml
    """
    glUniformMatrix4fv(
        transformUniformIndex, 1, GL_FALSE, transform.get_data())

    # as before pipe up the vertex data.
    upload_float_data(g_vertexDataBuffer, triangle_vertices)

    # https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBindVertexArray.xhtml
    glBindVertexArray(g_vertexArrayObject)

    """
        Tell OpenGL to draw triangles using data from the currently 
        bound vertex array object by grabbing three at a time 
        vertices from the array up to len(triangleVerts) vertices, 
        something like (but in hardware)
        for(i in range(0, len(triangleVerts), 3): ...  draw triangle ...
        https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glDrawArrays.xhtml
    """
    glDrawArrays(GL_TRIANGLES, 0, len(triangle_vertices))

    """
        Unbind the shader program & vertex array object to ensure 
        it does not affect anything else (in this simple program, 
        no great risk, but otherwise it pays to be careful)
    """
    glBindVertexArray(0)
    glUseProgram(0)

def begin_imgui_hud() -> None:
    """
        Begin an imgui renderpass.
    """
    global g_mousePos
    #imgui.set_next_window_position([1.0,1.0], imgui.ALWAYS, [float(imgui.get_io().display_size.x) - 5.0, 5.0])
    #imgui.set_next_window_position([1.0,1.0], imgui.ALWAYS, [float(imgui.get_io().display_size.x) - 5.0, 5.0])
    #imgui.set_next_window_position(1.0,1.0)
    imgui.set_next_window_position(5.0, 5.0)
    #imgui.setpi
    #imgui.set_next_window_size(float(imgui.get_io().display_size.x) - 5.0, 5.0)
#, imgui.ALWAYS, [])

    #imgui.SetNextWindowBgAlpha(0.5f); // Transparent background
    imgui_flags = imgui.WINDOW_NO_RESIZE \
        | imgui.WINDOW_ALWAYS_AUTO_RESIZE  \
        | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_FOCUS_ON_APPEARING
    if imgui.begin("UI", 0, imgui_flags):
        pass#imgui.text("Mouse Position: {%0.3f, %0.3f}" %(g_mousePos[0], g_mousePos[1]))

def end_imgui_hud(impl: ImGuiGlfwRenderer) -> None:
    """
        Finish describing the imgui HUD and draw it.

        Parameters:

            impl: the underlying window renderer
    """
    imgui.end()
    imgui.render()
    impl.render(imgui.get_draw_data())

def draw_magnified_region(lower_left: list[int], 
    size: list[int], factor: float) -> None:
    """
        Draw a region of the framebuffer, with some magnification
        applied.

        Parameters:

            lower_left: lower left corner of the screen to capture
                (in pixels)
            size: (w,h) of capture region, in pixels
            factor: zoom factor to apply in drawing back the region
    """
    global g_screenCaptureTexture
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    pixel_data = glReadPixels(lower_left[0], lower_left[1], 
        size[0], size[1], GL_BGR, GL_UNSIGNED_BYTE);

    if not g_screenCaptureTexture:
        g_screenCaptureTexture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, g_screenCaptureTexture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, g_screenCaptureTexture)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size[0], size[1], 0, GL_BGR, GL_UNSIGNED_BYTE, pixel_data);
    glBindTexture(GL_TEXTURE_2D, 0)

    imgui.image(g_screenCaptureTexture, size[0] * factor, size[1] * factor, (0,1), (1,0), (1,1,1,1), (1,1,1,1) )

def draw_obj_model(view_to_clip: Mat4, world_to_view: Mat4, 
    model_to_world: Mat4, model: ObjModel) -> None:
    """
        Set the parameters that the ObjModel implementation expects.
        Most of what happens here is beyond the scope of this lab!

        Parameters:

            view_to_clip: perspective projection matrix

            world_to_view: lookat matrix

            model_to_world: model matrix (commonly rotation/translation)

            model: the objModel to draw
    """

    """
        Lighting/Shading is very often done in view space, which is 
        why a transformation that lands positions in this space is 
        needed
    """
    model_to_view = world_to_view * model_to_world
    
    """
        This is a special transform that ensures that normal vectors 
        remain orthogonal to the surface they are supposed to be 
        even in the presence of non-uniform scaling.
        
        It is a 3x3 matrix as vectors don't need translation anyway 
        and this transform is only for vectors, not points. 
        If there is no non-uniform scaling this is just the same as 
        Mat3(modelToViewTransform)
    """
    model_to_view_normal = lu.inverse(lu.transpose(lu.Mat3(model_to_view)))

    """
        Bind the shader program such that we can set the uniforms 
        (model.render sets it again)
    """
    glUseProgram(model.defaultShader)

    """
        transform (rotate) light direction into view space 
        (as this is what the ObjModel shader wants)
    """
    viewspace_light_direction = np.array((0.0, 0.0, -1.0), dtype = np.float32)
    glUniform3fv(
        glGetUniformLocation(model.defaultShader, "viewSpaceLightDirection"), 
        1, viewspace_light_direction)

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
        "modelToClipTransform" : view_to_clip * world_to_view * model_to_world,
        "modelToViewTransform" : model_to_view,
        "modelToViewNormalTransform" : model_to_view_normal,
    }
    
    model.render(None, None, transforms)

def draw_coordinate_system(
    view_to_clip: Mat4, world_to_view: Mat4) -> None:
    """
        Draw the coordinate system.

        Parameters:

            view_to_clip: the perspective projection transform

            world_to_view: the lookat transform
    """
    global g_coordinateSystemModel

    glUseProgram(g_coordinateSystemModel.defaultShader)
    viewspace_light_direction = np.array((0.0, 0.0, -1.0), dtype=np.float32)
    glUniform3fv(
        glGetUniformLocation(g_coordinateSystemModel.defaultShader, "viewSpaceLightDirection"), 
        1, viewspace_light_direction)

    transforms = {
        "modelToClipTransform" : view_to_clip * world_to_view,
        "modelToViewTransform" : world_to_view,
        "modelToViewNormalTransform" : Mat3(world_to_view),
    }
    
    g_coordinateSystemModel.render(None, None, transforms)

g_numMsaaSamples = 8
g_currentMsaaSamples = 1

def init_glfw_and_resources(
    title: str, start_width: int, start_height: int, 
    init_resources: Callable[[], None] | None) -> "tuple[glfw.Window, ImGuiGlfwRenderer]":
    """
        Initalize glfw along with imgui, and do any additional
        resource initialization requested.

        Parameters:

            title: window title

            start_width, start_height: size of the window

            init_resources: optional callback to perform additional
                resource creation
        
        Returns:

            A handle to the window and the imgui implementation for it.

            eg. window, impl = init_glfw_and_resources("My window", 640, 480, None)
    """

    global g_simpleShader
    global g_vertexArrayObject
    global g_vertexDataBuffer
    global g_mousePos
    global g_coordinateSystemModel
    global g_numMsaaSamples 
    global g_currentMsaaSamples

    #glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, 1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    #glfw.window_hint(glfw.SAMPLES, g_currentMsaaSamples)

    window = glfw.create_window(start_width, start_height, title, None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window)

    print("--------------------------------------\nOpenGL\n  Vendor: %s\n  Renderer: %s\n  Version: %s\n--------------------------------------\n" % (glGetString(GL_VENDOR).decode("utf8"), glGetString(GL_RENDERER).decode("utf8"), glGetString(GL_VERSION).decode("utf8")), flush=True)
    imgui.create_context()
    impl = ImGuiGlfwRenderer(window)

    #glDebugMessageCallback(GLDEBUGPROC(debugMessageCallback), None)

    # (although this glEnable(GL_DEBUG_OUTPUT) should not have been needed when
    # using the GLUT_DEBUG flag above...)
    #glEnable(GL_DEBUG_OUTPUT)
    # This ensures that the callback is done in the context of the calling
    # function, which means it will be on the stack in the debugger, which makes it
    # a lot easier to figure out why it happened.
    #glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS)

    g_simpleShader = build_basic_shader(
    """
    #version 330
    in vec3 positionIn;

    void main() 
    {
	    gl_Position = vec4(positionIn, 1.0);
    }
    """,
    """
    #version 330
    out vec4 fragmentColor;

    void main() 
    {
	    fragmentColor = vec4(1.0);
    }
    """);

    # Create Vertex array object and buffer with dummy data for now, we'll fill it later when rendering the frame
    (g_vertexDataBuffer, g_vertexArrayObject) = create_vertex_array_object(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    glDisable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    #glEnable(GL_DEPTH_CLAMP)

    if init_resources:
        init_resources()

    g_coordinateSystemModel = ObjModel("data/coordinate_system.obj");

    return window,impl

def cleaup_gl_resources() -> None:
    """
        Cleanup any unneeded opengl objects, this is performed
        before creating a new context
    """

    global g_userShader
    global g_screenCaptureTexture

    if g_userShader:
        glDeleteProgram(g_userShader)
        g_userShader = 0
    if g_screenCaptureTexture:
        glDeleteTextures(g_screenCaptureTexture)
        g_screenCaptureTexture = None

def run_program(title: str, start_width: int, start_height: int, 
    render_frame: Callable[[int, int], None], 
    init_resources: Callable[[], None] | None = None, 
    draw_ui: Callable[[int, int], None] | None = None, 
    update: Callable[[float, dict[str, bool], list[float]], None] | None = None) -> None:
    """
        Run the program.

        Parameters:

            title: window title

            start_width, start_height: window size

            render_frame: callback function to draw a frame

            init_resources: optional callback to initialize resources

            draw_ui: optional callback to draw the imgui gui

            update: optional callback to update the program state
                each frame
    """
    global g_simpleShader
    global g_vertexArrayObject
    global g_vertexDataBuffer
    global g_mousePos
    global g_coordinateSystemModel
    global g_numMsaaSamples 
    global g_currentMsaaSamples

    if not glfw.init():
        sys.exit(1)

    window, impl = init_glfw_and_resources(
        title, start_width, start_height, init_resources)

    current_time = glfw.get_time()
    prevMouseX,prevMouseY = glfw.get_cursor_pos(window)

    while not glfw.window_should_close(window):
        prev_time = current_time
        current_time = glfw.get_time()
        dt = current_time - prev_time

        key_state = {}
        for name,id in g_glfwKeymap.items():
            key_state[name] = glfw.get_key(window, id) == glfw.PRESS

        for name,id in g_glfwMouseMap.items():
            key_state[name] = glfw.get_mouse_button(window, id) == glfw.PRESS

        mouseX,mouseY = glfw.get_cursor_pos(window)
        g_mousePos = [mouseX,mouseY]

        # Udpate 'game logic'
        if update:
            imIo = imgui.get_io()
            mouse_delta = [mouseX - prevMouseX,mouseY - prevMouseY]
            if imIo.want_capture_mouse:
                mouse_delta = [0,0]
            update(dt, key_state, mouse_delta)
        prevMouseX,prevMouseY = mouseX,mouseY

        width, height = glfw.get_framebuffer_size(window)

        imgui.new_frame()

        begin_imgui_hud()

        if draw_ui:
            draw_ui(width, height)

        render_frame(width, height)
    
        #drawCoordinateSystem()

        #mgui.show_test_window()

        end_imgui_hud(impl)
        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()
        impl.process_inputs()


    glfw.terminate()

def setup_fbo(msaa_fbo: int, fbo_width: int, fbo_height: int, 
    sample_count: int, color_renderbuffer: int = 0, 
    depth_renderbuffer: int = 0) -> tuple[int, int]:
    """
        Creates render buffers for a Frame Buffer Object (FBO) 
        for rendering to. An FBO can be thought of as an off-screen 
        full-screen window. To store the pixels, and depth buffer 
        samples we create and attach 'render buffers', using 
        glRenderbufferStorageMultisample we can set a number of 
        samples per pixel for multisampling.
        glRenderbufferStorageMultisample: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glRenderbufferStorageMultisample.xhtml
        
        It is also possible to use the contents of the buffers as 
        textures, but then you must attach a texture to the 
        framebuffer using
        glFramebufferTexture: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glFramebufferTexture.xhtml
        
        In this lab, we create the off-screen FBO to enable 
        multi-sampling AA (and more, to make it easy to play with 
        the number of sampels at run-time) this is easiest using the 
        render-buffer APIs. The call to 'glBlitFramebuffer' 
        'resolves' i.e, averages the multi-sampled image to the 
        non-MSAA default frame buffer (Note that creating a MSAA 
        frame buffer and blitting to it may cause problems).

        Parameters:

            msaa_fbo: the framebuffer object to attach renderbuffers
                to
            
            fbo_width, fbo_height: size of the framebuffer object
                (in pixels) 
            
            sample_count: number of samples to use with msaa
            
            color_renderbuffer: color attachment, if one exists 
            
            depth_renderbuffer: depth buffer, if one exists
        
        Returns:

            the configured color and depth renderbuffers.
    """
    if not color_renderbuffer:
        color_renderbuffer, depth_renderbuffer = glGenRenderbuffers(2)
    glBindRenderbuffer(GL_RENDERBUFFER, color_renderbuffer)
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, sample_count, GL_RGB8, fbo_width, fbo_height)
    glBindRenderbuffer(GL_RENDERBUFFER, depth_renderbuffer)
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, sample_count, GL_DEPTH_COMPONENT32, fbo_width, fbo_height)
    glBindRenderbuffer(GL_RENDERBUFFER, 0)

    glBindFramebuffer(GL_FRAMEBUFFER, msaa_fbo)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_renderbuffer)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_renderbuffer)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return color_renderbuffer, depth_renderbuffer

def run_program_msaa(title: str, start_width: int, 
    start_height: int, sample_count: int, 
    render_frame: Callable[[int, int], None], 
    init_resources: Callable[[], None] | None = None, 
    draw_ui: Callable[[int, int], None] | None = None, 
    update: Callable[[float, dict[str, bool], list[float]], None] | None = None) -> None:
    """
        Run the program, now with multisample antialiasing!

        Parameters:

            title: window title

            start_width, start_height: window size

            render_frame: callback function to draw a frame

            init_resources: optional callback to initialize resources

            draw_ui: optional callback to draw the imgui gui

            update: optional callback to update the program state
                each frame
    """
    global g_simpleShader
    global g_vertexArrayObject
    global g_vertexDataBuffer
    global g_mousePos
    global g_coordinateSystemModel
    global g_numMsaaSamples 
    global g_currentMsaaSamples

    if not glfw.init():
        sys.exit(1)

    window, impl = init_glfw_and_resources(
        title, start_width, start_height, init_resources)

    currentTime = glfw.get_time()
    prevMouseX,prevMouseY = glfw.get_cursor_pos(window)

    msaaFbo = glGenFramebuffers(1)
    # Lazy create  
    fboWidth = start_width
    fboHeight = start_height
    g_numMsaaSamples = sample_count
    g_currentMsaaSamples = sample_count

    color_renderbuffer, depth_renderbuffer = setup_fbo(
        msaaFbo, fboWidth, fboHeight, g_currentMsaaSamples)

    while not glfw.window_should_close(window):
        # Udpate 'game logic'
        prevTime = currentTime
        currentTime = glfw.get_time()
        dt = currentTime - prevTime

        keyStateMap = {}
        for name,id in g_glfwKeymap.items():
            keyStateMap[name] = glfw.get_key(window, id) == glfw.PRESS

        for name,id in g_glfwMouseMap.items():
            keyStateMap[name] = glfw.get_mouse_button(window, id) == glfw.PRESS

        mouseX,mouseY = glfw.get_cursor_pos(window)
        g_mousePos = [mouseX,mouseY]

        # Udpate 'game logic'
        if update:
            imIo = imgui.get_io()
            mouseDelta = [mouseX - prevMouseX,mouseY - prevMouseY]
            if imIo.want_capture_mouse:
                mouseDelta = [0,0]
            update(dt, keyStateMap, mouseDelta)
        prevMouseX,prevMouseY = mouseX,mouseY

        # Render here, e.g.  using pyOpenGL
        width, height = glfw.get_framebuffer_size(window)
    
        # in this case, re-create the FBO texture
        if fboWidth != width or  fboHeight != height or g_numMsaaSamples != g_currentMsaaSamples:
            fboWidth  = max(width, fboWidth)
            fboHeight = max(height, fboHeight)
            g_currentMsaaSamples = g_numMsaaSamples
            color_renderbuffer, depth_renderbuffer = setup_fbo(
                msaaFbo, fboWidth, fboHeight, g_currentMsaaSamples, 
                color_renderbuffer, depth_renderbuffer)

        imgui.new_frame()

        begin_imgui_hud()

        # Note: here we bind the off-screen frame buffer before calling the program to draw
        # since we take care to make sure it is the same size as the window, the rendering code does not need to know.
        glBindFramebuffer(GL_FRAMEBUFFER, msaaFbo)
    
        render_frame(width, height)
    
        #drawCoordinateSystem()

        #mgui.show_test_window()
        """
            reset the frame buffer binding to the default 
            (i.e., window-visible one)
        """
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, msaaFbo)
        """
            Copy the data from the render buffers attached to the
            MSAA FBO (bound to 'GL_READ_FRAMEBUFFER') into the 
            default FBO (bound to 'GL_FRAMEBUFFER')
        """
        glBlitFramebuffer(0,0,width,height,0,0,width,height, GL_COLOR_BUFFER_BIT, GL_LINEAR)
        """
            Reset the 'GL_READ_FRAMEBUFFER' binding point, 
            just in case...
        """
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)

        """
            we draw the UI after binding the default frame buffer 
            and resolving so we can easily copy a portion to 
            display the zoomed in view (otherwise it might have 
            made more sense to draw the UI with AA as well).
        """
        if draw_ui:
            draw_ui(width, height)

        end_imgui_hud(impl)
        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()
        impl.process_inputs()

    glfw.terminate()

def make_lookFrom(eye: np.ndarray, direction: np.ndarray, 
    up: np.ndarray) -> Mat4:
    """
        The reason we need a 'look from', and don't just use 
        lookAt(pos, pos+dir, up) is because if pos is large 
        (i.e., far from the origin) and 'dir' is a unit vector 
        (common case) then the precision loss in the addition 
        followed by subtraction in lookAt to get the direction 
        back is _significant_, and leads to jerky camera movements.

        Parameters:

            eye: camera position

            direction: camera direction

            up: camera's up vector
    """
    f = normalize(direction)
    U = np.array(up[:3], dtype=np.float32)
    s = normalize(np.cross(f, U))
    u = normalize(np.cross(s, f))

    """
        {side, up, forwards} now form an orthonormal
        basis for R3, being unit vectors in the camera's
        local {x,y,z} axes respectively.

        Now we use this to produce the affine inverse of
        the camera's frame of reference.
    """

    M = np.array((
        (           s[0],            u[0],          -f[0], 0.0),
        (           s[1],            u[1],          -f[1], 0.0),
        (           s[2],            u[2],          -f[2], 0.0),
        (-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye), 1.0)), 
        dtype=np.float32)
    return Mat4(M)

def make_lookAt(eye: np.ndarray, target: np.ndarray, 
    up: np.ndarray) -> Mat4:
    """
        Constructs and returns a 4x4 matrix representing a 
        view transform, i.e., from world to view space.
        this is basically the same as what we saw in 
        Lecture #2 for placing the car in the world, 
        except the inverse! (and also view-space 'forwards' is the 
        negative z-axis)

        Parameters:

            eye: The camera's position

            target: Point being looked at

            up: The camera's up vector (helps to orient the camera).
        
        Returns:

            An appropriate view transform.
    """
    return make_lookFrom(eye, 
        np.array(target[:3], dtype = np.float32) - np.array(eye[:3], dtype = np.float32), 
        up)

def make_perspective(
    fovy: float, aspect: float, n: float, f: float) -> Mat4:
    """
        Make a perspective projection matrix.

        Parameters:

            fovy: field of view (in degrees)

            aspect: aspect ratio of the screen (w/h)

            n: near distance

            f: far distance
        
        Returns:

            The perspective projection matrix.
    """
    radFovY = math.radians(fovy)
    tanHalfFovY = math.tan(radFovY / 2.0)
    sx = 1.0 / (tanHalfFovY * aspect)
    sy = 1.0 / tanHalfFovY
    zz = -(f + n) / (f - n)
    zw = -(2.0 * f * n) / (f - n)

    return Mat4([[sx,  0,  0,  0],
                 [ 0, sy,  0,  0],
                 [ 0,  0, zz, -1],
                 [ 0,  0, zw,  0]])

def get_uniform_location_debug(shader_program: int, name: str) -> int:
    """
        Attempt to fetch the location of the given uniform within the
        given shader program. Prints out a helpful message upon failure.
    """
    loc = glGetUniformLocation(shader_program, name)
    # Useful point for debugging, replace with silencable logging 
    #if loc == -1:
    #    print(f"Uniform \'{name}\' was not found")
    return loc
#endregion