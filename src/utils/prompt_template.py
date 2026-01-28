"""prompt_template.py: Contains the defined prompt templates for the LLM."""

__author__      = "Ravidu Silva"

from dataclasses import dataclass
@dataclass
class PromptChatTemplate:
    system_prompt_template: str
    user_prompt_template: str
    completion_prompt_template: str = ""

class PromptTemplate:

    MANIM_PLAN_GEN_CHAT_PROMPT = PromptChatTemplate(
        system_prompt_template="""You are an expert Manim Community Edition (ManimCE) code generation planner.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that I want to create.
2. Generate only a plan for the code, wrapped between `<PLAN>` and `</PLAN>`.
3. Do not include any explanations, comments, or instructions on how to run the code. Only include the plan and no code.

Example format:
<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<PLAN>
```text
1. Create a square with color red.
2. Center the square on the screen.
3. Use the play method to create the square.
4. Create a circle with the same color.
5. Use the play method to transform the square into a circle.
6. Use the wait method to pause the animation.
```
</PLAN>""",
        user_prompt_template="""Generate a plan for the following text script:
<TEXT_SCRIPT>
{reviewed_description}
</TEXT_SCRIPT>
"""
    )
    MANIM_VIDEO_GEN_W_PLAN_CHAT_TEMPLATE = PromptChatTemplate(
        system_prompt_template="""You are an expert Manim Community Edition (ManimCE) educator and Python developer.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that I want to create.
2. Analyze the plan provided in the `<PLAN>` block.
3. Generate only executable Python code for Manim, wrapped between `<CODE>` and `</CODE>`.
4. DO NOT include any explanations, comments, or instructions on how to run the code. Only include the code.


Example format:

<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<PLAN>
```text
1. Create a square with color red.
2. Center the square on the screen.
3. Use the play method to create the square.
4. Create a circle with the same color.
5. Use the play method to transform the square into a circle.
6. Use the wait method to pause the animation.
```
</PLAN>

<CODE>
```python
from manim import *
class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        circle = Circle(color=RED)
        self.play(Transform(square, circle))
        self.wait()
``` 
</CODE>""",
        user_prompt_template="""Generate code for the following text script based on the provided plan:
<TEXT_SCRIPT>
{reviewed_description}
</TEXT_SCRIPT>

<PLAN>
```text
{code_plan}
```
</PLAN>

"""
    )

    MANIM_VIDEO_GEN_CHAT_TEMPLATE = PromptChatTemplate(
        system_prompt_template="""You are an expert Manim Community Edition (ManimCE) educator and Python developer.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that user want to create.
2. DO NOT provide any explanations.
3. Generate only executable Python code for Manim, wrapped between `<CODE>` and `</CODE>`.
4. Do not include any explanations, comments, or instructions on how to run the code. Only include the code wrapped between `<CODE>` and `</CODE>`.


Example format:

<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<CODE>
```python
from manim import *
class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        circle = Circle(color=RED)
        self.play(Transform(square, circle))
        self.wait()
``` 
</CODE>"""
,
        user_prompt_template="""Generate code for the following text script:
<TEXT_SCRIPT>
{reviewed_description}
</TEXT_SCRIPT>

"""
,
        completion_prompt_template="""<CODE>
```python
{code}
```
</CODE>"""
    )

    MANIM_VIDEO_GEN_CHAT_RAG_FB_TEMPLATE = PromptChatTemplate(
        system_prompt_template="""You are an expert Manim Community Edition (ManimCE) educator and Python developer.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that user want to create.
2. Analyze the initial code provided in the `<INITIAL_CODE>` block.
3. Identify any incorrect or missing API calls in the initial code based on the provided API information in the `<API_INFO>` block.
4. Use the provided API information in the `<API_INFO>` and render errors (if any) in `<RENDER_ERRORS>` block to revise and fix the initial code.
5. Ensure the revised code correctly implements the user script using the provided API information in the `<API_INFO>` block.
6. Generate only executable Python code for Manim, wrapped between `<CODE>` and `</CODE>`.
7. DO NOT include any explanations, comments, or instructions on how to run the code. Only include the code wrapped between `<CODE>` and `</CODE>`.


Example format:

<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<INITIAL_CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(bg_color=RED)
        self.play(Create(square))
        circle = Circle(bg_color=RED)
        self.play(Transform(square, circle))
        self.wait()
``` 
</INITIAL_CODE>

<API_INFO>
```text
- **Circle**: signature `(radius: 'float | None' = None, color: 'ParsableManimColor' = ManimColor('#FC6255'), **kwargs: 'Any') -> 'None'` - A circle.

Parameters
----------
color
    The color of the shape.
kwargs
    Additional arguments to be passed to :class:`Arc`

- **Create**: signature `(mobject: 'VMobject | OpenGLVMobject | OpenGLSurface', lag_ratio: 'float' = 1.0, introducer: 'bool' = True, **kwargs) -> 'None'` - Incrementally show a VMobject.

Parameters
----------
mobject
    The VMobject to animate.

Raises
------
:class:`TypeError`
    If ``mobject`` is not an instance of :class:`~.VMobject`.

- **Square**: signature `(side_length: 'float' = 2.0, **kwargs: 'Any') -> 'None'` - A rectangle with equal side lengths.

Parameters
----------
side_length
    The length of the sides of the square.
kwargs
    Additional arguments to be passed to :class:`Rectangle`.

- **Transform**: signature `(mobject: 'Mobject | None', target_mobject: 'Mobject | None' = None, path_func: 'Callable | None' = None, path_arc: 'float' = 0, path_arc_axis: 'np.ndarray' = array([0., 0., 1.]), path_arc_centers: 'np.ndarray' = None, replace_mobject_with_target_in_scene: 'bool' = False, **kwargs) -> 'None'` - A Transform transforms a Mobject into a target Mobject.

Parameters
----------
mobject
    The :class:`.Mobject` to be transformed. It will be mutated to become the ``target_mobject``.
target_mobject
    The target of the transformation.
path_func
    A function defining the path that the points of the ``mobject`` are being moved
    along until they match the points of the ``target_mobject``, see :mod:`.utils.paths`.
path_arc
    The arc angle (in radians) that the points of ``mobject`` will follow to reach
    the points of the target if using a circular path arc, see ``path_arc_centers``.
    See also :func:`manim.utils.paths.path_along_arc`.
path_arc_axis
    The axis to rotate along if using a circular path arc, see ``path_arc_centers``.
path_arc_centers
    The center of the circular arcs along which the points of ``mobject`` are
    moved by the transformation.

    If this is set and ``path_func`` is not set, then a ``path_along_circles`` path will be generated
    using the ``path_arc`` parameters and stored in ``path_func``. If ``path_func`` is set, this and the
    other ``path_arc`` fields are set as attributes, but a ``path_func`` is not generated from it.
replace_mobject_with_target_in_scene
    Controls which mobject is replaced when the transformation is complete.

    If set to True, ``mobject`` will be removed from the scene and ``target_mobject`` will
    replace it. Otherwise, ``target_mobject`` is never added and ``mobject`` just takes its shape.

```
</API_INFO>

<RENDER_ERRORS>
```text
TypeError: Mobject.__init__() got an unexpected keyword argument 'bg_color'
```
</RENDER_ERRORS>

<CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        circle = Circle(color=RED)
        self.play(Transform(square, circle))
        self.wait()
```
</CODE>""",
        user_prompt_template="""Generate code for the following text script adapting initial code based on the provided API information and render errors occured from the initial code:

<TEXT_SCRIPT>
{reviewed_description}
</TEXT_SCRIPT>

<INITIAL_CODE>
```python
{initial_code}
```
</INITIAL_CODE>

<API_INFO>
```text
{api_info}
```
</API_INFO>

<RENDER_ERRORS>
```text
{render_errors}
```
</RENDER_ERRORS>

"""
    )

    MANIM_VIDEO_GEN_CHAT_RAG_ONLY_TEMPLATE = PromptChatTemplate(
        system_prompt_template="""You are an expert Manim Community Edition (ManimCE) educator and Python developer.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that user want to create.
2. Analyze the initial code provided in the `<INITIAL_CODE>` block.
3. Identify any incorrect or missing API calls in the initial code based on the provided API information in the `<API_INFO>` block.
4. Use the provided API information in the `<API_INFO>` block to revise and fix the initial code.
5. Ensure the revised code correctly implements the user script using the provided API information in the `<API_INFO>` block.
6. Generate only executable Python code for Manim, wrapped between `<CODE>` and `</CODE>`.
7. DO NOT include any explanations, comments, or instructions on how to run the code. Only include the code wrapped between `<CODE>` and `</CODE>`.


Example format:

<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<INITIAL_CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(bg_color=RED)
        self.play(Create(square))
        circle = Circle(bg_color=RED)
        self.play(Transform(square, circle))
        self.wait()
``` 
</INITIAL_CODE>

<API_INFO>
```text
- **Circle**: signature `(radius: 'float | None' = None, color: 'ParsableManimColor' = ManimColor('#FC6255'), **kwargs: 'Any') -> 'None'` - A circle.

Parameters
----------
color
    The color of the shape.
kwargs
    Additional arguments to be passed to :class:`Arc`

- **Create**: signature `(mobject: 'VMobject | OpenGLVMobject | OpenGLSurface', lag_ratio: 'float' = 1.0, introducer: 'bool' = True, **kwargs) -> 'None'` - Incrementally show a VMobject.

Parameters
----------
mobject
    The VMobject to animate.

Raises
------
:class:`TypeError`
    If ``mobject`` is not an instance of :class:`~.VMobject`.

- **Square**: signature `(side_length: 'float' = 2.0, **kwargs: 'Any') -> 'None'` - A rectangle with equal side lengths.

Parameters
----------
side_length
    The length of the sides of the square.
kwargs
    Additional arguments to be passed to :class:`Rectangle`.

- **Transform**: signature `(mobject: 'Mobject | None', target_mobject: 'Mobject | None' = None, path_func: 'Callable | None' = None, path_arc: 'float' = 0, path_arc_axis: 'np.ndarray' = array([0., 0., 1.]), path_arc_centers: 'np.ndarray' = None, replace_mobject_with_target_in_scene: 'bool' = False, **kwargs) -> 'None'` - A Transform transforms a Mobject into a target Mobject.

Parameters
----------
mobject
    The :class:`.Mobject` to be transformed. It will be mutated to become the ``target_mobject``.
target_mobject
    The target of the transformation.
path_func
    A function defining the path that the points of the ``mobject`` are being moved
    along until they match the points of the ``target_mobject``, see :mod:`.utils.paths`.
path_arc
    The arc angle (in radians) that the points of ``mobject`` will follow to reach
    the points of the target if using a circular path arc, see ``path_arc_centers``.
    See also :func:`manim.utils.paths.path_along_arc`.
path_arc_axis
    The axis to rotate along if using a circular path arc, see ``path_arc_centers``.
path_arc_centers
    The center of the circular arcs along which the points of ``mobject`` are
    moved by the transformation.

    If this is set and ``path_func`` is not set, then a ``path_along_circles`` path will be generated
    using the ``path_arc`` parameters and stored in ``path_func``. If ``path_func`` is set, this and the
    other ``path_arc`` fields are set as attributes, but a ``path_func`` is not generated from it.
replace_mobject_with_target_in_scene
    Controls which mobject is replaced when the transformation is complete.

    If set to True, ``mobject`` will be removed from the scene and ``target_mobject`` will
    replace it. Otherwise, ``target_mobject`` is never added and ``mobject`` just takes its shape.

```
</API_INFO>

<CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        circle = Circle(color=RED)
        self.play(Transform(square, circle))
        self.wait()
```
</CODE>""",
        user_prompt_template="""Generate code for the following text script adapting initial code based on the provided API information retrieved from the initial code:

<TEXT_SCRIPT>
{reviewed_description}
</TEXT_SCRIPT>

<INITIAL_CODE>
```python
{initial_code}
```
</INITIAL_CODE>

<API_INFO>
```text
{api_info}
```
</API_INFO>

"""
    )
    
    MANIM_VIDEO_GEN_CHAT_FB_ONLY_TEMPLATE = PromptChatTemplate(
        system_prompt_template="""You are an expert Manim Community Edition (ManimCE) educator and Python developer.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that I want to create.
2. Analyze the initial code provided in the `<INITIAL_CODE>` block.
3. Use the provided render errors (if any) in `<RENDER_ERRORS>` block to revise and fix the initial code.
4. Generate only executable Python code for Manim, wrapped between `<CODE>` and `</CODE>`.
5. DO NOT include any explanations, comments, or instructions on how to run the code. Only include the code wrapped between `<CODE>` and `</CODE>`.


Example format:

<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<INITIAL_CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(bg_color=RED)
        self.play(Create(square))
        circle = Circle(bg_color=RED)
        self.play(Transform(square, circle))
        self.wait()
``` 
</INITIAL_CODE>

<RENDER_ERRORS>
```text
TypeError: Mobject.__init__() got an unexpected keyword argument 'bg_color'
```
</RENDER_ERRORS>

<CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        circle = Circle(color=RED)
        self.play(Transform(square, circle))
        self.wait()
```
</CODE>""",
        user_prompt_template="""Generate code for the following text script adapting initial code based on the render errors occured from the initial code:

<TEXT_SCRIPT>
{reviewed_description}
</TEXT_SCRIPT>

<INITIAL_CODE>
```python
{initial_code}
```
</INITIAL_CODE>

<RENDER_ERRORS>
```text
{render_errors}
```
</RENDER_ERRORS>

"""
    )

  

    MANIM_VIDEO_GEN_PROMPT = \
"""You are an expert Manim Community Edition (ManimCE) educator and Python developer.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that I want to create.
2. DO NOT provide any explanations.
3. Generate only executable Python code for Manim, wrapped between `<CODE>` and `</CODE>`.
4. Do not include any explanations, comments, or instructions on how to run the code. Only include the code.


Example format:

<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<CODE>
```python
from manim import *
class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        circle = Circle(color=RED)
        self.play(Transform(square, circle))
        self.wait()
``` 
</CODE>

Now, generate code for the following text script:

<TEXT_SCRIPT>
{reviewed_description}
</TEXT_SCRIPT>

"""
    MANIM_VIDEO_GEN_W_THINKING_PROMPT = \
"""You are an expert Manim Community Edition (ManimCE) educator and Python developer.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that I want to create.
2. Analyze the `<think>` block which contains the pre-generated thinking content that can help you generate the code.
3. DO NOT provide any explanations.
4. Generate only executable Python code for Manim, wrapped between `<CODE>` and `</CODE>`.
5. Do not include any explanations, comments, or instructions on how to run the code. Only include the code.


Example format:

<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<think>
Okay, let's tackle this. The user wants a ManimCE animation where a red square turns into a circle. First, I need to create a square, color it red, then transform it into a circle.

So, the steps are: import necessary modules. Then, create a class that inherits from Scene. In the construct method, add the square. Wait, the square needs to be centered. Maybe using Square() from manim.mobject.geometry, and set color to RED. Then, use the transform method to change it into a Circle. Oh right, transform is a method in Manim that smoothly transitions between two mobjects. So, the code would be something like self.play(Transform(square, circle)). But I need to make sure the circle is the same size as the square. Wait, maybe the square is created first, then the circle is created with the same dimensions. Or maybe the square is transformed into a circle, which might require adjusting the radius. Alternatively, maybe using a Circle with the same stroke width as the square. Let me think. The square's side length is 2 units by default, so the circle's radius should be 1 to match. But maybe the user just wants a simple transformation. Let me check the standard example. Oh right, the Transform function can handle that. So the code would be:

from manim import *

class SquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        self.play(Transform(square, Circle()))
        self.wait()

Wait, but the circle might not have the same size. The Square is 2x2, and the Circle's default radius is 1, so it should fit. But maybe the user wants the circle to be the same size as the square. Alternatively, maybe the square is transformed into a circle with the same width. Hmm. The default Square has a side length of 2, so the circle's radius would be 1. So the circle would have a diameter of 2, same as the square's side. That should work. So the code should be correct. I need to make sure that the square is created, then transformed into a circle. Also, the color is set to red. The user said "red square", so the square is red, and the circle is probably the same color. But the problem says "transform it into a circle", so the circle would inherit the color. So the code is as above. Then, the user wants the code wrapped in <CODE> tags. So the final code should be that.
</think>

<CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        self.play(Transform(square, Circle()))
        self.wait()
``` 
</CODE>

Now, generate code for the following text script based on the provided thinking content:

<TEXT_SCRIPT>
{reviewed_description}
</TEXT_SCRIPT>

<think>
{thinking_content}
</think>

"""

    MANIM_VIDEO_GEN_W_RAG_PROMPT = \
"""You are an expert Manim Community Edition (ManimCE) educator and Python developer.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that I want to create.
2. Analyze the initial code provided in the `<INITIAL_CODE>` block.
3. Identify any incorrect or missing API calls in the initial code based on the provided API information in the `<API_INFO>` block.
4. Use the provided API information in the `<API_INFO>` block to revise the initial code.
5. Ensure the revised code correctly implements the user script using the provided API information in the `<API_INFO>` block.
6. Generate only executable Python code for Manim, wrapped between `<CODE>` and `</CODE>`.
7. DO NOT include any explanations, comments, or instructions on how to run the code. Only include the code.


Example format:

<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<INITIAL_CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(bg_color=RED)
        self.play(Create(square))
        circle = Circle(bg_color=RED)
        self.play(Transform(square, circle))
        self.wait()
``` 
</INITIAL_CODE>

<API_INFO>
```text
- **Circle**: signature `(radius: 'float | None' = None, color: 'ParsableManimColor' = ManimColor('#FC6255'), **kwargs: 'Any') -> 'None'` - A circle.

Parameters
----------
color
    The color of the shape.
kwargs
    Additional arguments to be passed to :class:`Arc`

- **Create**: signature `(mobject: 'VMobject | OpenGLVMobject | OpenGLSurface', lag_ratio: 'float' = 1.0, introducer: 'bool' = True, **kwargs) -> 'None'` - Incrementally show a VMobject.

Parameters
----------
mobject
    The VMobject to animate.

Raises
------
:class:`TypeError`
    If ``mobject`` is not an instance of :class:`~.VMobject`.

- **Square**: signature `(side_length: 'float' = 2.0, **kwargs: 'Any') -> 'None'` - A rectangle with equal side lengths.

Parameters
----------
side_length
    The length of the sides of the square.
kwargs
    Additional arguments to be passed to :class:`Rectangle`.

- **Transform**: signature `(mobject: 'Mobject | None', target_mobject: 'Mobject | None' = None, path_func: 'Callable | None' = None, path_arc: 'float' = 0, path_arc_axis: 'np.ndarray' = array([0., 0., 1.]), path_arc_centers: 'np.ndarray' = None, replace_mobject_with_target_in_scene: 'bool' = False, **kwargs) -> 'None'` - A Transform transforms a Mobject into a target Mobject.

Parameters
----------
mobject
    The :class:`.Mobject` to be transformed. It will be mutated to become the ``target_mobject``.
target_mobject
    The target of the transformation.
path_func
    A function defining the path that the points of the ``mobject`` are being moved
    along until they match the points of the ``target_mobject``, see :mod:`.utils.paths`.
path_arc
    The arc angle (in radians) that the points of ``mobject`` will follow to reach
    the points of the target if using a circular path arc, see ``path_arc_centers``.
    See also :func:`manim.utils.paths.path_along_arc`.
path_arc_axis
    The axis to rotate along if using a circular path arc, see ``path_arc_centers``.
path_arc_centers
    The center of the circular arcs along which the points of ``mobject`` are
    moved by the transformation.

    If this is set and ``path_func`` is not set, then a ``path_along_circles`` path will be generated
    using the ``path_arc`` parameters and stored in ``path_func``. If ``path_func`` is set, this and the
    other ``path_arc`` fields are set as attributes, but a ``path_func`` is not generated from it.
replace_mobject_with_target_in_scene
    Controls which mobject is replaced when the transformation is complete.

    If set to True, ``mobject`` will be removed from the scene and ``target_mobject`` will
    replace it. Otherwise, ``target_mobject`` is never added and ``mobject`` just takes its shape.

```
</API_INFO>

<CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        circle = Circle(color=RED)
        self.play(Transform(square, circle))
        self.wait()
```
</CODE>

Now, generate code for the following text script adapting initial code based on the provided API information:

<TEXT_SCRIPT>
{reviewed_description}
</TEXT_SCRIPT>

<INITIAL_CODE>
```python
{initial_code}
```
</INITIAL_CODE>

<API_INFO>
```text
{api_info}
```
</API_INFO>

"""

    MANIM_VIDEO_GEN_W_RAT_PROMPT = \
"""You are an expert Manim Community Edition (ManimCE) educator and Python developer.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that I want to create.
2. Analyze the initial code provided in the `<INITIAL_CODE>` block.
3. Identify any incorrect or missing API calls in the initial code based on the provided API information in the `<API_INFO>` block.
4. Use the provided API information in the `<API_INFO>` block to revise the initial code.
5. Ensure the revised code correctly implements the user script using the provided API information in the `<API_INFO>` block.
6. Generate only executable Python code for Manim, wrapped between `<CODE>` and `</CODE>`.
7. DO NOT include any explanations, comments, or instructions on how to run the code. Only include the code.


Example format:

<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<INITIAL_CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(bg_color=RED)
        self.play(Create(square))
        circle = Circle(bg_color=RED)
        self.play(Transform(square, circle))
        self.wait()
``` 
</INITIAL_CODE>

<API_INFO>
```text
- **Circle**: signature `(radius: 'float | None' = None, color: 'ParsableManimColor' = ManimColor('#FC6255'), **kwargs: 'Any') -> 'None'` - A circle.

Parameters
----------
color
    The color of the shape.
kwargs
    Additional arguments to be passed to :class:`Arc`

- **Create**: signature `(mobject: 'VMobject | OpenGLVMobject | OpenGLSurface', lag_ratio: 'float' = 1.0, introducer: 'bool' = True, **kwargs) -> 'None'` - Incrementally show a VMobject.

Parameters
----------
mobject
    The VMobject to animate.

Raises
------
:class:`TypeError`
    If ``mobject`` is not an instance of :class:`~.VMobject`.

- **Square**: signature `(side_length: 'float' = 2.0, **kwargs: 'Any') -> 'None'` - A rectangle with equal side lengths.

Parameters
----------
side_length
    The length of the sides of the square.
kwargs
    Additional arguments to be passed to :class:`Rectangle`.

- **Transform**: signature `(mobject: 'Mobject | None', target_mobject: 'Mobject | None' = None, path_func: 'Callable | None' = None, path_arc: 'float' = 0, path_arc_axis: 'np.ndarray' = array([0., 0., 1.]), path_arc_centers: 'np.ndarray' = None, replace_mobject_with_target_in_scene: 'bool' = False, **kwargs) -> 'None'` - A Transform transforms a Mobject into a target Mobject.

Parameters
----------
mobject
    The :class:`.Mobject` to be transformed. It will be mutated to become the ``target_mobject``.
target_mobject
    The target of the transformation.
path_func
    A function defining the path that the points of the ``mobject`` are being moved
    along until they match the points of the ``target_mobject``, see :mod:`.utils.paths`.
path_arc
    The arc angle (in radians) that the points of ``mobject`` will follow to reach
    the points of the target if using a circular path arc, see ``path_arc_centers``.
    See also :func:`manim.utils.paths.path_along_arc`.
path_arc_axis
    The axis to rotate along if using a circular path arc, see ``path_arc_centers``.
path_arc_centers
    The center of the circular arcs along which the points of ``mobject`` are
    moved by the transformation.

    If this is set and ``path_func`` is not set, then a ``path_along_circles`` path will be generated
    using the ``path_arc`` parameters and stored in ``path_func``. If ``path_func`` is set, this and the
    other ``path_arc`` fields are set as attributes, but a ``path_func`` is not generated from it.
replace_mobject_with_target_in_scene
    Controls which mobject is replaced when the transformation is complete.

    If set to True, ``mobject`` will be removed from the scene and ``target_mobject`` will
    replace it. Otherwise, ``target_mobject`` is never added and ``mobject`` just takes its shape.

```
</API_INFO>

<CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        circle = Circle(color=RED)
        self.play(Transform(square, circle))
        self.wait()
```
</CODE>

Now, generate code for the following text script adapting initial code based on the provided API information:

<TEXT_SCRIPT>
{reviewed_description}
</TEXT_SCRIPT>

<INITIAL_CODE>
```python
{initial_code}
```
</INITIAL_CODE>

<API_INFO>
```text
{api_info}
```
</API_INFO>

"""


    NO_DOCUMENTATION_PROMPT_PART = \
"NO DOCUMENTATION AVAILABLE FOR `{invalid_param}`. Refer RENDER ERRORS (if any) for more details."

    MANIM_QNA_PROMPT = \
"""### Instruction:
{question}

### Context:
This question comes from the Manim {source_type} file `{source_file}`.

### Response:"""

    MANIM_QNA_RESPONSE = \
"""{answer}"""

    MANIM_VID_SFT_GEN_RESPONSE = \
"""<CODE>
```python
{code}
```
</CODE>"""


    # OLD PROMPT TEMPLATE, KEEPING FOR BACKWARD COMPATIBILITY
    MANIM_VID_GEN_PROMPT = \
"""You are an expert Manim Community Edition (ManimCE) educator and Python developer.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that I want to create.
2. DO NOT think out loud or provide any explanations.
3. Generate only executable Python code for Manim, wrapped between `<CODE>` and `</CODE>`.
4. Do not include any explanations, comments, or instructions on how to run the code. Only include the code.

Example format:

<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<CODE>
```python
from manim import *
class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        circle = Circle(color=RED)
        self.play(Transform(square, circle))
        self.wait()
``` 
</CODE>

Now, generate code for the following text script:

<TEXT_SCRIPT>
{text_script}
</TEXT_SCRIPT>

{response}"""

    # OLD PROMPT, kept for backward compatibility
    MANIM_VID_GEN_RAG_PROMPT = \
"""You are an expert Manim Community Edition (ManimCE) educator and Python developer.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that I want to create.
2. Analyze the initial code provided in the `<INITIAL_CODE>` block.
3. Identify any incorrect or missing API calls in the initial code based on the provided API information in the `<API_INFO>` block.
4. Use the provided API information in the `<API_INFO>` block to revise the initial code.
5. Ensure the revised code correctly implements the user script using the provided API information in the `<API_INFO>` block.
6. Generate only executable Python code for Manim, wrapped between `<CODE>` and `</CODE>`.
7. DO NOT include any explanations, comments, or instructions on how to run the code. Only include the code.
8. DO NOT think out loud or provide any explanations.


Example format:

<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<INITIAL_CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(bg_color=RED)
        self.play(Create(square))
        circle = Circle(bg_color=RED)
        self.play(Transform(square, circle))
        self.wait()
``` 
</INITIAL_CODE>

<API_INFO>
```text
- **Circle**: signature `(radius: 'float | None' = None, color: 'ParsableManimColor' = ManimColor('#FC6255'), **kwargs: 'Any') -> 'None'` - A circle.

Parameters
----------
color
    The color of the shape.
kwargs
    Additional arguments to be passed to :class:`Arc`

- **Create**: signature `(mobject: 'VMobject | OpenGLVMobject | OpenGLSurface', lag_ratio: 'float' = 1.0, introducer: 'bool' = True, **kwargs) -> 'None'` - Incrementally show a VMobject.

Parameters
----------
mobject
    The VMobject to animate.

Raises
------
:class:`TypeError`
    If ``mobject`` is not an instance of :class:`~.VMobject`.

- **Square**: signature `(side_length: 'float' = 2.0, **kwargs: 'Any') -> 'None'` - A rectangle with equal side lengths.

Parameters
----------
side_length
    The length of the sides of the square.
kwargs
    Additional arguments to be passed to :class:`Rectangle`.

- **Transform**: signature `(mobject: 'Mobject | None', target_mobject: 'Mobject | None' = None, path_func: 'Callable | None' = None, path_arc: 'float' = 0, path_arc_axis: 'np.ndarray' = array([0., 0., 1.]), path_arc_centers: 'np.ndarray' = None, replace_mobject_with_target_in_scene: 'bool' = False, **kwargs) -> 'None'` - A Transform transforms a Mobject into a target Mobject.

Parameters
----------
mobject
    The :class:`.Mobject` to be transformed. It will be mutated to become the ``target_mobject``.
target_mobject
    The target of the transformation.
path_func
    A function defining the path that the points of the ``mobject`` are being moved
    along until they match the points of the ``target_mobject``, see :mod:`.utils.paths`.
path_arc
    The arc angle (in radians) that the points of ``mobject`` will follow to reach
    the points of the target if using a circular path arc, see ``path_arc_centers``.
    See also :func:`manim.utils.paths.path_along_arc`.
path_arc_axis
    The axis to rotate along if using a circular path arc, see ``path_arc_centers``.
path_arc_centers
    The center of the circular arcs along which the points of ``mobject`` are
    moved by the transformation.

    If this is set and ``path_func`` is not set, then a ``path_along_circles`` path will be generated
    using the ``path_arc`` parameters and stored in ``path_func``. If ``path_func`` is set, this and the
    other ``path_arc`` fields are set as attributes, but a ``path_func`` is not generated from it.
replace_mobject_with_target_in_scene
    Controls which mobject is replaced when the transformation is complete.

    If set to True, ``mobject`` will be removed from the scene and ``target_mobject`` will
    replace it. Otherwise, ``target_mobject`` is never added and ``mobject`` just takes its shape.

```
</API_INFO>

<CODE>
```python
from manim import *

class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        circle = Circle(color=RED)
        self.play(Transform(square, circle))
        self.wait()
```
</CODE>

Now, generate code for the following text script adapting initial code based on the provided API information:

<TEXT_SCRIPT>
{text_script}
</TEXT_SCRIPT>

<INITIAL_CODE>
```python
{initial_code}
```
</INITIAL_CODE>

<API_INFO>
```text
{api_info}
```
</API_INFO>

{response}"""

