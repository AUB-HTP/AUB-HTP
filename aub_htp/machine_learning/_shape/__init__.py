from typing import Literal

MethodLiteral = Literal["method1", "method2", "method3"]

from .method1 import estimate_shape_method1 
from .method2 import estimate_shape_method2
from .method3 import estimate_shape_method3