from typing import Annotated, Literal, Callable

import numpy as np
import numpy.typing as npt


T_BOARD = Annotated[npt.NDArray[np.bool], Literal["size", "size", 2]]
T_MOVE = tuple[int, int]
T_AI_FN = Callable[[T_BOARD, bool, tuple[int, int]], T_MOVE]
