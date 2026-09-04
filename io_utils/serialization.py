# -*- coding: utf-8 -*-
"""
io_utils/serialization.py

Helpers for converting numpy/dataclass objects into JSON-safe values.
将数据转换为数据交换友好的json格式数据

"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serializable value."""
    if is_dataclass(obj):
        obj_dict = asdict(obj)
        # 这不是直接return，而是递归调用自己，这一步转换为dict了，然后转入下一步
        return to_jsonable(obj_dict)
    # isinstance（对象，类型）判断对象的类型
    if isinstance(obj, dict):
        json_dict = {}
        for key, value in obj.items():
            json_key = str(key)
            json_value = to_jsonable(value)
            json_dict[json_key] = json_value
        return json_dict
# tuple 存放任意类型元素，但自身不可变（不能 append/pop/insert）。
    if isinstance(obj, (list, tuple)):
        json_list = []
        for value in obj:
            json_value = to_jsonable(value)
            json_list.append(json_value)
        return json_list

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        # 数组元素的总个数
        if obj.size <= 100:
            array_list = obj.tolist()
            # 递归清洗再检查一遍是否转为json
            return to_jsonable(array_list)
    # 省略了else，因为前面每个if分支都return了，如果都不满足就继续往下走
        shape = []
        # 这里的item是shape的每个元素，shape是一个tuple，表示数组的维度大小，比如(3,4)表示3行4列；int(item)是把这个维度大小转换为整数类型，最终得到一个列表形式的shape，比如[3,4]。
        for item in obj.shape:
            shape.append(int(item))

        return {
            "type": "ndarray",
            "shape": shape,
            "dtype": str(obj.dtype),
        }
# np.generic 是 numpy 中所有标量类型的父类，包含了 int64、float64 等具体类型。isinstance(obj, np.generic) 用于检查 obj 是否是 numpy 标量类型的实例。
    if isinstance(obj, np.generic):
        # 通过调用 obj.item() 方法将 numpy 标量类型转换为对应的 Python 原生标量类型（如 int、float 等）。这样做是为了确保返回的值是 JSON 可序列化的。
        scalar_value = obj.item()
        return to_jsonable(scalar_value)

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    return obj
