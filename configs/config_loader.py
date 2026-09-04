# -*- coding: utf-8 -*-
"""
configs/config_loader.py

YAML loader for the project configuration files.

PyYAML is required because the shipped configurations contain block lists such
as DTW phase specifications. The former minimal fallback parser could silently
handle only a subset of the configuration grammar and is therefore no longer
used by load_config().
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any


def dict_to_namespace(value: Any) -> Any:
    """
    Python 解析 YAML 文件后，得到的是嵌套字典。在读取参数时必须写繁琐的方括号和引号：
    而经过 dict_to_namespace 转换后，代码可以写得像访问对象属性一样极其优雅、直观：
    递归地将字典转换为 SimpleNamespace，以便可以使用点号 (.) 访问键值。
    作用是让你可以像用“点”来访问属性，而不是用“方括号”来查字典。config.well_params.dt

    Args:
        value (Any): 输入的字典、列表或标量值。

    Returns:
        Any: 转换后的 SimpleNamespace、列表或原始标量值。
    """
    if isinstance(value, dict):
        converted_dict = {}
        for key, item in value.items():
            converted_item = dict_to_namespace(item)
            converted_dict[key] = converted_item

        # 这里的 ** 表示解包，把字典里的键值对一个一个拿出来，
        # 作为关键字参数传给 SimpleNamespace。
        return SimpleNamespace(**converted_dict)

    if isinstance(value, list):
        converted_list = []
        for item in value:
            converted_item = dict_to_namespace(item)
            converted_list.append(converted_item)
        return converted_list

    return value


def _parse_scalar(text: str) -> Any:
    """
    解析 YAML 中的标量值，支持字符串、数字、布尔值、None 以及简单的列表。

    Args:
        text (str): 要解析的文本字符串。

    Returns:
        Any: 解析后的 Python 对象（int, float, bool, None, list 或 str）。
    """
    text = text.strip() #去掉字符串两边的空格，数据清洗，保证后续的逻辑是有效内容

    if not text:
        return {} # 如果字符串为空，返回空字典

    # 判断是否为带引号的字符串
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]

    # 判断是否为列表
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        # 取从索引 1 开始，到倒数第 1 个（不包含）为止的内容。
        # text[0] 是左方括号 [。
        # text[-1] 是右方括号 ]。
        # 结果：它把两头的方括号砍掉了。
        if not inner:
            return []

        parsed_list = []
        for part in inner.split(","):
            part_text = part.strip()
            parsed_value = _parse_scalar(part_text)
            parsed_list.append(parsed_value)
        return parsed_list

    # 判断是否为布尔值或None
    lowered = text.lower() # 转为小写，为了方便比较
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none"):
        return None

    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """
    一个简单的 YAML 解析器备份方案，用于在没有安装 PyYAML 时加载配置。
    仅支持基本的缩进结构，不支持 YAML 列表块（以 "- " 开头的行）。

    Args:
        path (str | Path): YAML 配置文件的路径。

    Returns:
        dict[str, Any]: 解析后的字典对象。

    Raises:
        ValueError: 如果检测到不支持的列表块或行格式无效。
    """
    path = Path(path)
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if stripped.startswith("- "):
            raise ValueError("The fallback YAML parser does not support list blocks.")

        if ":" not in stripped:
            raise ValueError(f"Invalid config line: {raw_line}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        parsed = _parse_scalar(value)
        parent[key] = parsed

        if value == "":
            if not isinstance(parsed, dict):
                parsed = {}
                parent[key] = parsed
            stack.append((indent, parsed))

    return root


import warnings


def _validate_config_consistency(cfg: SimpleNamespace, path: Path) -> None:
    """检查实验名称/输出路径与实际算法参数之间是否存在矛盾提示。"""
    tv_cfg = getattr(cfg, "tv_wavelet", None)
    if tv_cfg is None:
        return

    exp_cfg = getattr(cfg, "experiment", None)
    paths_cfg = getattr(cfg, "paths", None)

    exp_name = str(getattr(exp_cfg, "name", "")) if exp_cfg else ""
    result_dir = str(getattr(paths_cfg, "result_dir", "")) if paths_cfg else ""
    combined_str = f"{exp_name} {result_dir}".lower()

    # 检查 nu 值对齐
    if hasattr(tv_cfg, "student_nu"):
        nu_val = float(getattr(tv_cfg, "student_nu"))
        if "nu5" in combined_str and abs(nu_val - 5.0) > 1e-5:
            warnings.warn(
                f"[Config Alignment Warning] {path.name}: 路径/名称包含 'nu5'，但实际 student_nu={nu_val}！",
                UserWarning,
            )
        if "nu10" in combined_str and abs(nu_val - 10.0) > 1e-5:
            warnings.warn(
                f"[Config Alignment Warning] {path.name}: 路径/名称包含 'nu10'，但实际 student_nu={nu_val}！",
                UserWarning,
            )

    # 检查 robust_fallback 对齐
    if hasattr(tv_cfg, "robust_fallback"):
        fallback_val = str(getattr(tv_cfg, "robust_fallback")).lower()
        if "nofallback" in combined_str and fallback_val != "none":
            warnings.warn(
                f"[Config Alignment Warning] {path.name}: 路径/名称包含 'nofallback'，但实际 robust_fallback='{fallback_val}'！",
                UserWarning,
            )


def load_config(path: str | Path) -> SimpleNamespace:
    """
    加载 YAML 配置文件。优先尝试使用 PyYAML，如果不可用则使用内置的简单解析器。
    返回一个支持点号访问的 SimpleNamespace 对象。

    Args:
        path (str | Path): YAML 配置文件路径。

    Returns:
        SimpleNamespace: 包含配置信息的命名空间对象。
    """
    path = Path(path)

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "读取项目 YAML 配置需要 PyYAML。请执行：pip install PyYAML"
        ) from exc

    with open(path, "r", encoding="utf-8-sig") as file:
        cfg_dict = yaml.safe_load(file)

    if not isinstance(cfg_dict, dict):
        raise ValueError(f"配置文件顶层必须是字典：{path}")

    cfg = dict_to_namespace(cfg_dict)
    cfg.config_path = str(path)

    _validate_config_consistency(cfg, path)

    return cfg

