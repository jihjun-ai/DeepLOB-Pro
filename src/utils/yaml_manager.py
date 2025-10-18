"""Object-style YAML helper built on top of :mod:`ruamel.yaml`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Union

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

__all__ = ["YAMLManager"]

_YAML = YAML()
_YAML.preserve_quotes = True
_YAML.width = 4096
_YAML.indent(mapping=2, sequence=2, offset=0)


class _NodeProxy:
    """Proxy object that exposes attribute and mapping style access."""

    __slots__ = ("_manager", "_node", "_path")

    def __init__(self, manager: "YAMLManager", node: Any, path: list[str]):
        object.__setattr__(self, "_manager", manager)
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_path", path)

    @property
    def value(self) -> Any:
        if isinstance(self._node, (CommentedMap, CommentedSeq)):
            return self
        return self._node

    def save(self) -> None:
        self._manager._dump()

    def k(self, key: str) -> "_NodeProxy":
        return self.__getitem__(key)

    def __getattr__(self, name: str) -> Any:
        if name in _NodeProxy.__slots__:
            return object.__getattribute__(self, name)

        node = self._node
        if isinstance(node, CommentedMap) and name in node:
            child = node[name]
            return _NodeProxy(self._manager, child, self._path + [name]).value
        raise AttributeError(name)

    def __getitem__(self, key: Union[str, int]) -> Any:
        node = self._node
        if isinstance(node, CommentedMap):
            child = node[key]
            return _NodeProxy(self._manager, child, self._path + [str(key)]).value
        if isinstance(node, CommentedSeq):
            child = node[key]
            return _NodeProxy(self._manager, child, self._path + [str(key)]).value
        raise TypeError("Current node is not indexable")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _NodeProxy.__slots__:
            object.__setattr__(self, name, value)
            return

        node = self._node
        if isinstance(node, CommentedMap):
            node[name] = value
        else:
            raise TypeError("Only mapping nodes support attribute assignment")

    def __setitem__(self, key: Union[str, int], value: Any) -> None:
        node = self._node
        if isinstance(node, (CommentedMap, CommentedSeq)):
            node[key] = value
        else:
            raise TypeError("Current node is not indexable")

    def __iter__(self) -> Iterator:
        node = self._node
        if isinstance(node, CommentedMap):
            yield from node
        elif isinstance(node, CommentedSeq):
            yield from range(len(node))
        else:
            raise TypeError("Current node is not iterable")

    def as_plain_object(self) -> Any:
        def _strip(n: Any) -> Any:
            if isinstance(n, CommentedMap):
                return {k: _strip(v) for k, v in n.items()}
            if isinstance(n, CommentedSeq):
                return [_strip(v) for v in n]
            return n

        return _strip(self._node)

    def __repr__(self) -> str:
        path = "/".join(self._path) or "/"
        return f"<YAMLNode path={path} type={type(self._node).__name__}>"


class YAMLManager(_NodeProxy):
    """Manager that exposes comment-preserving YAML editing helpers."""

    __slots__ = ("_file",)

    def __init__(self, path: Union[str, Path]):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        data = _YAML.load(file_path.read_text(encoding="utf-8"))
        object.__setattr__(self, "_file", file_path)
        super().__init__(manager=self, node=data, path=[])

    def _dump(self) -> None:
        import time
        import io

        tmp = self._file.with_suffix(self._file.suffix + ".tmp")

        # 🔥 修復：先將 YAML 內容序列化到字串緩衝區，避免檔案句柄關閉問題
        try:
            # 使用 StringIO 作為緩衝區
            buffer = io.StringIO()
            _YAML.dump(self._node, buffer)
            yaml_content = buffer.getvalue()
            buffer.close()

            # 將內容寫入臨時檔案
            tmp.write_text(yaml_content, encoding="utf-8")

        except Exception as e:
            # 如果序列化失敗，記錄詳細錯誤
            if tmp.exists():
                tmp.unlink()  # 清理臨時檔案
            raise RuntimeError(f"YAML 序列化失敗: {e}") from e

        # 🔥 Windows 檔案鎖定問題：如果檔案被 IDE 開啟，重試 3 次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                tmp.replace(self._file)
                return  # 成功
            except PermissionError as e:
                if attempt < max_retries - 1:
                    # 等待一下再試
                    time.sleep(0.5)
                else:
                    # 最後一次失敗，拋出詳細錯誤訊息
                    if tmp.exists():
                        tmp.unlink()  # 清理臨時檔案
                    raise PermissionError(
                        f"無法保存 YAML 檔案 '{self._file}'：檔案可能被其他程序（如 IDE）開啟。"
                        f"請關閉編輯器中的檔案，或等待檔案解鎖後再試。"
                    ) from e

    def save(self) -> None:
        self._dump()

    def save_as(self, path: Union[str, Path]) -> None:
        """Save the current YAML data to a new file path.
        
        Args:
            path: The new file path to save to.
        """
        new_path = Path(path)
        
        # Ensure the directory exists
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to the new path
        with new_path.open("w", encoding="utf-8", newline="") as handle:
            _YAML.dump(self._node, handle)

    def reload(self) -> None:
        data = _YAML.load(self._file.read_text(encoding="utf-8"))
        object.__setattr__(self, "_node", data)

    def as_dict(self) -> dict:
        return super().as_plain_object()

    def __getitem__(self, key: Union[str, int]) -> Any:  # type: ignore[override]
        return super().__getitem__(key)

    def __setitem__(self, key: Union[str, int], value: Any) -> None:  # type: ignore[override]
        super().__setitem__(key, value)

    def k(self, key: str) -> _NodeProxy:
        return super().k(key)
