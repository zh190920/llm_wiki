"""
提示词模板管理器 - 基于 YAML 的提示词模板系统
借鉴 WeKnora 的动态提示词设计，支持运行时编辑和占位符替换
"""
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """
    提示词模板管理器

    特性：
    - 从 YAML 文件加载提示词模板
    - 支持 {{placeholder}} 风格的占位符
    - 运行时覆盖任何提示词模板
    - 自定义覆盖持久化到单独文件
    - 提供 get_prompt(), set_prompt(), list_templates(), reset_prompt() 方法
    """

    def __init__(self, templates_path: Optional[str] = None, overrides_path: Optional[str] = None):
        """
        初始化提示词模板管理器

        Args:
            templates_path: YAML 模板文件路径
            overrides_path: 自定义覆盖持久化路径
        """
        if templates_path is None:
            # 自动查找模板文件（多路径搜索）
            search_paths = [
                Path(__file__).parent / "prompt_templates.yaml",       # config/ 目录
                Path(__file__).parent.parent / "agent" / "prompt_templates.yaml",  # agent/ 目录
                Path(__file__).parent.parent / "prompt_templates.yaml",  # 项目根目录
            ]
            for p in search_paths:
                if p.exists():
                    templates_path = str(p)
                    break

        self._templates_path = templates_path
        self._overrides_path = overrides_path or os.path.join(
            os.path.dirname(templates_path) if templates_path else str(Path(__file__).parent),
            "prompt_overrides.yaml"
        )
        self._templates: Dict[str, str] = {}
        self._overrides: Dict[str, str] = {}
        self._placeholder_pattern = re.compile(r'\{\{(\w+)\}\}')

        # 加载模板
        self._load_templates()
        self._load_overrides()

    def _load_templates(self):
        """从 YAML 文件加载模板"""
        if not self._templates_path or not Path(self._templates_path).exists():
            logger.warning(f"提示词模板文件不存在: {self._templates_path}")
            return

        try:
            with open(self._templates_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            for key, value in data.items():
                if isinstance(value, str):
                    self._templates[key] = value
                elif isinstance(value, dict) and "template" in value:
                    self._templates[key] = value["template"]

            logger.info(f"加载 {len(self._templates)} 个提示词模板")
        except Exception as e:
            logger.error(f"加载提示词模板失败: {e}")

    def _load_overrides(self):
        """从文件加载自定义覆盖"""
        if not Path(self._overrides_path).exists():
            return

        try:
            with open(self._overrides_path, "r", encoding="utf-8") as f:
                self._overrides = yaml.safe_load(f) or {}
            logger.info(f"加载 {len(self._overrides)} 个自定义提示词覆盖")
        except Exception as e:
            logger.error(f"加载提示词覆盖失败: {e}")
            self._overrides = {}

    def _save_overrides(self):
        """持久化自定义覆盖到磁盘"""
        try:
            os.makedirs(os.path.dirname(self._overrides_path), exist_ok=True)
            with open(self._overrides_path, "w", encoding="utf-8") as f:
                yaml.dump(self._overrides, f, allow_unicode=True, default_flow_style=False)
        except Exception as e:
            logger.error(f"保存提示词覆盖失败: {e}")

    def get_prompt(self, name: str, **kwargs) -> str:
        """
        获取提示词模板并填充占位符

        Args:
            name: 模板名称
            **kwargs: 占位符键值对

        Returns:
            填充后的提示词文本
        """
        # 优先使用覆盖模板
        template = self._overrides.get(name) or self._templates.get(name)

        if template is None:
            logger.warning(f"提示词模板 '{name}' 不存在")
            return ""

        # 填充占位符
        return self._fill_placeholders(template, **kwargs)

    def set_prompt(self, name: str, template: str):
        """
        设置自定义提示词模板（覆盖默认模板）

        Args:
            name: 模板名称
            template: 模板内容
        """
        self._overrides[name] = template
        self._save_overrides()
        logger.info(f"已设置自定义提示词: {name}")

    def list_templates(self) -> Dict[str, Dict[str, str]]:
        """
        列出所有模板信息

        Returns:
            字典，包含每个模板的名称、是否被覆盖、预览
        """
        result = {}
        all_keys = set(list(self._templates.keys()) + list(self._overrides.keys()))
        for key in all_keys:
            is_overridden = key in self._overrides
            template = self._overrides.get(key) or self._templates.get(key, "")
            preview = template[:100].replace("\n", " ") + "..." if len(template) > 100 else template.replace("\n", " ")
            result[key] = {
                "name": key,
                "overridden": is_overridden,
                "preview": preview,
            }
        return result

    def reset_prompt(self, name: str) -> bool:
        """
        重置模板为默认值（删除覆盖）

        Args:
            name: 模板名称

        Returns:
            是否成功重置
        """
        if name in self._overrides:
            del self._overrides[name]
            self._save_overrides()
            logger.info(f"已重置提示词: {name}")
            return True
        return False

    def _fill_placeholders(self, template: str, **kwargs) -> str:
        """填充模板中的 {{placeholder}} 占位符"""
        def replacer(match):
            key = match.group(1)
            return str(kwargs.get(key, match.group(0)))

        return self._placeholder_pattern.sub(replacer, template)
