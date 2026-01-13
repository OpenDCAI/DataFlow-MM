from typing import List, Dict, Any

class NamedPlaceholderPromptTemplate:
    """
    通用多占位符 Prompt 模板类。

    模板示例：
        template = (
            "Descriptions:\n"
            "{descriptions}\n\n"
            "Collect all details for {type} in the scene. "
            "Do not include any analysis or your opinions."
        )

    然后 build_prompt 会用：
        template.format(descriptions=..., type=...)
    来生成最终 prompt。
    """

    def __init__(self, template: str, join_list_with: str = "\n"):
        """
        参数：
        - template: 带命名占位符的字符串，例如：
            \"\"\"Descriptions:
            {descriptions}

            Collect all details for {type} in the scene.\"\"\"
        - join_list_with: 如果某个占位符的值是 list/tuple，如何拼接
        """
        self.template = template
        self.join_list_with = join_list_with

    def build_prompt(self, need_fields, **kwargs) -> str:
        """
        参数：
        - need_fields: 本次会使用到的字段名集合（来自算子的 input_keys.keys()）
        - kwargs: 每个字段名 -> 该行的值（来自 DataFrame）

        逻辑：
        - 对于每个字段名 k ∈ need_fields：
            取 kwargs[k] 作为占位符 {k} 的值
        - 如果值是 list/tuple，则用 join_list_with 拼起来
        - 最后 self.template.format(**format_values)
        """
        format_values = {}

        for key in need_fields:
            value = kwargs.get(key, "")
            # 支持 list / tuple
            if isinstance(value, (list, tuple)):
                value = self.join_list_with.join(str(v) for v in value)
            else:
                value = str(value)
            format_values[key] = value

        return self.template.format(**format_values)
