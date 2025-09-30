import pandas as pd
from typing import Callable, Any, Dict, List, Optional

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class PromptUpdate(OperatorABC):
    """
    Simple version: batch update the value of each message in DataFrame['conversation']
    Usage:
        op = PromptUpdate(update_fn=lambda v: "[PREFIX] " + v)
        op.run(storage)
    """
    def __init__(
        self,
        update_fn: Callable[[str], str],
        conversation_key: str = "conversation",
        only_roles: Optional[List[str]] = None,  # Optional: only modify specified roles (e.g. ["human"])
        system_prompt: str = "You are a helpful assistant.",  # Reserved, not used
    ):
        self.logger = get_logger()
        self.update_fn = update_fn
        self.conversation_key = conversation_key
        self.only_roles = set(only_roles or [])
        self.system_prompt = system_prompt

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "Simple operator to batch update conversation.value" if lang == "zh" else "Simple operator to update conversation.value."
    def run(
        self,
        storage: DataFlowStorage,
        output_answer_key: str = None,  # Compatibility field, still unused
    ):
        if storage is None:
            raise ValueError("storage cannot be None.")

        self.logger.info("Running PromptUpdate...")
        df = storage.read("dataframe")
        self.logger.info(f"Loading dataframe, rows: {len(df)}")

        if self.conversation_key not in df.columns:
            raise KeyError(f"DataFrame does not contain '{self.conversation_key}' column")

        def _safe_call_update_fn(value: str, row: pd.Series) -> str:
            """
            Compatible with two function signatures:
            - update_fn(value)
            - update_fn(value, row)
            """
            try:
                # Try with row as an argument first
                return self.update_fn(value, row)  # type: ignore[misc]
            except TypeError:
                # Fallback to the old signature
                return self.update_fn(value)

        def _update_conversation_row(row: pd.Series) -> Any:
            """
            Update the conversation column based on the entire row.
            You can use row['xxx'] for contextual information here.
            """
            conv_list = row.get(self.conversation_key, None)
            if not isinstance(conv_list, list):
                return conv_list

            new_list = []
            for msg in conv_list:
                if isinstance(msg, dict) and "value" in msg:
                    # Role filter (if only_roles is set)
                    if self.only_roles and msg.get("from") not in self.only_roles:
                        new_list.append(msg)
                        continue

                    new_msg = msg.copy()
                    try:
                        # Let update_fn access both value and row context
                        new_msg["value"] = _safe_call_update_fn(str(msg["value"]), row)
                    except Exception as e:
                        self.logger.exception(
                            f"update_fn failed, rolled back to original value. Error: {e}"
                        )
                        new_msg["value"] = msg["value"]
                    new_list.append(new_msg)
                else:
                    new_list.append(msg)
            return new_list

        # Key: apply to the entire row so _update_conversation_row can access other fields in the row
        df[self.conversation_key] = df.apply(_update_conversation_row, axis=1)

        storage.write(df)
        self.logger.info("PromptUpdate finished.")
        return self.conversation_key
