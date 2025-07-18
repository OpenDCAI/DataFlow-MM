import inspect
import types
from dataflow import get_logger
from dataflow.core import WrapperABC, OperatorABC

def BatchWrapper(operator: OperatorABC, batch_size: int = 32) -> WrapperABC:
    """
    把一个 OperatorABC 实例包装成按 batch_size 批处理的 Wrapper
    同时让 wrapper.run 的函数签名 = operator.run 的签名
    """
    logger = get_logger()
    logger.info(f"Creating BatchWrapper for {operator.__class__.__name__} with batch size {batch_size}")

    # 1) 先取得被包装 operator.run 的签名
    original_run = operator.run
    sig = inspect.signature(original_run)

    # 2) 动态定义一个 run 方法
    def run(self, *args, **kwargs):
        """
        这个文档和签名都会被下面的 __wrapped__ / __signature__ 覆盖成 operator.run 的
        """
        # 假设 operator.run 返回一个可迭代的数据项流
        it = self._operator.run(*args, **kwargs)
        batch = []
        for item in it:
            batch.append(item)
            if len(batch) >= self._batch_size:
                # 交给原 operator 处理这一小批
                self._operator.process(batch)
                batch = []
        # 处理最后不满 batch_size 的残余
        if batch:
            self._operator.process(batch)

    # 3) 把 operator.run 本身记录到 __wrapped__（便于追踪）
    run.__wrapped__ = original_run
    # 4) 把签名贴到 run.__signature__ 上（inspect.signature/run-time 补全会拿到它）
    run.__signature__ = sig

    # 5) 动态创建一个新的 Wrapper 子类，并把这个 run 方法挂上去
    BatchWrapperImpl = type(
        f"BatchWrapper_{operator.__class__.__name__}",
        (WrapperABC,),
        {
            "__init__": lambda self: (
                setattr(self, "_operator", operator),
                setattr(self, "_batch_size", batch_size)
            ),
            "run": run
        }
    )

    # 6) 返回实例
    return BatchWrapperImpl()


# 测试示例
if __name__ == "__main__":
    from dataflow.operators.generate.vqa.prompted_vqa import PromptedVQA
    from dataflow.core import VLMServingABC

    vlm_serving = VLMServingABC()      # 你的实际 VLMServing
    op = PromptedVQA(vlm_serving=vlm_serving)

    # 这时候 batch_op.run(...) 在 IDE 里提示的签名和 PromptedVQA.run 一模一样
    batch_op = BatchWrapper(op, batch_size=16)
    batch_op.run(
        storage
        )   # IDE/编辑器会自动补全 run 的参数