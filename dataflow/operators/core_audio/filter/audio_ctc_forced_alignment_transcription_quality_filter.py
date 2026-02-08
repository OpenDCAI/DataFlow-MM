import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.operators.core_audio import CTCForcedAlignmentSampleEvaluator

from typing import Union, List, Dict, Any

@OPERATOR_REGISTRY.register()
class CTCForcedAlignmentFilter(OperatorABC):
    def __init__(
        self, 
        model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner", 
        device: Union[str, List[str]] = "cuda", 
        num_workers: int = 1,
        sampling_rate: int = 16000,
        language: str = "en",
        micro_batch_size: int = 16,
        chinese_to_pinyin: bool = False,
        threshold: float = 0.8,
        threshold_mode: str = "min",
        romanize: bool = True,
    ):
        self.logger = get_logger(__name__)
        self.evaluator = CTCForcedAlignmentSampleEvaluator(
            model_path=model_path, 
            device=device,
            num_workers=num_workers,
            sampling_rate=sampling_rate,
            language=language,
            micro_batch_size=micro_batch_size,
            chinese_to_pinyin=chinese_to_pinyin,
            romanize=romanize,
        )
        self.sampling_rate = sampling_rate
        self.language = language
        self.micro_batch_size = micro_batch_size
        self.chinese_to_pinyin = chinese_to_pinyin
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.romanize = romanize
        

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            desc = (
                "CTCForcedAlignmentFilter 算子在内部调用 CTCForcedAlignmentSampleEvaluator 进行 CTC 强制对齐，\n"
                "再基于对齐结果中的 spans 分数，对语音-文本样本进行质量过滤，仅保留对齐质量高于阈值的样本。\n\n"

                "一、__init__ 初始化参数\n"
                "----------------------\n"
                "def __init__(\n"
                "    self,\n"
                "    model_path: str = \"MahmoudAshraf/mms-300m-1130-forced-aligner\",\n"
                "    device: Union[str, List[str]] = \"cuda\",\n"
                "    num_workers: int = 1,\n"
                "    sampling_rate: int = 16000,\n"
                "    language: str = \"en\",\n"
                "    micro_batch_size: int = 16,\n"
                "    chinese_to_pinyin: bool = False,\n"
                "    threshold: float = 0.8,\n"
                "    threshold_mode: str = \"min\",\n"
                "    romanize: bool = True,\n"
                "):\n\n"
                "- model_path: str\n"
                "  强制对齐模型的路径（Hugging Face Hub 或本地路径），会直接透传给内部的\n"
                "  CTCForcedAlignmentSampleEvaluator。\n\n"
                "- device: Union[str, List[str]]\n"
                "  推理设备配置：\n"
                "  * 串行模式：可以是 \"cpu\"、\"cuda\" 等单个设备字符串；\n"
                "  * 并行模式：可以是设备列表 [\"cuda:0\", \"cuda:1\"] 等，子进程将轮流使用这些设备。\n\n"
                "- num_workers: int\n"
                "  进程数：\n"
                "  * num_workers <= 1：串行运行，仅在主进程加载并运行模型；\n"
                "  * num_workers  > 1：多进程运行，每个子进程各自持有一份对齐模型实例。\n\n"
                "- sampling_rate: int\n"
                "  音频加载与重采样使用的采样率（Hz），会传给内部 evaluator。\n\n"
                "- language: str\n"
                "  文本语言代码（ISO 639-1/639-3），用于正则化和罗马化，例如 'en'、'zh'、'ara'。\n\n"
                "- micro_batch_size: int\n"
                "  生成 CTC 发射概率时的微批大小（传给 generate_emissions），也由内部 evaluator 使用。\n\n"
                "- chinese_to_pinyin: bool\n"
                "  若为 True，在对齐前先使用 pypinyin 将中文文本转为拼音（空格分词），用于 evaluator 对齐。\n\n"
                "- threshold: float\n"
                "  过滤阈值。根据 threshold_mode 聚合得到的样本整体分数 val 若小于该值，则样本被丢弃。\n\n"
                "- threshold_mode: str\n"
                "  样本整体得分的聚合方式，仅支持：\n"
                "  * 'min' : 使用该样本 spans 列表中所有 span_dict['score'] 的最小值；\n"
                "  * 'mean': 使用所有 span_dict['score'] 的均值。\n\n"
                "- romanize: bool\n"
                "  若为 True，内部 evaluator 会使用 uroman 将文本罗马化（对非拉丁文字特别有用）。\n\n"
                "初始化行为：\n"
                "- 创建 logger，并实例化一个 CTCForcedAlignmentSampleEvaluator，所有对齐相关配置\n"
                "  （model_path、device、num_workers、sampling_rate、language 等）都在此时固定。\n"
                "- 将阈值和阈值模式（threshold、threshold_mode）等过滤相关配置保存在当前算子实例中。\n\n"

                "二、run 接口参数\n"
                "----------------\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = \"audio\",\n"
                "    input_conversation_key: str = \"conversation\",\n"
                "    output_answer_key: str = \"forced_alignment_results\",\n"
                "):\n\n"
                "- storage: DataFlowStorage\n"
                "  数据流存储对象。\n\n"
                "- input_audio_key: str = \"audio\"\n"
                "  DataFrame 中音频路径所在列名；每行可为字符串或仅含单个路径的列表。\n\n"
                "- input_conversation_key: str = \"conversation\"\n"
                "  DataFrame 中文本/对话所在列名，具体解析规则由内部 evaluator 负责：\n"
                "  * 若为字符串，视为完整转录文本；\n"
                "  * 若为列表且首元素为包含 'value' 键的 dict，则使用 conversation[0]['value']。\n\n"
                "- output_answer_key: str = \"forced_alignment_results\"\n"
                "  对齐结果在当前 DataFrame 中所写入的列名，每行对应 evaluator 的一条对齐记录。\n\n"

                "三、运行行为\n"
                "------------\n"
                "1）断言 threshold_mode 合法：必须为 'mean' 或 'min'，否则抛出异常。\n\n"
                "2）从 storage 中读取上游写入的 DataFrame。\n\n"
                "3）调用内部 evaluator.eval：\n"
                "   - 使用在 __init__ 中配置的 model_path、device、num_workers、sampling_rate、language 等，\n"
                "     对每条 (audio, text) 样本执行 CTC 强制对齐；\n"
                "   - 返回 records 列表，其中每个元素是一个 dict：\n"
                "     { 'spans': [...], 'word_timestamps': [...], 'error': str | None }。\n\n"
                "4）将 records 写入 DataFrame：\n"
                "   - 创建 dataframe 的拷贝；\n"
                "   - 新增一列 output_answer_key，将每条 record 填入对应行。\n\n"
                "5）逐行过滤样本：\n"
                "   - 初始化一个空列表 output_dataframe，用于收集保留的样本。\n"
                "   - 对于 DataFrame 中的每一行：\n"
                "     * 若 row[output_answer_key]['error'] 非 None，则跳过该样本（视为对齐失败或异常）。\n"
                "     * 否则，从 row[output_answer_key]['spans'] 中取出所有 span_dict['score']：\n"
                "       - 若 threshold_mode == 'min' ，val = 所有 score 的最小值；\n"
                "       - 若 threshold_mode == 'mean'，val = 所有 score 的平均值。\n"
                "     * 当 val >= self.threshold 时，将该行转换为 dict 并 append 到 output_dataframe 列表中。\n\n"
                "6）写回或仅日志：\n"
                "   - 若 output_dataframe 非空：将其构造为新的 DataFrame，并通过 storage.write 覆盖写回；\n"
                "   - 若 output_dataframe 为空：只打印日志 \"All data has been filtered out!\"，不再写回 DataFrame。\n\n"
                "7）资源释放（可选）：\n"
                "   - 在 close() 中，如内部 evaluator 处于并行模式（is_parallel=True），则关闭其进程池。\n\n"

                "四、输出结果\n"
                "run 执行结束后：\n"
                "- 若存在通过过滤的样本：\n"
                "  * storage 中的 DataFrame 将被覆盖为过滤后的 DataFrame；\n"
                "  * 行数等于通过阈值筛选的样本数；\n"
                "  * 该 DataFrame 中包含 output_answer_key 列，其每行的值为 evaluator 输出的对齐记录：\n"
                "    - 'spans': 帧级对齐片段列表，每个元素包含 label/start/end/score；\n"
                "    - 'word_timestamps': 若 retain_word_level_alignment=True，则包含词/片段级时间戳；\n"
                "    - 'error': 对齐是否异常的错误信息（正常样本为 None）。\n\n"
                "- 若所有样本都被过滤：\n"
                "  * 不会对 storage 中的 DataFrame 进行覆盖写入；\n"
                "  * 仅在日志中打印“All data has been filtered out!” 以示提示。\n\n"
                "综上，本算子适用于在大规模语音-文本对中自动筛选对齐质量较高的样本，可用作\n"
                "数据清洗、训练集构建或评估阶段的质量控制环节。\n"
            )
        else:
            desc = (
                "CTCForcedAlignmentFilter wraps CTCForcedAlignmentSampleEvaluator to first perform\n"
                "CTC-based forced alignment on audio-text pairs, and then filters samples based on\n"
                "the alignment quality derived from span scores. Only samples whose aggregated\n"
                "score exceeds a configurable threshold are kept, and the filtered dataframe is\n"
                "written back to storage.\n\n"

                "1. __init__ parameters\n"
                "def __init__(\n"
                "    self,\n"
                "    model_path: str = \"MahmoudAshraf/mms-300m-1130-forced-aligner\",\n"
                "    device: Union[str, List[str]] = \"cuda\",\n"
                "    num_workers: int = 1,\n"
                "    sampling_rate: int = 16000,\n"
                "    language: str = \"en\",\n"
                "    micro_batch_size: int = 16,\n"
                "    chinese_to_pinyin: bool = False,\n"
                "    threshold: float = 0.8,\n"
                "    threshold_mode: str = \"min\",\n"
                "    romanize: bool = True,\n"
                "):\n\n"
                "- model_path: str\n"
                "  Path to the CTC forced alignment model (Hugging Face hub or local path).\n"
                "  This is passed directly to the internal CTCForcedAlignmentSampleEvaluator.\n\n"
                "- device: Union[str, List[str]]\n"
                "  Inference device configuration:\n"
                "  * Serial mode: a single string such as \"cpu\" or \"cuda\";\n"
                "  * Parallel mode: a list of devices, e.g. [\"cuda:0\", \"cuda:1\"], and workers\n"
                "    will be distributed over these devices.\n\n"
                "- num_workers: int\n"
                "  Number of worker processes:\n"
                "  * num_workers <= 1: run in serial mode in the main process only;\n"
                "  * num_workers  > 1: multi-process mode, each worker holds its own model instance.\n\n"
                "- sampling_rate: int\n"
                "  Sampling rate (Hz) used for audio loading/resampling, passed through to the evaluator.\n\n"
                "- language: str\n"
                "  Text language code (ISO 639-1/639-3), used for normalization and romanization,\n"
                "  e.g. 'en', 'zh', 'ara'.\n\n"
                "- micro_batch_size: int\n"
                "  Micro batch size used when generating emissions in generate_emissions.\n\n"
                "- chinese_to_pinyin: bool\n"
                "  If True, Chinese text is converted into pinyin before alignment (space-separated).\n\n"
                "- threshold: float\n"
                "  Filtering threshold. Given an aggregated sample score val, any sample with\n"
                "  val < threshold is discarded.\n\n"
                "- threshold_mode: str\n"
                "  Strategy for aggregating per-span scores into a single sample score. Must be one of:\n"
                "  * 'min' : use the minimum score across all spans of a sample;\n"
                "  * 'mean': use the average score of all spans.\n\n"
                "- romanize: bool\n"
                "  If True, the evaluator romanizes text using uroman based on the given language code,\n"
                "  which is often helpful for non-Latin scripts.\n\n"
                "Initialization behavior:\n"
                "- A logger is created, and a CTCForcedAlignmentSampleEvaluator instance is initialized\n"
                "  with all alignment-related settings (model_path, device, num_workers, sampling_rate,\n"
                "  language, micro_batch_size, etc.).\n"
                "- Filtering-related parameters (threshold, threshold_mode, etc.) are stored in the\n"
                "  filter operator instance for use during run().\n\n"

                "2. run interface\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = \"audio\",\n"
                "    input_conversation_key: str = \"conversation\",\n"
                "    output_answer_key: str = \"forced_alignment_results\",\n"
                "):\n\n"
                "- storage: DataFlowStorage\n"
                "  DataFlow storage object. \n\n"
                "- input_audio_key: str = \"audio\"\n"
                "  Column name that contains audio paths. Each row may be a string path or a single-element\n"
                "  list containing the path.\n\n"
                "- input_conversation_key: str = \"conversation\"\n"
                "  Column name that contains text/transcripts. The exact parsing logic is delegated to\n"
                "  the evaluator:\n"
                "  * If the cell is a string, it is treated as the full transcript;\n"
                "  * If the cell is a list whose first element is a dict with key 'value', then\n"
                "    conversation[0]['value'] is used.\n\n"
                "- output_answer_key: str = \"forced_alignment_results\"\n"
                "  Name of the column used to store per-row alignment records returned by the evaluator.\n\n"

                "3. Runtime behavior\n"
                "1) Assert that threshold_mode is valid: it must be either 'mean' or 'min'; otherwise,\n"
                "   an assertion error is raised.\n\n"
                "2) Read the input DataFrame from storage.\n\n"
                "3) Call self.evaluator.eval:\n"
                "   - Uses the configuration set in __init__ (model_path, device, num_workers,\n"
                "     sampling_rate, language, etc.) to perform CTC forced alignment for each\n"
                "     (audio, text) pair;\n"
                "   - Returns a list of records, where each record is a dict of the form:\n"
                "     { 'spans': [...], 'word_timestamps': [...], 'error': str | None }.\n\n"
                "4) Attach alignment results to the DataFrame:\n"
                "   - Make a copy of the original dataframe;\n"
                "   - Add a new column named output_answer_key, storing each record at its corresponding row.\n\n"
                "5) Filter rows based on alignment quality:\n"
                "   - Initialize an empty list output_dataframe for the kept samples.\n"
                "   - For each row in the dataframe copy:\n"
                "     * If row[output_answer_key]['error'] is not None, skip this sample (alignment failed\n"
                "       or encountered an exception).\n"
                "     * Otherwise, collect all span_dict['score'] from row[output_answer_key]['spans'].\n"
                "       - If threshold_mode == 'min' : val = min(all scores);\n"
                "       - If threshold_mode == 'mean': val = average(all scores).\n"
                "     * If val >= self.threshold, convert the row to a dict and append it to output_dataframe.\n\n"
                "6) Write back or only log:\n"
                "   - If output_dataframe is non-empty: convert it into a new DataFrame and write it back\n"
                "     to storage via storage.write, overwriting the existing 'dataframe'.\n"
                "   - If output_dataframe is empty: log \"All data has been filtered out!\" and do not\n"
                "     overwrite the dataframe in storage.\n\n"
                "7) Resource cleanup (optional):\n"
                "   - In close(), if the internal evaluator is running in parallel mode (is_parallel=True),\n"
                "     its underlying process pool is closed and joined.\n\n"

                "4. Output\n"
                "After run completes:\n"
                "- If at least one sample passes the filter:\n"
                "  * The DataFrame under storage is overwritten with the filtered\n"
                "    DataFrame containing only the kept samples;\n"
                "  * The filtered DataFrame includes the column output_answer_key, where each row contains\n"
                "    the full alignment record from the evaluator:\n"
                "    - 'spans'           : frame-level alignment spans with label/start/end/score;\n"
                "    - 'word_timestamps' : optional word/segment-level timestamps if enabled;\n"
                "    - 'error'           : None for successfully aligned samples.\n\n"
                "- If all samples are filtered out:\n"
                "  * The stored DataFrame is left unchanged;\n"
                "  * Only a log message \"All data has been filtered out!\" is emitted.\n\n"
                "In summary, this operator is suitable for automatically cleaning large-scale speech–text\n"
                "datasets by retaining only those pairs whose forced alignment confidence exceeds a\n"
                "specified threshold, making it useful for training data preparation or quality control.\n"
            )
        return desc

    def run(self, 
            storage: DataFlowStorage,
            input_audio_key: str = "audio",
            input_conversation_key: str = "conversation",
            output_answer_key: str = "forced_alignment_results",
        ):
        assert self.threshold_mode in ['mean', 'min'], f"threshold_mode must be 'mean' or 'min', got '{self.threshold_mode}'"

        dataframe = storage.read()
        records = self.evaluator.eval(
            dataframe=dataframe,
            input_audio_key=input_audio_key,
            input_conversation_key=input_conversation_key,
        )

        dataframe = dataframe.copy()
        dataframe.loc[:, output_answer_key] = records
        output_dataframe = []

        for idx, row in dataframe.iterrows():
            if row[output_answer_key]['error'] is not None:
                continue
            
            alignment_list = row[output_answer_key]['alignment']

            if self.threshold_mode == 'min':
                val = min(alignment_dict['score'] for alignment_dict in alignment_list)
            else:
                val = sum(alignment_dict['score'] for alignment_dict in alignment_list) / len(alignment_list)
                    
            if val >= self.threshold:
                output_dataframe.append(row.to_dict())

        if output_dataframe:
            output_dataframe = pd.DataFrame(output_dataframe)
            storage.write(output_dataframe)
        else:
            self.logger.info(f"All data has been filtered out!")

    def close(self):
        if self.evaluator.is_parallel:
            self.evaluator.pool.close()
            self.evaluator.pool.join()