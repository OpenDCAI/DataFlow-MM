# Operators文件夹下各个文件夹的定义规范：
# Definition Standards for Subfolders under the Operators Directory:

- Generate：数据条目数量不变，每一个条目出现新的Key，对应value是一段新的长文本
- Generate: The number of data entries remains unchanged; each entry gains a new key with a corresponding value that is a new long text.

- Eval: 数据条目数量不变，每一个条目出现新的Key，对应value是分数或者类别
- Eval: The number of data entries remains unchanged; each entry gains a new key with a corresponding value that is a score or category.

- Filter：从多条数据条目过滤成少量数据条目，每一个条目内容不变，或者仅多了一个eval用的数值字段
- Filter: Reduces multiple data entries to a smaller number; each entry's content remains unchanged or gains only an additional numerical field for evaluation.

- Refine：数据条目数量不变，每一个条目对于某一个Key进行修改
- Refine: The number of data entries remains unchanged; each entry has a modification applied to a specific key.