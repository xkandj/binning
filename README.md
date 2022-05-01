# binning

## binning: better for ai engineer to get bin result. Now it supports distance, frequency, enumerate, chi-square, custom bin. it support parallel compute when numbers of features are greater than one hundred or the number of samples are greater than one million, it has better performance.

### | when the distribution type of feature is "CONTINUOUS", we can use distance, frequency, chi-square and custom bin. others, we can use enumerate, chi-square and custom bin

---

### 特征分箱，目前支持等距、等频、枚举、卡方、自定义分箱

- 连续特征
  - 等距，等频，卡方，自定义
- 离散特征
  - 枚举，卡方，自定义

---

## Quick Start

Installation

```python
pip install binning
```

---

## Example

```python
import pandas as pd
from binning import __version__
from binning.binprocessing import BinProcessing

# version, 获取版本
print(__version__)

# custom log function, 自定义日志函数
def log_fun(mess):
    print(f"打印日志信息：{mess}")

# parallel, 是否并行
parallel = False

# distribution: 1 is continuous, 0 is discrete
# distance bin, 等距分箱
data = {"feature": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
"label": [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]}
df = pd.DataFrame(data)
features_dict = {"feature": 1}
kw_params = {"label": "label", "bins": 3}

bp = BinProcessing("DISTANCE_BIN", features_dict, df, parallel, log_fun, **kw_params)
bins_dict = bp.get_bins_dict()
print(bins_dict)

# frequency bin, 等频分箱
data = {"feature": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
"label": [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]}
df = pd.DataFrame(data)
features_dict = {"feature": 1}
kw_params = {"label": "label", "q": 3, "min_samples": 10}

bp = BinProcessing("FREQUENCY_BIN", features_dict, df, parallel, log_fun, \*\*kw_params)
bins_dict = bp.get_bins_dict()
print(bins_dict)

# enumerate bin, 枚举分箱
data = {"feature": ["A", "B", "A", "B", "C", "A", "C", "B", "A", "C"],
"label": [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]}
df = pd.DataFrame(data)
features_dict = {"feature": 0}
kw_params = {"label": "label"}

bp = BinProcessing("ENUMERATE_BIN", features_dict, df, parallel, log_fun, \*\*kw_params)
bins_dict = bp.get_bins_dict()
print(bins_dict)

# chi-square bin, 卡方分箱
data = {"feature": ["A", "B", "A", "B", "C", "A", "C", "B", "A", "C"],
"feature2": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
"label": [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]}
df = pd.DataFrame(data)
features_dict = {"feature": 0, "feature2": 1}
kw_params = {"label": "label"}
# 连续特征参数
kw_params["con_bins"] = 3
kw_params["con_min_samples"] = 5
kw_params["con_threshold"] = 3.8
# 离散特征参数
kw_params["cat_bins"] = 2
kw_params["cat_min_samples"] = 5
kw_params["cat_threshold"] = 3.7

bp = BinProcessing("CHIMERGE_BIN", features_dict, df, parallel, log_fun, \*\*kw_params)
bins_dict = bp.get_bins_dict()
print(bins_dict)

# custome bin, 自定义分箱
data = {"feature": ["A", "B", "A", "B", "C", "A", "C", "B", "A", "C"],
"feature2": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
"label": [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]}
df = pd.DataFrame(data)
features_dict = {"feature": 0, "feature2": 1}
kw_params = {"label": "label"}
# 连续特征参数
kw_params["con_param"] = "2.1,4.1"
kw_params["con_min_samples"] = 5
# 离散特征参数
kw_params["cat_param"] = ["A", "C"]
kw_params["cat_min_samples"] = 3
bp = BinProcessing("CUSTOM_BIN", features_dict, df, parallel, log_fun, \*\*kw_params)
bins_dict = bp.get_bins_dict()
print(bins_dict)
```
