# binning and calculate woe，iv value

## binning: better for ai engineer to get bin result. Now it supports distance, frequency, enumerate, chi-square, custom bin. it support parallel compute when numbers of features are greater than one hundred or the number of samples are greater than one million, it has better performance.

### | when the distribution type of feature is "CONTINUOUS", we can use distance, frequency, chi-square and custom bin. others, we can use enumerate, chi-square and custom bin

### | we can calculate woe and iv value by the bin result

---

# 特征分箱及 WOE&IV 计算

## 特征分箱，目前支持等距、等频、枚举、卡方、自定义分箱

- 分箱方式
  - 连续特征
    - 等距，等频，卡方，自定义
  - 离散特征
    - 枚举，卡方，自定义
- 合并分箱
  - 根据不同的特征类型进行不同合并分箱，同时支持不进行合并分箱
  - 合并原则
    - 连续特征合并分箱根据最小样本量
    - 离散特征合并分箱根据最小样本量，正负样本比率
  - 等频和枚举分箱方式不进行合并分箱

---

## Quick Start

Installation

```python
pip install binning
```

---

### Example: bins dict

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
features = {"feature": 1}
kw_params = {"label": "label", "bins": 3}

bp = BinProcessing("DISTANCE_BIN", features, df, parallel, log_fun, **kw_params)
bins_dict = bp.get_bins_dict()
print(bins_dict)

# frequency bin, 等频分箱
data = {"feature": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
"label": [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]}
df = pd.DataFrame(data)
features = {"feature": 1}
kw_params = {"label": "label", "q": 3, "min_samples": 10}

bp = BinProcessing("FREQUENCY_BIN", features, df, parallel, log_fun, **kw_params)
bins_dict = bp.get_bins_dict()
print(bins_dict)

# enumerate bin, 枚举分箱
data = {"feature": ["A", "B", "A", "B", "C", "A", "C", "B", "A", "C"],
"label": [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]}
df = pd.DataFrame(data)
features = {"feature": 0}
kw_params = {"label": "label"}

bp = BinProcessing("ENUMERATE_BIN", features, df, parallel, log_fun, **kw_params)
bins_dict = bp.get_bins_dict()
print(bins_dict)

# chi-square bin, 卡方分箱
data = {"feature": ["A", "B", "A", "B", "C", "A", "C", "B", "A", "C"],
"feature2": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
"label": [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]}
df = pd.DataFrame(data)
features = {"feature": 0, "feature2": 1}
kw_params = {"label": "label"}
# 连续特征参数
kw_params["con_bins"] = 3
kw_params["con_min_samples"] = 5
kw_params["con_threshold"] = 3.8
# 离散特征参数
kw_params["cat_bins"] = 2
kw_params["cat_min_samples"] = 5
kw_params["cat_threshold"] = 3.7

bp = BinProcessing("CHIMERGE_BIN", features, df, parallel, log_fun, **kw_params)
bins_dict = bp.get_bins_dict()
print(bins_dict)

# custome bin, 自定义分箱
data = {"feature": ["A", "B", "A", "B", "C", "A", "C", "B", "A", "C"],
"feature2": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
"label": [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]}
df = pd.DataFrame(data)
features = {"feature": 0, "feature2": 1}
kw_params = {"label": "label"}
# 连续特征参数
kw_params["con_param"] = "2.1,4.1"
kw_params["con_min_samples"] = 5
# 离散特征参数
kw_params["cat_param"] = ["A", "C"]
kw_params["cat_min_samples"] = 3
bp = BinProcessing("CUSTOM_BIN", features, df, parallel, log_fun, **kw_params)
bins_dict = bp.get_bins_dict()
print(bins_dict)
```

---

### Example: feature dict and woe, iv value

```python
import pandas as pd
from binning.binprocessing import BinProcessing
from binning.calprocessing import CalProcessing

def get_features_dict(df: pd.DataFrame,
                      features: Dict[str, int]) -> Dict[str, Any]:
    """get features dict

      Args:
          df (pd.DataFrame): 数据源
          features (Dict[str,int]): 特征集

      Returns:
          Dict[str,Any]: 特征字典
    """
    ret_features_dict = {}

    con_features = {k: v for k, v in features.items() if v == 1}
    cat_features = {k: v for k, v in features.items() if v == 0}
    if self.con_cut == BinType.DISTINCE_BIN.value and con_features:
        kw_params = {"label": self.label_b, "bins": self.con_group,
                      "min_samples": self.con_cut_param["minSampleNum"]}
        bp = BinProcessing("DISTANCE_BIN", con_features, df, parallel, self._log, **kw_params)
        ret_dict = bp.get_features_dict()
        ret_features_dict.update(ret_dict)

    if self.con_cut == BinType.FREQUENCY_BIN.value and con_features:
        kw_params = {"label": self.label_b, "q": self.con_group}
        bp = BinProcessing("FREQUENCY_BIN", con_features, df, parallel, self._log, **kw_params)
        ret_dict = bp.get_features_dict()
        ret_features_dict.update(ret_dict)

    if self.cat_cut == BinType.DISCRE_ENUM_BIN.value and cat_features:
        kw_params = {"label": self.label_b}
        bp = BinProcessing("ENUMERATE_BIN", cat_features, df, parallel, self._log, **kw_params)
        ret_dict = bp.get_features_dict()
        ret_features_dict.update(ret_dict)

    if self.is_con_feature_chi2 or self.is_cat_feature_chi2:
        chi_features = self.get_chi2_features_dict(features, con_features, cat_features)
        kw_params = {"label": self.label_b}
        if self.is_con_feature_chi2 and self.con_cut_param:
            kw_params["con_bins"] = self.con_cut_param["maxBinNum"]
            kw_params["con_min_samples"] = self.con_cut_param["minSampleNum"]
            kw_params["con_threshold"] = self.con_cut_param["threshold"]
        if self.is_cat_feature_chi2 and self.cat_cut_param:
            kw_params["cat_bins"] = self.cat_cut_param["maxBinNum"]
            kw_params["cat_min_samples"] = self.cat_cut_param["minSampleNum"]
            kw_params["cat_threshold"] = self.cat_cut_param["threshold"]
        bp = BinProcessing("CHIMERGE_BIN", chi_features, df, parallel, self._log, **kw_params)
        ret_dict = bp.get_features_dict()
        ret_features_dict.update(ret_dict)

    if self.single_cut == BinType.CUSTOM_BIN.value:
      kw_params = {"label": self.label_b}
      kw_params["con_param"] = self.single_bin_param.get('userDefineParam')
      kw_params["con_min_samples"] = self.min_sample_num
      kw_params["cat_param"] = self.single_bin_param.get('discreteDefineParam')
      kw_params["cat_min_samples"] = self.min_sample_num
      bp = BinProcessing("CUSTOM_BIN", features, df, parallel, self._log, **kw_params)
      ret_dict = bp.get_features_dict()
      ret_features_dict.update(ret_dict)

def get_woe_iv(con_bin_type: str,
              cat_bin_type: str,
              features: Dict[str, int],
              features_dict: Dict[str, Any],
              bin_params: Dict[str, Any],
              min_samples: Dict[str, int],
              log: Callable) -> Tuple[Dict[str, Any], pd.Series]:
    """get woe, iv value

    Args:
        con_bin_type (str): 连续特征分箱类型
        cat_bin_type (str): 离散特征分箱类型
        features (Dict[str,int]): 特征字典
        features_dict (Dict[str,Any]): 特征信息
        bin_params (Dict[str,Any]): 分箱参数
        min_samples (Dict[str,int]): 最小样本量
        log (Callable): 日志函数

    Returns:
        Tuple[Dict[str,Any], pd.Series]: woe, iv
    """
    kw_params = {"con_bin_type": con_bin_type,
                "cat_bin_type": cat_bin_type,
                "features": features,
                "features_dict": features_dict,
                "bin_params": bin_params}
    if min_samples.get("con_min_samples"):
        kw_params["con_min_samples"] = min_samples["con_min_samples"]
    if min_samples.get("cat_min_samples"):
        kw_params["cat_min_samples"] = min_samples["cat_min_samples"]

    cp = CalProcessing("WOEIV", log, **kw_params)
    woe, iv = cp.get_cal_val()
    return (woe, iv)
```
