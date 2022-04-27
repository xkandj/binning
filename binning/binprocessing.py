from math import ceil
from typing import Any, Callable, Dict, List

import pandas as pd
from fmpc.utils.EnvUtil import EnvUtil
from fmpc.utils.LogUtils import get_fmpc_logger
from wares.common.binning.chimerge_bin import ChimergeBin
from wares.common.binning.custom_bin import CustomBin
from wares.common.binning.distance_bin import DistanceBin
from wares.common.binning.enumerate_bin import EnumerateBin
from wares.common.binning.enums import BinType
from wares.common.binning.frequency_bin import FrequencyBin
from wares.common.binning.tool import Tool
from wares.common.binning.wrap_utils import exec_except, exec_time

logger = get_fmpc_logger(__name__)


class BinProcessing:
    """分箱逻辑入口

    Args:
        bin_type (str): 分箱类型
        features_dict (Dict[str,int]): 特征信息
                        {"feature":distribution}: {"feature":1, "feature2":0}
                        1: 连续， 0: 离散
        df (pd.DataFrame): 分箱数据源
        log (Callable): 日志函数
    """

    def __init__(self,
                 bin_type: str,
                 features_dict: Dict[str, int],
                 df: pd.DataFrame,
                 parallel: bool,
                 log: Callable,
                 **kwargs):
        self.bin_type = bin_type
        self.features_dict = features_dict.copy()
        self.df = df.copy()
        self.parallel = parallel
        self.log = log

        self.bins_merge = True
        self.params = None
        self.handler = None

        self.tool = Tool()
        self.preprocess(kwargs)

    @exec_time
    @exec_except
    def preprocess(self, kwargs):
        """preprocess"""
        self._check(kwargs)
        self._validate()
        self._assemble_params(kwargs)
        self._assemble_handler()

    def _check(self, kwargs):
        """check"""
        if not self.bin_type:
            raise ValueError(f"分箱方式为空，请检查")

        if not self.features_dict:
            raise ValueError(f"特征信息为空，请检查")

        if self.df.empty:
            raise ValueError(f"分箱数据源为空，请检查")

        if not kwargs:
            raise ValueError(f"参数不全，请检查")

    def _validate(self):
        """validate"""
        if not set(self.features_dict.keys()) <= set(self.df.columns):
            raise ValueError(f"特征列{self.features_dict.keys()}不完全在数据源中{self.df.columns}，请检查")

    def _get_parallel_info(self):
        """parallel info"""
        if self.parallel:
            features = list(self.features_dict.keys())
            n_features = len(features)
            n_cpu = ceil(EnvUtil.get_available_cpu_count() / 4)
            n_split, n_parallel = self.tool.get_n_split_n_parallel(n_cpu, n_features)
            features_lst = self.tool.data_slice(features, n_split)

            logger.info(f'==>> 最大并行CPU数:{n_cpu}，实际并行CPU数:{n_parallel}')
            logger.info(f'==>> 数据总特征数:{n_features}，单个并行特征数:{n_split}')
            logger.info(f'==>> 拆分特征:{features_lst}')

            return {"parallel": self.parallel,
                    "n_parallel": n_parallel,
                    "features_lst": features_lst}
        else:
            return {"parallel": self.parallel}

    def _assemble_params(self, kwargs):
        """assemble params"""
        if not kwargs.get("label"):
            kwargs["label"] = "y"
            self.df["y"] = 1
        if kwargs.get("bins_merge"):
            self.bins_merge = kwargs.get("bins_merge")
        kwargs["features_dict"] = self.features_dict
        kwargs["df"] = self.df
        kwargs["parallel_info"] = self._get_parallel_info()
        self.params = kwargs

    def _assemble_handler(self):
        """assembling handler"""
        if self.bin_type == BinType.DISTANCE_BIN.name:
            self.handler = DistanceBin(**self.params)
        elif self.bin_type == BinType.FREQUENCY_BIN.name:
            self.handler = FrequencyBin(**self.params)
        elif self.bin_type == BinType.ENUMERATE_BIN.name:
            self.handler = EnumerateBin(**self.params)
        elif self.bin_type == BinType.CUSTOM_BIN.name:
            self.handler = CustomBin(**self.params)
        elif self.bin_type == BinType.CHIMERGE_BIN.name:
            self.handler = ChimergeBin(**self.params)
        else:
            raise TypeError(f"暂不支持此分箱{self.bin_type}")

    def get_bins_dict(self) -> Dict[str, Any]:
        """get bins result(dict)

        Returns:
            Dict[str, Any]: {"feature": List[Any], ...}
        """
        # 生成特征字典
        if self.bin_type == BinType.CHIMERGE_BIN.value:
            if self.params.get("role") == "HOST":
                features_dict = self.handler.bin_process_host(log=self.log)
            else:
                features_dict = self.handler.bin_process(log=self.log)
        else:
            features_dict = self.handler.bin_process(log=self.log)

        # 分箱合并
        if self.bins_merge:
            result_dict = self.handler.bin_process_merging(features_dict)
        else:
            result_dict = features_dict

        # 生成分箱区间
        bins_dict = self.tool.convert_bins_dict(self.features_dict, result_dict)

        return bins_dict

    def get_bins_result(self) -> Dict[str, Any]:
        """get bin result

        Returns:
            Dict[str, Any]: {"feature": {"success": 1, "msg": "", "result": group_result}...}
        """
        if self.bin_type == BinType.CHIMERGE_BIN.value:
            if self.params.get("role") == "HOST":
                bins_result = self.handler.bin_process_host(log=self.log)
            else:
                bins_result = self.handler.bin_process(log=self.log)
        else:
            bins_result = self.handler.bin_process(log=self.log)

        return bins_result

    def bins_merging(self) -> Dict[str, Any]:
        """"bins merging

        Returns:
            Dict[str, Any]: {"feature": {"df": pd.DataFrame, "nan": "NaN"}}
        """
        return self.handler.bin_process_merging()

    def convert_bins_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """convert data to bins lst 

        Args:
            data_dict (Dict[str, Any]): dict {"feature": {}}

        Returns:
            List[Any]: List
        """
        bins_dict = {}
        for feature, data in data_dict.items():
            bins_lst = []
            for interval in data.get["result"].index:
                bins_lst.append(interval.left)
                bins_lst.append(interval.right)
            bins_dict[feature] = list(set(bins_lst))
        return bins_dict

    @staticmethod
    def chimerge_bin_assist_host(log: Callable,
                                 flnodes_tuple: Dict[str, str],
                                 priv: str,
                                 data_transfer: Dict[str, Any]):
        """多方，用于GUEST方协助HOST进行卡方分箱

        Args:
            log (Callable): 日志
            flnodes_tuple (Dict[str, str]): 其他节点信息，{nid: nid}
            priv (str): 私钥
            data_transfer (Dict[str, Any]): 通信集合
                                           {"algo_data_transfer": self.algo_data_transfer,
                                            "listener": self.listener,
                                            "job_id": self.job_id,
                                            "ctx": self.ctx,
                                            "curr_nid": self.curr_nid}
        """
        ChimergeBin.bin_process_assist_host(log, flnodes_tuple, priv, data_transfer)