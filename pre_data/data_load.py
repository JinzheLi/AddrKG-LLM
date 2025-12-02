import re
from typing import List, Iterable

from torch.utils.data import Dataset
from typing import Callable, Optional


import pandas as pd
from typing import List, Optional

class DataCleaner:
    """
    文本清洗工具类：支持去空格、去标点、去括号内容、去尾部数字。
    """

    # 预编译正则提高性能
    _RE_BRACKETS = re.compile(r'[（(].*?[）)]')
    _RE_NUM_SUFFIX = re.compile(r'[\d]+(号|门|室|号室|楼|号楼|单元)?$')
    _RE_TRAILING_NUM = re.compile(r'[\d]+$')
    _RE_SYMBOLS = re.compile(r'[。，,.]')  # 可扩展其他符号

    def __init__(self):
        self.sentences: List[str] = []

    def load(
        self,
        data: Iterable[str],
        remove_empty: bool = True,
        remove_symbols: bool = True,
        remove_brackets: bool = True,
        remove_numbers: bool = True
    ) -> List[str]:
        """
        清洗文本数据。

        :param data: 可迭代字符串
        :param remove_empty: 是否移除空格
        :param remove_symbols: 是否移除标点
        :param remove_brackets: 是否移除括号内容
        :param remove_numbers: 是否移除末尾数字及号楼信息
        :return: 清洗后的字符串列表
        """
        cleaned = []
        for s in data:
            if remove_empty:
                s = self._remove_spaces(s)
            if remove_symbols:
                s = self._remove_symbols(s)
            if remove_brackets:
                s = self._remove_brackets(s)
            if remove_numbers:
                s = self._remove_numbers(s)
            cleaned.append(s)
        self.sentences = cleaned
        return cleaned

    def _remove_spaces(self, text: str) -> str:
        return text.replace(" ", "")

    def _remove_symbols(self, text: str) -> str:
        return self._RE_SYMBOLS.sub("", text)

    def _remove_brackets(self, text: str) -> str:
        return self._RE_BRACKETS.sub("", text)

    def _remove_numbers(self, text: str) -> str:
        prev = None
        while text != prev:
            prev = text
            text = self._RE_NUM_SUFFIX.sub("", text)
            text = self._RE_TRAILING_NUM.sub("", text)
        return text



class AddressDataLoader:
    """
    通用地址数据加载器：支持CSV文件加载，并可选字段清洗。
    """

    def __init__(self, file_path: str, use_cols: Optional[List[str]] = None):
        """
        :param file_path: CSV文件路径
        :param use_cols: 指定加载的列，默认为None加载全部
        """
        self.file_path = file_path
        self.use_cols = use_cols

    def load(self) -> pd.DataFrame:
        """
        加载CSV为DataFrame，并进行基础处理
        """
        df = pd.read_csv(self.file_path, usecols=self.use_cols)
        # 类型转换
        if "lat" in df.columns:
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        if "lng" in df.columns:
            df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
        return df


class AddressDataset(Dataset):
    """
    用于深度学习模型的地址数据Dataset
    """

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[Callable] = None):
        """
        :param dataframe: Pandas DataFrame
        :param transform: 可选的预处理函数，如tokenizer
        """
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample = {
            "id": row["id"],
            'prov': '北京市',
            "name": row["name"],
            "district": row["district"],
            "township": row["township"],
            "address": row["address"],
            "lat": row["lat"],
            "lng": row["lng"]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
