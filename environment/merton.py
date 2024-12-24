import numpy as np
import math

class MertonBandwidthSimulator:
    def __init__(self, length, mu=0.5, sigma=0.1, a=0.02, b=0.01, lamda=0.5, base=1e6):
        """
        Mertonジャンプモデルを使用して帯域幅をシミュレーションする。

        Args:
            length (int): シミュレーションするデータポイント数。
            mu (float): ドリフト率。
            sigma (float): 拡散の標準偏差。
            a (float): ジャンプ成分のスケール。
            b (float): ジャンプの分散のスケール。
            lamda (float): ポアソン分布のジャンプ発生率。
            base (float): 帯域幅のベース値（初期値）。
        """
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b
        self.lamda = lamda
        self.base = base
        self.series_bw = self._generate_series(length)

    def _generate_series(self, length):
        delta_t = 1
        series = []
        last_j = math.log(self.base)

        for _ in range(length):
            Z = np.random.normal(0, 1)
            N = np.random.poisson(self.lamda)
            Z_2 = np.random.normal(0, 1)

            M = self.a * N + self.b * (N**0.5) * Z_2
            new_j = last_j + (self.mu - 0.5 * self.sigma**2) * delta_t + self.sigma * (delta_t**0.5) * Z + M
            series.append(math.exp(new_j))
            last_j = new_j

        # 長さが正しいか確認
        assert len(series) == length, f"Generated series length {len(series)} does not match expected length {length}"
        return series


    def regenerate_bw(self, last_bw, length):
        """
        指定された長さの新しい帯域幅シーケンスを生成する。

        Args:
            last_bw (float): 前回の帯域幅（Mbps）。
            length (int): シミュレーションするデータポイント数。

        Returns:
            list: 新しい帯域幅シーケンス。
        """
        delta_t = 1
        new_sequence = []
        last_j = math.log(last_bw)

        for _ in range(length):
            Z = np.random.normal(0, 1)
            N = np.random.poisson(self.lamda)
            Z_2 = np.random.normal(0, 1)

            M = self.a * N + self.b * (N**0.5) * Z_2
            new_j = last_j + (self.mu - 0.5 * self.sigma**2) * delta_t + self.sigma * (delta_t**0.5) * Z + M
            new_bw = math.exp(new_j)
            new_sequence.append(new_bw)
            last_j = new_j

        return new_sequence