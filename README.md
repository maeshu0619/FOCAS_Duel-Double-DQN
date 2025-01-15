# 通信品質と視覚最適化を両立した超解像による動画配信システム

**Introduction**

2024年度の卒業研究のコードです。
DQNによって学習を行います。


**Installation**

1. **リポジトリのクローン方法:**
   ```bash
   git clone https://github.com/maeshu0619/graduation_experiment2024.git
   ```
   
2. **ライブラリのダウンロード:**
    ```bash
    cd your-repo-name
    pip install -r requirements.txt
    ```
   
**トレーニング方法**
ターミナル上でpython train.py --mode 0 --late 15 --net 0の様に入力してください

mode:
    0: ABRのトレーニングを行います
    1: FOCASのトレーニング行います
    2: 提案手法（適応型FOCAS）のトレーニングを行います

late:
    15: レイテンシ制約15msとしてトレーニングを行います
    20: レイテンシ制約20msとしてトレーニングを行います
    25: レイテンシ制約25msとしてトレーニングを行います

net:
    0: 悪質な通信環境（伝送レート）でトレーニングを行います
    1: 並の通信環境（伝送レート）でトレーニングを行います
    2: 良質な通信環境（伝送レート）でトレーニングを行います
