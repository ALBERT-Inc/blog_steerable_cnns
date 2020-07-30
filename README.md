Steerable CNNs に関するブログ用リポジトリ

## インストール

Python 3.6 以上が必要です。CUDA 10.1 と Linux x86_64 でのみ動作確認しています。CuPy が動く環境なら他の環境でも問題なく動作すると思います。

```shell
$ pip install .
```

でインストール可能です。

## リポジトリ構成

* `steerable_cnns/group/`: Steerable CNNs に必要な有限群に関する処理、および具体的な群の定義
* `steerable_cnns/network/`: Chainer を用いての、convolution などの処理と ResNet の実装
* `sample/`: ニューラルネットを訓練するサンプル（Chainer のサンプルコードを元にしています）
* `run-qa.sh`: コードリントと単体テストを実行

## 訓練実行

```
$ python sample/train_cifar.py --dataset cifar100 --augmentation
```

## 謝辞
このリポジトリは、[grafi-tt](https://github.com/grafi-tt) さんが以前実装したものを整理したものです。感謝いたします。
