# 工业设备预测性维护项目 (Predictive Maintenance Project)

## 项目简介

本项目是一个基于深度学习的工业设备预测性维护系统，旨在通过对工业设备运行数据的分析，实现设备故障的早期预警和剩余使用寿命（RUL）预测。项目结合了 CNN（卷积神经网络）故障分类和 Autoencoder（自编码器）异常检测两种方法，对设备运行状态进行实时监控和预测。

## 项目结构

```
predictive_maintenance_project/
├── data/                  # 数据目录（数据文件较大，不包含在仓库中，详见下方"数据集与下载"章节）
├── models/                # 模型定义与训练代码
├── notebooks/             # Jupyter Notebook 实验与分析
├── src/                   # 源代码（数据预处理、特征工程、评估等）
├── results/               # 实验结果与可视化
├── requirements.txt       # Python 依赖
├── .gitignore             # Git 忽略规则
└── README.md              # 项目说明
```

## 技术路线

| 模块 | 方法 | 数据集 | 目标 |
|------|------|--------|------|
| 故障分类 | CNN（卷积神经网络） | FEMTO-ST 轴承退化数据集 | 识别轴承运行状态（正常/退化/故障） |
| 异常检测 | Autoencoder（自编码器） | MIMII DG 工业异常声音数据集 | 检测设备运行声音中的异常 |
| RUL 预测 | 深度学习回归 | FEMTO-ST 轴承退化数据集 | 预测轴承剩余使用寿命 |

## 数据集与下载

> **注意**：由于数据文件体积较大，本仓库 `data/` 目录下不包含实际数据文件。请根据以下说明自行下载并放置到相应子目录中。

### 1. FEMTO-ST 轴承退化数据集（PHM IEEE 2012 Data Challenge）

- **下载链接**：[https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset/](https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset/)
- **构建机构**：FEMTO-ST Institute
- **用途**：用于 CNN 故障分类和轴承剩余使用寿命（RUL）预测
- **数据描述**：
  - 训练集（Learning_set）：3 种不同工况下的 6 个轴承加速退化数据
  - 测试集（Test_set / Full_Test_Set）：11 个轴承的测试数据
  - 每个轴承采集水平和垂直两个方向的振动信号
  - 采样频率：25.6 kHz
- **建议目录结构**：
  ```
  data/
  └── femto/
      ├── Learning_set/
      ├── Test_set/
      ├── Full_Test_Set/
      └── IEEEPHM2012-Challenge-Details.pdf
  ```
- **下载方式**：
  ```bash
  # 方法1：直接 clone 仓库
  cd data/
  git clone https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset.git femto

  # 方法2：在 GitHub 页面点击 "Code" -> "Download ZIP"，解压后放入 data/femto/ 目录
  ```
- **引用**：
  > Nectoux, P., Gouriveau, R., Medjaher, K., Ramasso, E., Chebel-Morello, B., Zerhouni, N., ... Varnier, C. (2012). PRONOSTIA: An experimental platform for bearings accelerated degradation tests. In *IEEE International Conference on Prognostics and Health Management* (pp. 1-8). Denver.

---

### 2. MIMII DG 工业异常声音数据集

- **下载链接**：[https://zenodo.org/record/6529888](https://zenodo.org/record/6529888)
- **DOI**：10.5281/zenodo.6529888
- **构建机构**：Hitachi, Ltd.
- **用途**：用于 Autoencoder 异常检测
- **数据描述**：
  - 包含五种工业设备的运行声音数据：
    - 轴承（bearing） - 772.3 MB
    - 风扇（fan） - 928.5 MB
    - 齿轮箱（gearbox） - 946.4 MB
    - 滑轨（slider） - 913.7 MB
    - 阀门（valve） - 825.4 MB
  - 每种设备类型包含正常和异常运行的声音数据
  - 每种设备包含三个子集（sections），对应不同的域偏移类型
  - 总计约 4.4 GB
- **许可协议**：Creative Commons Attribution 4.0 International (CC BY 4.0)
- **建议目录结构**：
  ```
  data/
  └── mimii/
      ├── bearing/
      ├── fan/
      ├── gearbox/
      ├── slider/
      └── valve/
  ```
- **下载方式**：
  ```bash
  cd data/mimii/

  # 逐个下载（推荐，避免大文件下载中断）
  wget https://zenodo.org/records/6529888/files/bearing.zip
  wget https://zenodo.org/records/6529888/files/fan.zip
  wget https://zenodo.org/records/6529888/files/gearbox.zip
  wget https://zenodo.org/records/6529888/files/slider.zip
  wget https://zenodo.org/records/6529888/files/valve.zip

  # 解压
  unzip bearing.zip && unzip fan.zip && unzip gearbox.zip && unzip slider.zip && unzip valve.zip

  # 清理 zip 文件
  rm *.zip
  ```
- **MD5 校验**：
  | 文件名 | MD5 |
  |--------|-----|
  | bearing.zip | 6381a00f9efc0ced779c8ad847e4ff59 |
  | fan.zip | a1a9b488934a82426bacc933d87aacde |
  | gearbox.zip | c165dfef8c404256bd719c6fe1f7036f |
  | slider.zip | 8c3a5466cf53e54872fd94998a67bfac |
  | valve.zip | 1da37b2e82942dfba720984541e2ef60 |
- **引用**：
  > Dohi, K., Nishida, T., Purohit, H., Tanabe, R., Endo, T., Yamamoto, M., Nikaido, Y., & Kawaguchi, Y. (2022). MIMII DG: Sound Dataset for Malfunctioning Industrial Machine Investigation for Domain Generalization Task [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6529888

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 推荐 Python 版本

- Python >= 3.8

### 主要依赖库

- TensorFlow / PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Scipy

## 使用方法

1. **下载数据集**：根据上方"数据集与下载"章节的链接下载数据，解压后放入 `data/` 目录
2. **数据预处理**：运行 `src/` 目录下的预处理脚本
3. **模型训练**：运行 `models/` 目录下的训练脚本
4. **结果评估**：查看 `notebooks/` 中的分析 Notebook 或运行评估脚本

## 许可证

本项目仅供学习和研究使用。使用的数据集分别遵循其各自的许可协议。

## 联系方式

如有问题，欢迎通过 GitHub Issues 反馈。
