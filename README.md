# 基于 sEMG 的 PD 运动症状评估

## 项目简介

## 运行环境

## 快速开始

### 1. 安装依赖

```bash
pip install .
```

### 2. 运行项目

1. 定义配置文件：在 `src/config` 目录下创建配置文件；
2. 运行脚本：在 `scripts` 目录下运行 `main.py` 脚本，或自定义脚本；
3. 检查结果：查看 `reports` 目录下的报告。

### 3. 代码格式化

```bash
black .
```

## 项目结构

```bash
├─data                   # 数据（已分割）
├─model                  # 训练好的模型
├─reports                # 实验报告
├─scripts                # 运行入口
├─src
│  ├─config              # 配置文件
│  ├─core                # 流水线模块工厂
│  ├─interfaces          # 流水线模块接口
│  ├─pipeline            # 流水线模块实现
│  │  ├─classification   # 分类模型
│  │  ├─data             # 加载数据
│  │  ├─feature          # 特征提取
│  │  ├─preprocess       # 预处理
│  │  └─split            # 分割数据集
│  └─utils               # 工具函数
└─tests                  # 单元测试
    ├─integration
    ├─test_data
    └─unit
```
