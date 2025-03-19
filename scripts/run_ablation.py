from src.config.config import load_config
from src.core.factories import setup_factories
from src.pipeline.pipeline import StandardMLPipeline
from experiments.ablation.feature_ablation import FeatureAblationStudy

def main():
    # 加载配置
    config = load_config("ablation_config", "experiments/ablation/configs/ablation.yaml")
    
    # 设置流水线
    factories = setup_factories()
    pipeline = StandardMLPipeline(**factories)
    
    # 运行消融实验
    study = FeatureAblationStudy(pipeline, config)
    # results = study.run(features_data, config.ablation)   # TODO: 消融实验相关配置与运行逻辑之后再改
    
    # 保存和可视化结果
    ...

if __name__ == "__main__":
    main() 