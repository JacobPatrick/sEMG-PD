from typing import Type
from pipeline.pipeline import MLPipeline


class PipelineFactory:
    def __init__(self):
        self._pipelines = {}

    def register(self, name: str, pipeline_class: Type[MLPipeline], **kwargs):
        """注册新的pipeline类型"""
        self._pipelines[name] = (pipeline_class, kwargs)

    def create(self, name: str) -> MLPipeline:
        """创建pipeline实例"""
        if name not in self._pipelines:
            raise ValueError(f"Unknown pipeline type: {name}")
        pipeline_class, kwargs = self._pipelines[name]
        return pipeline_class(**kwargs)
