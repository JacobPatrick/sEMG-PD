class TestSVMTrainer:
    """SVM训练器单元测试"""
    
    def test_model_training(self, mock_feature_data):
        """测试模型训练"""
        trainer = SVMTrainer()
        # 使用已知的特征数据，便于验证训练结果
        model = trainer.train(mock_feature_data, config)
        
        # 验证模型参数
        assert isinstance(model, SVC)
        assert model.kernel == 'rbf' 