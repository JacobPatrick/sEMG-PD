from sklearn.model_selection import GridSearchCV


def grid_param_tunning(model, param_grid, X_train, y_train):
    """模型参数网格调优

    Args:
        model: 待调优的模型
        param_grid: 参数调优范围字典
        X_train: 训练集特征
        y_train: 训练集标签

    Returns:
        best_params: 最佳参数字典
        best_score: 最佳分数
    """
    # 创建GridSearchCV对象
    # FIXME: 暂时设置不进行交叉验证，后续获得足够样本量可以正常进行
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    # 执行网格搜索
    grid_search.fit(X_train, y_train)
    # 返回最佳参数和最佳分数
    return grid_search.best_params_, grid_search.best_score_
