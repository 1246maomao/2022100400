请安装以下Python库：

numpy==1.24.3
pandas==1.5.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2

安装依赖
pip install -r requirements.txt

项目结构.
├── README.md               # 项目说明文档
├── requirements.txt        # Python依赖库列表
├── ml_pipeline.py          # 主程序代码
├── Random Forest_confusion_matrix.png      # 随机森林混淆矩阵
├── Gradient Boosting_confusion_matrix.png  # 梯度提升机混淆矩阵
├── SVM_confusion_matrix.png                # SVM混淆矩阵
├── combined_feature_importance.png         # 特征重要性比较图
└── model_performance_comparison.png        # 模型性能比较图

运行完整流水线
python ml_pipeline.py

可以在ml_pipeline.py中修改以下参数：
# 数据集参数
n_samples = 5000    # 样本数量
n_features = 20     # 特征数量
weights = [0.7, 0.3] # 类别不平衡比例

# 特征工程参数
k = 15              # 选择的最佳特征数量

# 模型参数
models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}
