import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# ================ 1. 数据集构建 ================
def create_dataset(n_samples=5000, n_features=20, random_state=42):
    # 生成合成数据集
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],  # 类别不平衡
        random_state=random_state
    )
    
    # 转换为DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df = df.astype({col: 'float32' for col in feature_names})  # 内存优化
    df['target'] = y
    
    # 添加缺失值以模拟真实场景
    mask = np.random.random(df.shape) < 0.05  # 5%的数据为缺失值
    df = df.mask(mask)
    
    # 添加一些异常值
    for col in feature_names[:5]:
        outliers = np.random.choice(range(len(df)), size=50)
        df.loc[outliers, col] *= 5
    
    return df

# ================ 2. 数据清洗 ================
def clean_data(df):
    # 复制数据以避免修改原始数据
    cleaned_df = df.copy()
    
    # 处理目标变量中的缺失值
    cleaned_df = cleaned_df.dropna(subset=['target'])
    
    numerical_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()
    numerical_cols.remove('target')
    
    # 处理异常值（使用IQR方法）
    for col in numerical_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        median_val = cleaned_df[col].median()
        cleaned_df.loc[cleaned_df[col] < lower_bound, col] = median_val
        cleaned_df.loc[cleaned_df[col] > upper_bound, col] = median_val
    
    # 处理特征中的缺失值（使用中位数填充）
    imputer = SimpleImputer(strategy='median')
    cleaned_df[numerical_cols] = imputer.fit_transform(cleaned_df[numerical_cols])
    
    return cleaned_df

# ================ 3. 特征工程与数据分割 ================
def prepare_data(df):
    # 分离特征和目标变量
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 特征选择
    selector = SelectKBest(score_func=f_classif, k=15)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # 分割数据集为训练集和测试集（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, selected_features

# ================ 4. 模型训练与评估 ================
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # 定义要评估的模型
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    results = {}
    
    # 训练和评估每个模型
    for name, model in models.items():
        print(f"\n正在训练 {name} 模型...")
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"  交叉验证准确率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # 训练模型
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # 预测
        start_predict = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_predict
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(y_test))
        
        # 评估指标
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if hasattr(model, 'predict_proba') else 0.5
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'report': report,
            'confusion_matrix': cm,
            'train_time': train_time,
            'predict_time': predict_time
        }
        
        print(f"\n{name} 模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"训练时间: {train_time:.4f}秒")
        print(f"预测时间: {predict_time:.4f}秒")
        print(f"分类报告:\n{report}")
        
        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['预测负类', '预测正类'], 
                   yticklabels=['实际负类', '实际正类'])
        plt.title(f'{name} 混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('实际标签')
        plt.tight_layout()
        plt.savefig(f'{name}_confusion_matrix.png')
        plt.close()
    
    return results

# ================ 5. 特征重要性分析 ================
def analyze_feature_importance(results, feature_names):
    # 只处理支持特征重要性的模型
    importance_models = {}
    for name, result in results.items():
        if hasattr(result['model'], 'feature_importances_'):
            importance_models[name] = result
    
    if not importance_models:
        print("\n没有模型支持特征重要性分析")
        return
    
    # 创建组合图表
    n_models = len(importance_models)
    plt.figure(figsize=(5 * n_models, 8))
    plt.suptitle('特征重要性比较', fontsize=16)
    
    # 为每个模型创建子图
    for i, (name, result) in enumerate(importance_models.items(), 1):
        importance = result['model'].feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # 只显示top 10特征
        top_k = min(10, len(feature_names))
        top_indices = indices[:top_k]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance[top_indices]
        
        plt.subplot(1, n_models, i)
        plt.barh(range(top_k), top_importance[::-1], color='skyblue')
        plt.yticks(range(top_k), top_features[::-1])
        plt.title(f'{name}')
        plt.xlabel('重要性分数')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    plt.savefig('combined_feature_importance.png')
    plt.close()
    
    # 单独输出随机森林的特征重要性
    if 'Random Forest' in importance_models:
        print("\n随机森林特征重要性排名:")
        importance = importance_models['Random Forest']['model'].feature_importances_
        indices = np.argsort(importance)[::-1]
        for j in range(len(feature_names)):
            print(f"{j+1}. {feature_names[indices[j]]}: {importance[indices[j]]:.4f}")

# ================ 6. 模型性能比较 ================
def compare_model_performance(results):
    """比较不同模型的性能并可视化"""
    # 提取性能指标
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    auc_scores = [results[name]['auc'] for name in model_names]
    f1_scores = [results[name]['f1'] for name in model_names]
    train_times = [results[name]['train_time'] for name in model_names]
    predict_times = [results[name]['predict_time'] for name in model_names]
    
    # 创建图表
    plt.figure(figsize=(18, 10))
    
    # 准确率比较
    plt.subplot(2, 3, 1)
    plt.bar(model_names, accuracies, color='skyblue')
    plt.title('模型准确率比较')
    plt.ylim(0.7, 1.0)
    plt.ylabel('准确率')
    
    # AUC比较
    plt.subplot(2, 3, 2)
    plt.bar(model_names, auc_scores, color='lightgreen')
    plt.title('模型AUC比较')
    plt.ylim(0.7, 1.0)
    plt.ylabel('AUC')
    
    # F1分数比较
    plt.subplot(2, 3, 3)
    plt.bar(model_names, f1_scores, color='salmon')
    plt.title('模型F1分数比较')
    plt.ylim(0.7, 1.0)
    plt.ylabel('F1分数')
    
    # 训练时间比较
    plt.subplot(2, 3, 4)
    plt.bar(model_names, train_times, color='gold')
    plt.title('模型训练时间比较')
    plt.ylabel('训练时间 (秒)')
    
    # 预测时间比较
    plt.subplot(2, 3, 5)
    plt.bar(model_names, predict_times, color='violet')
    plt.title('模型预测时间比较')
    plt.ylabel('预测时间 (秒)')
    
    # 综合性能雷达图
    plt.subplot(2, 3, 6, polar=True)
    metrics = ['准确率', 'AUC', 'F1分数', '训练时间', '预测时间']
    n_metrics = len(metrics)
    
    # 标准化指标值
    def normalize(values, invert=False):
        min_val, max_val = min(values), max(values)
        if invert:  # 对于时间指标，越小越好
            return [1 - (v - min_val)/(max_val - min_val + 1e-10) for v in values]
        else:
            return [(v - min_val)/(max_val - min_val + 1e-10) for v in values]
    
    norm_acc = normalize(accuracies)
    norm_auc = normalize(auc_scores)
    norm_f1 = normalize(f1_scores)
    norm_train_time = normalize(train_times, invert=True)
    norm_predict_time = normalize(predict_times, invert=True)
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合多边形
    
    for i, name in enumerate(model_names):
        values = [
            norm_acc[i],
            norm_auc[i],
            norm_f1[i],
            norm_train_time[i],
            norm_predict_time[i]
        ]
        values += values[:1]  # 闭合多边形
        plt.plot(angles, values, 'o-', linewidth=2, label=name)
        plt.fill(angles, values, alpha=0.1)
    
    plt.xticks(angles[:-1], metrics)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"])
    plt.title('模型综合性能雷达图')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    plt.close()

# ================ 主程序 ================
if __name__ == "__main__":
    start_time = time.time()
    print("="*50)
    print("开始机器学习项目流水线")
    print("="*50)
    
    # 1. 构建数据集
    print("\n[阶段1] 构建数据集...")
    df = create_dataset(n_samples=5000)
    print(f"数据集构建完成，形状: {df.shape}")
    print(f"类别分布:\n{df['target'].value_counts(normalize=True)}")
    
    # 2. 数据清洗
    print("\n[阶段2] 数据清洗...")
    cleaned_df = clean_data(df)
    print(f"数据清洗完成，剩余样本数: {len(cleaned_df)}")
    print(f"清洗后类别分布:\n{cleaned_df['target'].value_counts(normalize=True)}")
    
    # 3. 数据准备
    print("\n[阶段3] 数据准备...")
    X_train, X_test, y_train, y_test, selected_features = prepare_data(cleaned_df)
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    print(f"选择的特征 ({len(selected_features)}个): {', '.join(selected_features)}")
    
    # 4. 模型训练与评估
    print("\n[阶段4] 模型训练与评估...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 5. 特征重要性分析
    print("\n[阶段5] 特征重要性分析...")
    analyze_feature_importance(results, selected_features)
    
    # 6. 模型性能比较
    print("\n[阶段6] 模型性能比较...")
    compare_model_performance(results)
    
    # 打印最终比较结果
    print("\n" + "="*50)
    print("最终模型性能比较:")
    print("{:<20} {:<10} {:<10} {:<10} {:<12} {:<12}".format(
        '模型', '准确率', 'AUC', 'F1分数', '训练时间(秒)', '预测时间(秒)'))
    for name, res in results.items():
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<12.4f} {:<12.4f}".format(
            name, res['accuracy'], res['auc'], res['f1'], res['train_time'], res['predict_time']))
    
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"项目完成! 总执行时间: {total_time:.2f}秒")
    print("所有结果已保存为图表文件")
    print("="*50)