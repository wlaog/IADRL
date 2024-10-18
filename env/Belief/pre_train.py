import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

class ModelComparison:
    def __init__(self, data_path):
        # 加载数据
        data = pd.read_csv(data_path)
        self.X = data[['xcar', 'vcar', 'acar', 'ttccar', 'xped', 'vped', 'aped', 'ttcped']]
        self.y = data['belief']
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2)
        
        # 标准化数据
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # 初始化模型
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'SVR': SVR(kernel='rbf')
        }
        
        # 存储评估结果和训练后的模型
        self.results = {}
        self.trained_models = {}

    def train_model(self, model_name):
        # 训练单个模型
        model = self.models[model_name]
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        # 保存评估结果
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        self.results[model_name] = {'MSE': mse, 'R2': r2, 'CV MSE': -cv_scores.mean()}
        self.trained_models[model_name] = model  # 将训练好的模型保存在 self.trained_models 中
        
        print(f"{model_name}: MSE={mse:.4f}, R2={r2:.4f}, CV MSE={-cv_scores.mean():.4f}")

    def evaluate_models(self):
        # 训练并评估所有模型
        for model_name in self.models.keys():
            self.train_model(model_name)

    def compare_models(self):
        # 将结果转换为 DataFrame 以便绘图
        results_df = pd.DataFrame(self.results).T
        
        # 绘制 MSE 和 R2 比较图
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=results_df.index, y="MSE", data=results_df)
        plt.title("Mean Squared Error (MSE) Comparison")
        plt.ylabel("MSE")
        plt.xlabel("Model")
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=results_df.index, y="R2", data=results_df)
        plt.title("R-Squared (R2) Comparison")
        plt.ylabel("R2")
        plt.xlabel("Model")
        
        plt.tight_layout()
        plt.show()
        
        # 选择最佳模型并绘制真实值 vs 预测值图
        best_model_name = results_df['R2'].idxmax()
        best_model = self.trained_models[best_model_name]  # 从 self.trained_models 中获取最佳模型
        y_pred_best = best_model.predict(self.X_test)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, y_pred_best, alpha=0.7)
        plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], '--r', linewidth=2)
        plt.xlabel("True Belief Values")
        plt.ylabel("Predicted Belief Values")
        plt.title(f"True vs Predicted Belief Values - {best_model_name}")
        plt.show()

# 定义 Shapley 分析函数
def shap_analysis(model, X_train, X_test, feature_names):
    # 检查是否可以进行 shap 解释
    if not hasattr(model, "predict"):
        raise ValueError("传入的模型无效，无法进行预测！")
    
    # 计算 SHAP 值
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    # Summary Plot - 显示特征的全局重要性
    plt.title("Feature Importance Summary (SHAP values)")
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    
    # Dependence Plot - 显示某个特征的具体影响（例如 'xcar'）
    plt.title("Dependence Plot of 'xcar'")
    shap.dependence_plot("xcar", shap_values.values, X_test, feature_names=feature_names)

# main 部分
if __name__ == "__main__":
    data_path = 'env/Belief/exp_data.csv'  # 数据文件路径
    model_comp = ModelComparison(data_path)   # 实例化 ModelComparison 类
    
    # 单独训练 Random Forest 模型
    model_comp.train_model('Random Forest')
    
    # 随机森林模型训练完成后可以从 trained_models 中获取模型
    rf_model = model_comp.trained_models['Random Forest']
    
    # # 进行预测示例（以测试集数据为例）
    # y_pred_rf = rf_model.predict(model_comp.X_test)
    # # 打印预测结果和实际值对比
    # print("Random Forest 预测结果：", y_pred_rf[:10])  # 显示前10个预测值
    # print("实际值：", model_comp.y_test[:10].values)     # 显示前10个实际值

    # 新的原始数据（请确保列名与训练数据相同）
    new_data = pd.DataFrame({
        'xcar': [11.83173049],
        'vcar': [3.54032],
        'acar': [-1.09153],
        'ttccar': [3.341994646],
        'xped': [6.071083624],
        'vped': [0.47598],
        'aped': [-0.46742],
        'ttcped': [12.75491328]
    })

    # 标准化新数据
    new_data_scaled = model_comp.scaler.transform(new_data)  # 使用之前训练的 scaler

    # 使用训练好的 Random Forest 模型进行预测
    new_predictions = rf_model.predict(new_data_scaled)
    
    # 输出预测结果
    print("对新数据的预测结果：", new_predictions)

    # # 对 Random Forest 模型进行 Shapley 分析
    # rf_model = model_comp.trained_models['Random Forest']
    # shap_analysis(rf_model, model_comp.X_train, model_comp.X_test, feature_names=model_comp.X.columns)

    model_comp.evaluate_models()  # 训练并评估所有模型
    model_comp.compare_models()    # 比较模型并绘制图表
