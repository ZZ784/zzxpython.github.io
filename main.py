import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,roc_curve,auc,classification_report

# 中文显示设置
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 1.数据读取 10分 =====================
df = pd.read_csv("pic_clinical_data.csv",encoding="gbk")
print("数据集行列数：",df.shape)
print("\n数据前5行")
print(df.head())
print("\n数据基本信息")
print(df.info())
print("\n描述性统计")
print(df.describe())

# ===================== 2.数据预处理 10分 =====================
# 去除重复值
df = df.drop_duplicates()
# 年龄列缺失值填充
df["年龄"] = df["年龄"].fillna(df["年龄"].median())
# 空腹血糖列缺失值填充
df["空腹血糖"] = df["空腹血糖"].fillna(df["空腹血糖"].mean())
# 异常值过滤
df = df[(df["年龄"]>0)&(df["年龄"]<120)]
# 性别编码转换
df["性别"] = df["性别"].map({"男":1,"女":0})
# 最终清洗数据
df_clean = df.dropna()
print(f"\n清洗后数据量：{df_clean.shape}")

# ===================== 3.统计分析 20分 =====================
# 1.患病分布柱状图
plt.figure(figsize=(6,4))
df_clean["患病标签"].value_counts().plot(kind="bar",color=["#6699cc","#ee7766"])
plt.title("疾病患病分布")
plt.xlabel("0-未患病 1-患病")
plt.ylabel("人数")
plt.tight_layout()
plt.savefig("disease_dist.png",dpi=300)
plt.close()

# 2.年龄分布直方图
plt.figure(figsize=(6,4))
sns.histplot(df_clean["年龄"],kde=True,color="#55aabb")
plt.title("患者年龄分布")
plt.xlabel("年龄")
plt.ylabel("频次")
plt.tight_layout()
plt.savefig("age_dist.png",dpi=300)
plt.close()

# 3.指标相关性热图
plt.figure(figsize=(8,6))
corr_mat = df_clean.corr()
sns.heatmap(corr_mat,cmap="coolwarm",annot=True,fmt=".2f")
plt.title("临床指标相关性热图")
plt.tight_layout()
plt.savefig("corr_heat.png",dpi=300)
plt.close()

# 4.组间指标均值对比
group_stat = df_clean.groupby("患病标签")[["年龄","收缩压","空腹血糖"]].mean()
print("\n患病/未患病组指标均值对比")
print(group_stat)

# ===================== 4.预测模型建立 10分 =====================
# 划分特征与标签
X = df_clean.drop("患病标签",axis=1)
y = df_clean["患病标签"]
# 划分训练集测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# 随机森林分类模型
rf_model = RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(X_train,y_train)
print("\n模型训练完成")

# ===================== 5.模型评估与可视化 20分 =====================
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:,1]

# 分类评估报告
print("\n模型分类评估报告")
print(classification_report(y_test,y_pred))

# ROC曲线
plt.figure(figsize=(6,4))
fpr,tpr,_ = roc_curve(y_test,y_prob)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,lw=2,label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("假阳性率")
plt.ylabel("真阳性率")
plt.title("模型ROC曲线")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png",dpi=300)
plt.close()

# 特征重要性
plt.figure(figsize=(6,4))
feat_import = pd.Series(rf_model.feature_importances_,index=X.columns)
feat_import.sort_values().plot(kind="barh",color="#77bb88")
plt.title("特征重要性排序")
plt.tight_layout()
plt.savefig("feature_import.png",dpi=300)
plt.close()

print("\n全部分析流程执行完毕，图表已保存至本地")
input()
