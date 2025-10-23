import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


# 基础设置
SEED = 760
np.random.seed(SEED)
tf.random.set_seed(SEED)

TIMESTEPS = 7
LR = 1e-4
EPOCHS = 40
BATCH_SIZE = 16


# Step 1. 加载并清洗新西兰数据

akl = pd.read_csv("AKL_Data_normalised_Train_Set.csv")
chc = pd.read_csv("CHCNZ_Data_normalised_Train_Set.csv")
wlg = pd.read_csv("WLGNZ_Data_normalised_Train_Set.csv")

nz_df = pd.concat([akl, chc, wlg], axis=0).reset_index(drop=True)
print(f"Combined NZ dataset size: {nz_df.shape}")

# 转为纯数值
nz_df = nz_df.apply(pd.to_numeric, errors="coerce").fillna(0)

# 分离特征与标签
X = nz_df.drop(columns=["outage_flag"], errors="ignore").astype("float32")
y = nz_df["outage_flag"].astype(int)

# ------------------------------------------------------------
# Step 2. 构造时间序列输入 (LSTM 输入要求 3D)
# ------------------------------------------------------------
def create_sequences(X, y, timesteps=7):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X.iloc[i:(i + timesteps)].values)
        ys.append(y.iloc[i + timesteps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X, y, TIMESTEPS)
print(f"LSTM input shape: {X_seq.shape}, target shape: {y_seq.shape}")

# ------------------------------------------------------------
# Step 3. 使用 SMOTE 平衡样本
# ------------------------------------------------------------
print("Applying SMOTE oversampling for class balance...")
X_flat = X_seq.reshape(X_seq.shape[0], -1)
sm = SMOTE(random_state=SEED)
X_res, y_res = sm.fit_resample(X_flat, y_seq)
X_seq_bal = X_res.reshape(-1, TIMESTEPS, X_seq.shape[2])
y_seq_bal = y_res
print(f"After SMOTE: class counts = {np.bincount(y_seq_bal)}")

# ------------------------------------------------------------
# Step 4. 加载美国预训练模型
# ------------------------------------------------------------
usa_model = load_model("usa_lstm_model.h5")
print("USA pretrained model loaded.")
usa_model.summary()

# ------------------------------------------------------------
# Step 5. 构建新西兰专用模型（结构不同 → 重建）
# ------------------------------------------------------------
n_features = X_seq.shape[2]
n_timesteps = X_seq.shape[1]
print(f"Building new NZ model: timesteps={n_timesteps}, features={n_features}")

transfer_model = Sequential([
    Input(shape=(n_timesteps, n_features)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

# 尝试迁移 Dense 层权重（若维度匹配则加载，否则跳过）
usa_weights = usa_model.get_layer("dense").get_weights()
try:
    transfer_model.layers[-1].set_weights(usa_weights)
    print("Loaded USA Dense layer weights successfully.")
except ValueError:
    print("Dense layer shape mismatch — skipping weight transfer.")

# 使用 Focal Loss（比 BCE 更适合不平衡任务）
focal_loss = BinaryFocalCrossentropy(gamma=2.0, alpha=0.75)

transfer_model.compile(
    optimizer=Adam(learning_rate=LR),
    loss=focal_loss,
    metrics=["accuracy"]
)

transfer_model.summary()

# ------------------------------------------------------------
# Step 6. 训练 (Fine-tuning)
# ------------------------------------------------------------
history = transfer_model.fit(
    X_seq_bal, y_seq_bal,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    shuffle=True,
    verbose=1
)

# ------------------------------------------------------------
# Step 7. 评估模型性能
# ------------------------------------------------------------
y_pred_prob = transfer_model.predict(X_seq)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n[Classification Report]")
print(classification_report(y_seq, y_pred, digits=4))

f1 = f1_score(y_seq, y_pred)
recall = recall_score(y_seq, y_pred)
auc = roc_auc_score(y_seq, y_pred_prob)
print(f"F1 Score = {f1:.4f} | Recall = {recall:.4f} | AUC = {auc:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_seq, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - NZ Fine-tuned LSTM (Enhanced)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("NZ_LSTM_confusion_matrix_enhanced.png", dpi=300)
plt.show()

# 概率分布可视化
plt.figure()
plt.hist(y_pred_prob, bins=50, color='purple', alpha=0.7)
plt.title("Prediction Probability Distribution (outage=1)")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("NZ_LSTM_pred_distribution.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Step 8. 保存微调模型
# ------------------------------------------------------------
transfer_model.save("nz_finetuned_enhanced.h5")
print("\nFine-tuned enhanced model saved as nz_finetuned_enhanced.h5")
