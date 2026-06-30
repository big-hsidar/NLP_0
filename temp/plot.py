import matplotlib.pyplot as plt

# ==================== 数据（从日志提取） ====================
epochs = list(range(1, 25))

train_loss = [
    6.5577, 6.4467, 6.2273, 6.0169, 5.8516, 5.7089, 5.5724, 5.4289,
    5.2790, 5.1259, 4.9716, 4.8182, 4.6578, 4.4913, 4.3245, 4.1632,
    4.0064, 3.8516, 3.6984, 3.3850, 3.2544, 3.1508, 3.0515, 2.9552
]

val_loss = [
    6.1652, 6.0483, 5.8004, 5.6673, 5.5461, 5.4870, 5.4015, 5.3587,
    5.3325, 5.2996, 5.2867, 5.2854, 5.2971, 5.3296, 5.3442, 5.3859,
    5.4337, 5.4777, 5.5107, 5.5545, 5.5967, 5.6247, 5.6665, 5.7035
]

top1 = [0.0420, 0.0520, 0.0864, 0.1053, 0.1239, 0.1285, 0.1412, 0.1452,
        0.1496, 0.1565, 0.1608, 0.1627, 0.1670, 0.1642, 0.1662, 0.1593,
        0.1590, 0.1579, 0.1580, 0.1534, 0.1547, 0.1537, 0.1488, 0.1490]

top3 = [0.1066, 0.1323, 0.1720, 0.1977, 0.2181, 0.2304, 0.2456, 0.2555,
        0.2598, 0.2638, 0.2690, 0.2704, 0.2721, 0.2715, 0.2694, 0.2673,
        0.2600, 0.2567, 0.2575, 0.2482, 0.2457, 0.2447, 0.2423, 0.2414]

top5 = [0.1526, 0.1791, 0.2214, 0.2513, 0.2708, 0.2817, 0.3015, 0.3092,
        0.3130, 0.3179, 0.3212, 0.3249, 0.3258, 0.3219, 0.3211, 0.3197,
        0.3130, 0.3079, 0.3045, 0.3005, 0.2951, 0.2931, 0.2887, 0.2902]

top10 = [0.2230, 0.2443, 0.2935, 0.3275, 0.3476, 0.3546, 0.3716, 0.3812,
         0.3859, 0.3934, 0.3956, 0.3988, 0.4003, 0.3949, 0.3956, 0.3905,
         0.3836, 0.3797, 0.3775, 0.3719, 0.3679, 0.3656, 0.3623, 0.3599]

mrr = [0.1032, 0.1201, 0.1569, 0.1796, 0.1988, 0.2060, 0.2203, 0.2270,
       0.2311, 0.2369, 0.2415, 0.2432, 0.2465, 0.2434, 0.2440, 0.2393,
       0.2360, 0.2336, 0.2331, 0.2274, 0.2267, 0.2251, 0.2208, 0.2203]

# ==================== 绘图 ====================
plt.rcParams['font.size'] = 11
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# ---- 子图 1：损失曲线 ----
ax1.plot(epochs, train_loss, label='Train Loss',
         marker='o', linewidth=2, markersize=6)
ax1.plot(epochs, val_loss, label='Validation Loss',
         marker='s', linewidth=2, markersize=6)
ax1.axvline(x=12, color='red', linestyle='--', alpha=0.8,
            linewidth=1.5, label='Best Model (Epoch 12)')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curves', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

# ---- 子图 2：准确率 & MRR ----
ax2.plot(epochs, top1, label='Top-1 Acc',
         marker='o', linewidth=2, markersize=5)
ax2.plot(epochs, top3, label='Top-3 Acc',
         marker='s', linewidth=2, markersize=5)
ax2.plot(epochs, top5, label='Top-5 Acc',
         marker='^', linewidth=2, markersize=5)
ax2.plot(epochs, top10, label='Top-10 Acc',
         marker='D', linewidth=2, markersize=5)
ax2.plot(epochs, mrr, label='MRR', marker='*', linewidth=2, markersize=7)
ax2.axvline(x=12, color='red', linestyle='--',
            alpha=0.8, linewidth=1.5, label='Best Model')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Score')
ax2.set_title('Accuracy & MRR', fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('lstm_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()
