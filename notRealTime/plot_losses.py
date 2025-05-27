import matplotlib.pyplot as plt

# 训练记录
steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
style_loss = [191.882370, 52.532497, 13.773139, 6.743725, 4.546108,
              3.352822, 2.514446, 1.965243, 1.556310, 1.340904]
content_loss = [37.860607, 36.277004, 34.227951, 31.228165, 28.683067,
                26.722250, 25.450394, 24.532776, 23.818813, 23.339428]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(steps, style_loss, marker='o', label='Style Loss')
plt.plot(steps, content_loss, marker='s', label='Content Loss')

plt.title('Loss Trend (chicago → StarryNight)', fontsize=14)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
