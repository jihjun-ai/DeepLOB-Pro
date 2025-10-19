import json

# 讀取新數據分佈
with open('data/processed_v5_balanced/npz/normalization_meta.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

# 訓練集分佈
train_dist = meta['data_split']['results']['train']['label_dist']
total = sum(train_dist)

print('=' * 80)
print('新數據分佈驗證 (config_pro_v5_balanced_optimal.yaml)')
print('=' * 80)
print()
print('訓練集 (1,249,419 樣本):')
print(f'  Class 0 (下跌): {train_dist[0]:>7,} ({train_dist[0]/total*100:5.2f}%)')
print(f'  Class 1 (持平): {train_dist[1]:>7,} ({train_dist[1]/total*100:5.2f}%)')
print(f'  Class 2 (上漲): {train_dist[2]:>7,} ({train_dist[2]/total*100:5.2f}%)')
print(f'  總計:          {total:>7,}')
print()

# 與原始數據對比
original = [367457, 562361, 319601]
print('與原始數據對比 (config_pro_v5_ml_optimal.yaml):')
print(f'  Class 0: {original[0]/sum(original)*100:5.2f}% → {train_dist[0]/total*100:5.2f}% (變化: {train_dist[0]/total*100 - original[0]/sum(original)*100:+5.2f}%)')
print(f'  Class 1: {original[1]/sum(original)*100:5.2f}% → {train_dist[1]/total*100:5.2f}% (變化: {train_dist[1]/total*100 - original[1]/sum(original)*100:+5.2f}%)')
print(f'  Class 2: {original[2]/sum(original)*100:5.2f}% → {train_dist[2]/total*100:5.2f}% (變化: {train_dist[2]/total*100 - original[2]/sum(original)*100:+5.2f}%)')
print()

# 目標達成度
target = [31.5, 35.0, 33.5]
print('目標達成度 (目標: 30-33% / 33-37% / 30-33%):')
for i, name in enumerate(['Class 0', 'Class 1', 'Class 2']):
    actual = train_dist[i]/total*100
    deviation = abs(actual - target[i])
    status = '✅' if deviation < 3.0 else '⚠️' if deviation < 5.0 else '❌'
    print(f'  {name}: {actual:5.2f}% (目標 {target[i]:.1f}%, 偏差 {deviation:4.2f}%) {status}')

# 總偏差
total_deviation = sum(abs(train_dist[i]/total*100 - target[i]) for i in range(3))
print()
print(f'總偏差: {total_deviation:.2f}%')
if total_deviation < 8.0:
    print('評價: ✅ 優秀！非常接近目標分佈')
elif total_deviation < 12.0:
    print('評價: ✅ 良好，可接受範圍')
else:
    print('評價: ⚠️ 需要微調')