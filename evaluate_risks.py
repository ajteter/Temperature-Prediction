import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
import data_handler
import model

print("="*80)
print("评估季节性加权赛马方法的潜在风险")
print("="*80)

# 加载数据
gistemp_data = data_handler.get_clean_data('GLB.Ts+dSST.txt')
enso_data = data_handler.get_enso_data('https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt')
full_df = pd.concat([gistemp_data, enso_data], axis=1)
full_df.dropna(inplace=True)
df_modern = full_df.loc["1970-01-01":].copy()

target_date = pd.to_datetime("202511", format='%Y%m')
target_month = 11
last_available_date = df_modern.index.max()

print("\n" + "="*80)
print("风险1：评估样本量不足")
print("="*80)

# 测试不同窗口的样本量
windows = [15, 18, 21, 24, 27, 30, 33, 36]
sample_analysis = []

for months in windows:
    split_date = target_date - relativedelta(months=months)
    
    if split_date < df_modern.index.min() + relativedelta(months=36):
        continue
        
    train_df = df_modern.loc[:split_date]
    test_df = df_modern.loc[split_date + relativedelta(months=1):last_available_date]
    
    if test_df.empty:
        continue
    
    test_df_same_month = test_df[test_df.index.month == target_month]
    
    sample_analysis.append({
        'window': months,
        'train_end': split_date.strftime('%Y-%m'),
        'test_total': len(test_df),
        'test_same_month': len(test_df_same_month),
        'reduction_ratio': len(test_df_same_month) / len(test_df) if len(test_df) > 0 else 0,
        'same_month_dates': [d.strftime('%Y-%m') for d in test_df_same_month.index]
    })

print("\n窗口 | 训练截止 | 测试集总数 | 同月样本数 | 缩减比例 | 同月日期")
print("-----|----------|------------|------------|----------|----------")
for s in sample_analysis:
    print(f"-{s['window']:>2}月 | {s['train_end']} | {s['test_total']:>10} | {s['test_same_month']:>10} | {s['reduction_ratio']:>8.1%} | {', '.join(s['same_month_dates'])}")

avg_reduction = np.mean([s['reduction_ratio'] for s in sample_analysis])
min_samples = min([s['test_same_month'] for s in sample_analysis])
max_samples = max([s['test_same_month'] for s in sample_analysis])

print(f"\n关键指标：")
print(f"  平均缩减比例：{avg_reduction:.1%} (理论值应为 1/12 = 8.3%)")
print(f"  最少样本数：{min_samples} 个")
print(f"  最多样本数：{max_samples} 个")

print(f"\n✓ 风险1评估：")
if max_samples <= 2:
    print(f"  ⚠️ 高风险！样本量极少（最多{max_samples}个），RMSE统计显著性很低")
    print(f"  建议：需要更长的历史数据或考虑其他评估方法")
elif max_samples <= 5:
    print(f"  ⚠️ 中等风险。样本量较少（最多{max_samples}个），可能存在偶然性")
    print(f"  建议：结合置信区间或bootstrap方法增强可靠性")
else:
    print(f"  ✓ 风险较低。样本量充足（最多{max_samples}个）")

# 风险2：近期趋势敏感度
print("\n" + "="*80)
print("风险2：对近期趋势的敏感度")
print("="*80)

# 分析最近12个月的趋势
recent_12m = df_modern.tail(12)
recent_6m = df_modern.tail(6)
recent_3m = df_modern.tail(3)

print(f"\n最近12个月平均值：{recent_12m['Anomaly'].mean():.2f}")
print(f"最近6个月平均值：{recent_6m['Anomaly'].mean():.2f}")
print(f"最近3个月平均值：{recent_3m['Anomaly'].mean():.2f}")

# 计算趋势变化率
trend_12m = np.polyfit(range(12), recent_12m['Anomaly'].values, 1)[0]
trend_6m = np.polyfit(range(6), recent_6m['Anomaly'].values, 1)[0]

print(f"\n最近12个月趋势斜率：{trend_12m:.2f} (每月)")
print(f"最近6个月趋势斜率：{trend_6m:.2f} (每月)")

if abs(trend_6m) > abs(trend_12m) * 1.5:
    print(f"\n⚠️ 检测到近期趋势加速！")
    print(f"  6个月趋势是12个月趋势的 {abs(trend_6m/trend_12m):.1f} 倍")
else:
    print(f"\n✓ 近期趋势相对稳定")

# 测试：季节性方法 vs 全局方法对近期数据的敏感度
print("\n" + "="*80)
print("对比测试：两种方法对近期突变的反应")
print("="*80)

# 模拟场景：使用-24月窗口
months = 24
split_date = target_date - relativedelta(months=months)
train_df = df_modern.loc[:split_date]
test_df = df_modern.loc[split_date + relativedelta(months=1):last_available_date]

print(f"\n测试窗口：-{months}月")
print(f"训练截止：{split_date.strftime('%Y-%m')}")
print(f"测试期间：{test_df.index.min().strftime('%Y-%m')} 到 {test_df.index.max().strftime('%Y-%m')}")

# 分析训练集是否包含近期高温数据
train_recent = train_df.tail(12)
print(f"\n训练集最后12个月平均值：{train_recent['Anomaly'].mean():.2f}")
print(f"当前最新12个月平均值：{recent_12m['Anomaly'].mean():.2f}")
print(f"差异：{recent_12m['Anomaly'].mean() - train_recent['Anomaly'].mean():.2f}")

if recent_12m['Anomaly'].mean() - train_recent['Anomaly'].mean() > 10:
    print(f"\n⚠️ 训练集未包含最新的高温趋势！")
    print(f"  这可能导致预测偏低")
else:
    print(f"\n✓ 训练集基本反映了当前温度水平")

# 检查11月的历史数据是否有突变
nov_data = df_modern[df_modern.index.month == target_month].tail(5)
print(f"\n最近5个11月的数据：")
for date, value in nov_data['Anomaly'].items():
    print(f"  {date.strftime('%Y-%m')}: {value:.2f}")

nov_trend = np.polyfit(range(len(nov_data)), nov_data['Anomaly'].values, 1)[0]
print(f"\n11月数据趋势斜率：{nov_trend:.2f} (每年)")

print("\n" + "="*80)
print("✓ 风险2评估：")
print("="*80)

if abs(trend_6m) > 2 and abs(trend_6m) > abs(trend_12m) * 1.5:
    print("⚠️ 高风险！近期存在显著趋势加速")
    print("  季节性方法可能对跨季节突变反应不足")
    print("  建议：考虑添加近期趋势调整因子")
elif recent_12m['Anomaly'].mean() - train_recent['Anomaly'].mean() > 15:
    print("⚠️ 中等风险。训练集与当前状态差异较大")
    print("  建议：使用更短的训练窗口或添加趋势权重")
else:
    print("✓ 风险较低。训练集基本反映当前气候状态")

print("\n" + "="*80)
print("总结与建议")
print("="*80)

risk_score = 0
if max_samples <= 2:
    risk_score += 3
elif max_samples <= 5:
    risk_score += 1

if abs(trend_6m) > abs(trend_12m) * 1.5:
    risk_score += 2
elif recent_12m['Anomaly'].mean() - train_recent['Anomaly'].mean() > 15:
    risk_score += 1

print(f"\n综合风险评分：{risk_score}/5")
if risk_score >= 4:
    print("⚠️ 高风险！建议谨慎使用或改进方法")
elif risk_score >= 2:
    print("⚠️ 中等风险。方法可用但需注意局限性")
else:
    print("✓ 低风险。方法适用于当前数据情况")

print("\n改进建议：")
if max_samples <= 5:
    print("  1. 增加训练窗口数量，获取更多同月样本")
    print("  2. 使用bootstrap重采样增强统计可靠性")
if abs(trend_6m) > abs(trend_12m) * 1.5:
    print("  3. 添加近期趋势调整因子（权重0.1-0.2）")
    print("  4. 考虑混合策略：季节性权重 + 全局RMSE")
