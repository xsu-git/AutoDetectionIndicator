#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/9 11:16     Xsu         1.0         None
'''

# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化后的主执行文件
实现更灵活的信号生成和回测
"""
from process.loader import load_process_data, load_peak_data
from process.feature_detection import FeatureExtractorLoader
from utils.data_preprocess_util import feather_to_csv
from process.model_train import MachineLearnTrain as EnhancedSignalGenerator
from process.backtester import EnhancedBacktester
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')


class OptimizedTradingSystem:
    """优化后的交易系统"""

    def __init__(self):
        self.signal_generator = None
        self.backtester = None
        self.features_df = None
        self.df_tech = None

    def run_optimized_strategy(self, data_file: str, peak_file: str):
        """执行优化策略"""

        print("=" * 80)
        print("🚀 启动优化后的量化交易系统")
        print("=" * 80)

        # 1. 数据加载
        print("\n📊 加载数据...")
        self.df_tech = load_process_data(data_file)
        peak_tech = load_peak_data(peak_file)

        # 2. 特征提取
        print("\n🔧 提取特征...")
        self.features_df = FeatureExtractorLoader(
            self.df_tech, peak_tech
        ).extract_all_features(exclude=["PeakStructureFeature"])

        # 3. 准备训练数据
        target_cols = [col for col in self.features_df.columns
                       if col.startswith(('future_', 'significant_', 'direction_'))]
        feature_cols = [col for col in self.features_df.columns
                        if col not in target_cols]

        X = self.features_df[feature_cols].copy()
        y_dict = {col: self.features_df[col].copy() for col in target_cols}

        print(f"\n📊 数据维度:")
        print(f"  - 特征矩阵: {X.shape}")
        print(f"  - 目标变量: {len(y_dict)}")
        print(f"  - 特征数量: {len(feature_cols)}")

        # 4. 训练增强模型
        print("\n🤖 训练增强模型...")
        self.signal_generator = self._train_enhanced_models(X, y_dict, feature_cols)

        # 5. 生成交易信号
        print("\n📡 生成交易信号...")
        signals_df = self.signal_generator.generate_trading_signals(
            self.df_tech, self.features_df
        )

        # 6. 信号统计
        self._analyze_signals(signals_df)

        # 7. 执行回测
        print("\n📈 执行增强回测...")
        self.backtester = EnhancedBacktester(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005,
            max_positions=5
        )

        backtest_results = self.backtester.backtest(signals_df, self.df_tech)

        # 8. 输出结果
        self._print_results(backtest_results)

        # 9. 可视化
        self._visualize_results(backtest_results, signals_df)

        return backtest_results

    def _train_enhanced_models(self, X, y_dict, feature_cols):
        """训练增强模型"""

        generator = EnhancedSignalGenerator()

        # 数据预处理
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X.fillna(0)),
            columns=X.columns,
            index=X.index
        )
        generator.scalers['features'] = scaler
        generator.feature_cols = feature_cols

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)

        # 训练回归模型（预测收益率）
        for target in ['future_return_3', 'future_return_5']:
            if target in y_dict:
                y_clean = y_dict[target].fillna(0)

                # 使用梯度提升回归
                model = GradientBoostingRegressor(
                    n_estimators=150,  # 减少过拟合
                    learning_rate=0.05,  # 更保守的学习率
                    max_depth=5,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    subsample=0.7,
                    random_state=42
                )

                # 交叉验证
                cv_scores = cross_val_score(
                    model, X_scaled, y_clean,
                    cv=tscv, scoring='neg_mean_squared_error'
                )
                print(f"  {target} - CV MSE: {-cv_scores.mean():.6f}")

                model.fit(X_scaled, y_clean)
                generator.models[target] = model

        # 训练分类模型（预测方向）
        for target in ['direction_3', 'direction_5', 'significant_move_3', 'significant_move_5']:
            if target in y_dict:
                y_clean = y_dict[target].fillna(0)

                # 使用随机森林
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_split=15,
                    min_samples_leaf=8,
                    max_features='sqrt',
                    random_state=42,
                    class_weight='balanced'
                )

                # 交叉验证
                cv_scores = cross_val_score(
                    model, X_scaled, y_clean,
                    cv=tscv, scoring='accuracy'
                )
                print(f"  {target} - CV Accuracy: {cv_scores.mean():.4f}")

                model.fit(X_scaled, y_clean)
                generator.models[target] = model

        generator.is_trained = True
        return generator

    def _analyze_signals(self, signals_df):
        """分析信号分布"""
        print("\n📊 信号分布分析:")

        signal_counts = signals_df['trading_signal'].value_counts()

        for signal, count in signal_counts.items():
            if signal != 'NO_SIGNAL':
                percentage = count / len(signals_df) * 100
                print(f"  {signal}: {count} ({percentage:.2f}%)")

        # 统计买卖信号
        buy_signals = signals_df[signals_df['trading_signal'].str.contains('BUY', na=False)]
        sell_signals = signals_df[signals_df['trading_signal'].str.contains('SELL', na=False)]

        print(f"\n  总买入信号: {len(buy_signals)}")
        print(f"  总卖出信号: {len(sell_signals)}")
        print(f"  信号覆盖率: {(len(buy_signals) + len(sell_signals)) / len(signals_df) * 100:.2f}%")

    def _print_results(self, results):
        """打印回测结果"""
        metrics = results['metrics']

        print("\n" + "=" * 80)
        print("📊 回测结果汇总")
        print("=" * 80)

        print("\n💰 收益指标:")
        print(f"  总收益率: {metrics['total_return']:.2%}")
        print(f"  年化收益率: {metrics.get('annualized_return', 0):.2%}")
        print(f"  利润因子: {metrics.get('profit_factor', 0):.2f}")

        print("\n📈 交易统计:")
        print(f"  总交易次数: {metrics['total_trades']}")
        print(f"  胜率: {metrics['win_rate']:.2%}")
        print(f"  平均盈利: ${metrics.get('avg_win', 0):.2f}")
        print(f"  平均亏损: ${metrics.get('avg_loss', 0):.2f}")
        print(f"  平均持仓时间: {metrics.get('avg_holding_period', 0):.1f} 小时")

        print("\n⚠️ 风险指标:")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  Calmar比率: {metrics.get('calmar_ratio', 0):.3f}")

        print("\n🎯 信号类型统计:")
        for signal_type, stats in metrics.get('signal_statistics', {}).items():
            print(f"\n  {signal_type}:")
            print(f"    交易次数: {stats['count']}")
            print(f"    胜率: {stats['win_rate']:.2%}")
            print(f"    平均收益: {stats['avg_return']:.4%}")
            print(f"    总盈亏: ${stats['total_pnl']:.2f}")

    def _visualize_results(self, results, signals_df):
        """可视化结果"""

        equity_df = pd.DataFrame(results['equity_curve'])

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # 1. 权益曲线
        ax1 = axes[0, 0]
        ax1.plot(equity_df.index, equity_df['total_value'], 'b-', linewidth=2)
        ax1.fill_between(equity_df.index, self.backtester.initial_capital,
                         equity_df['total_value'], alpha=0.3)
        ax1.set_title('权益曲线', fontsize=12, fontweight='bold')
        ax1.set_ylabel('总资产 ($)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.backtester.initial_capital, color='r',
                    linestyle='--', alpha=0.5, label='初始资金')

        # 2. 回撤曲线
        ax2 = axes[0, 1]
        cummax = equity_df['total_value'].cummax()
        drawdown = (equity_df['total_value'] - cummax) / cummax * 100
        ax2.fill_between(equity_df.index, 0, drawdown, color='red', alpha=0.5)
        ax2.set_title('回撤曲线', fontsize=12, fontweight='bold')
        ax2.set_ylabel('回撤 (%)')
        ax2.grid(True, alpha=0.3)

        # 3. 持仓数量
        ax3 = axes[1, 0]
        ax3.plot(equity_df.index, equity_df['num_positions'], 'g-', linewidth=1)
        ax3.fill_between(equity_df.index, 0, equity_df['num_positions'],
                         color='green', alpha=0.3)
        ax3.set_title('持仓数量', fontsize=12, fontweight='bold')
        ax3.set_ylabel('持仓数')
        ax3.grid(True, alpha=0.3)

        # 4. 信号分布
        ax4 = axes[1, 1]
        signal_types = ['STRONG_BUY', 'MODERATE_BUY', 'WEAK_BUY',
                        'STRONG_SELL', 'MODERATE_SELL', 'WEAK_SELL']
        signal_counts = [len(signals_df[signals_df['trading_signal'] == s])
                         for s in signal_types]
        colors = ['darkgreen', 'green', 'lightgreen',
                  'darkred', 'red', 'lightcoral']
        ax4.bar(range(len(signal_types)), signal_counts, color=colors)
        ax4.set_xticks(range(len(signal_types)))
        ax4.set_xticklabels([s.replace('_', '\n') for s in signal_types],
                            rotation=45, ha='right')
        ax4.set_title('信号类型分布', fontsize=12, fontweight='bold')
        ax4.set_ylabel('信号数量')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. 收益分布
        ax5 = axes[2, 0]
        trades = [t for t in results['trades'] if 'return' in t]
        if trades:
            returns = [t['return'] * 100 for t in trades]
            ax5.hist(returns, bins=30, color='blue', alpha=0.6, edgecolor='black')
            ax5.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax5.set_title('收益率分布', fontsize=12, fontweight='bold')
            ax5.set_xlabel('收益率 (%)')
            ax5.set_ylabel('频次')
            ax5.grid(True, alpha=0.3)

        # 6. 累计收益对比
        ax6 = axes[2, 1]
        cumulative_return = (equity_df['total_value'] / self.backtester.initial_capital - 1) * 100
        benchmark_return = (self.df_tech['close'] / self.df_tech['close'].iloc[0] - 1) * 100

        ax6.plot(equity_df.index, cumulative_return, 'b-', linewidth=2, label='策略')
        ax6.plot(self.df_tech.index[:len(benchmark_return)],
                 benchmark_return[:len(equity_df)], 'gray',
                 linewidth=1, alpha=0.7, label='买入持有')
        ax6.set_title('累计收益对比', fontsize=12, fontweight='bold')
        ax6.set_ylabel('累计收益率 (%)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 执行优化策略
    system = OptimizedTradingSystem()

    data_file = "origin_data.csv"
    peak_file = "peak_report_1756977444.json"

    results = system.run_optimized_strategy(data_file, peak_file)