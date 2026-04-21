# Analytic Continuation

这是一个面向数值实验的解析延拓工程骨架，目标是研究满足

`f(z) = g(z, f(z - 1))`

这类泛函方程的递推序列，重点覆盖变底数迭代幂次、固定底数迭代幂次、阶乘类递推以及用户自定义递推。

## 当前状态

首版骨架已经包含：

- 统一的递推序列抽象与四类序列实现
- 基底区间多项式解的表示与应变能计算
- 可运行的 M1 三次 Hermite 基线方法
- 可运行的 M2 五次 Hermite 方法
- Chebyshev 谱方法的首版主线实现
- M4-M5 的方法占位接口
- 基础评估工具、实验脚本与测试

## 目录

```text
analytic_continuation/
├── docs/
├── experiments/
├── results/
├── src/analytic_continuation/
│   ├── core/
│   ├── evaluation/
│   ├── methods/
│   └── utils/
└── tests/
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python experiments/exp_gamma_validation.py
```

如果希望在有标准解的验证实验中先自动调参，再用调好的 Chebyshev 配置求值：

```bash
python experiments/exp_gamma_validation.py --autotune-chebyshev
python experiments/exp_convergence.py --autotune-chebyshev
```

## 当前实现边界

当前已经完成三条求解链路：

- `HermiteCubicMethod`：M1 基线方法，解析优化左端一阶导数
- `HermiteQuinticMethod`：M2 方法，利用一阶/二阶导数约束并数值优化能量
- `ChebyshevMethod`：M3 系数空间谱方法，默认采用一组经过稳定性扫描回调的平衡参数
- `ChebyshevMethod.autotune(...)`：针对有 ground truth 的参考问题做离散网格调参

当前默认参数是一组偏稳健的折中配置，而不是直接使用扫描中的最高 degree 组合。对于 `FactorialType` 这类有标准解的问题，建议先用 `autotune` 扫一轮候选参数，再把选出的配置固定到实验脚本中。
