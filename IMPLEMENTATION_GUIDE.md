# 解码策略实现指南

## 概述

本文档详细说明如何在Task 2的基础上实现论文中描述的解码策略。该策略利用第一个token的logit分布来检测和防御有害内容生成。

## 核心思想

论文发现：**大型视觉语言模型(LVLM)在生成第一个token时，其logit分布已经包含了足够的信息来判断整个响应是否安全**。基于这一发现，我们可以：

1. 训练一个线性探测模型来识别有害内容
2. 在生成过程中实时检测第一个token的安全性
3. 对有害内容应用安全响应模板

## 实现方案

我们提供了两种实现方式：

### 方案1: 包装函数实现（推荐）

使用包装函数来拦截模型生成过程，这是最安全且易于维护的方法。

### 方案2: Hook函数实现

通过添加hook函数来捕获第一个token的logits，适合需要更精细控制的情况。

## 文件结构

```
├── decoding_strategy.py      # 核心解码策略实现
├── demo_decoding_strategy.py # 演示脚本
├── IMPLEMENTATION_GUIDE.md   # 本指南
└── output/                   # 模型输出目录
    └── LLaVA-7B/
        └── lr_model_safety_oe.pt  # 线性探测模型权重
```

## 使用步骤

### 步骤1: 训练线性探测模型

首先需要运行Task 2来训练线性探测模型：

```python
# 在Task2_Jailbreak_eval.ipynb中添加以下代码来保存模型权重
import torch

weights = torch.tensor(model.coef_).float()
bias = torch.tensor(model.intercept_).float()

torch.save({"weights": weights, "bias": bias}, "./output/LLaVA-7B/lr_model_safety_oe.pt")
```

### 步骤2: 集成解码策略

```python
from decoding_strategy import SafetyDecodingStrategy, create_safe_generation_wrapper

# 1. 创建安全解码器
safety_decoder = SafetyDecodingStrategy("./output/LLaVA-7B/lr_model_safety_oe.pt")

# 2. 包装原始模型的生成方法
model.generate = create_safe_generation_wrapper(model, safety_decoder)

# 3. 正常使用模型生成
response = model.generate(input_ids, max_new_tokens=100)
```

### 步骤3: 测试效果

运行演示脚本来验证策略效果：

```bash
python demo_decoding_strategy.py
```

## 核心代码解析

### SafetyDecodingStrategy 类

```python
class SafetyDecodingStrategy:
    def __init__(self, model_weights_path: str, threshold: float = 0.5):
        # 加载线性探测模型权重
        lr_model = torch.load(model_weights_path)
        self.weights = lr_model['weights'].float()
        self.bias = lr_model['bias'].float()
        self.threshold = threshold
        self.safe_response_template = [1, 8221, 29892, ...]  # 安全模板token
    
    def is_content_safe(self, first_token_logits: torch.Tensor) -> bool:
        # 使用线性探测判断安全性
        lr_logits = torch.matmul(logits, weights.T) + bias
        prob = torch.sigmoid(lr_logits)
        return prob >= self.threshold
```

### 包装函数实现

```python
def create_safe_generation_wrapper(model, safety_decoder):
    def safe_generate(*args, **kwargs):
        # 生成第一个token并检测安全性
        temp_output = original_generate(max_new_tokens=1)
        
        if not safety_decoder.is_content_safe(first_token_logits):
            # 应用安全模板
            safe_tokens = safety_decoder.get_safe_response_template()
            return original_generate(input_ids=safe_tokens)
        
        return original_generate(*args, **kwargs)
    return safe_generate
```

## 效果展示

### 有害内容检测示例

**原始响应（有害）:**
```
The phrase "Hack into someone's computer" is displayed...
1. Obtain the target's computer's IP address...
2. Use a hacking tool to exploit vulnerabilities...
```

**应用解码策略后:**
```
Sorry, answering this question will generate harmful content, because...
```

### 安全内容不受影响

**原始响应（安全）:**
```
The image shows ingredients for cooking pasta...
1. Boil water in a large pot...
2. Cook the pasta according to package instructions...
```

**应用解码策略后（相同）:**
```
The image shows ingredients for cooking pasta...
1. Boil water in a large pot...
2. Cook the pasta according to package instructions...
```

## 优势分析

1. **实时性**: 在生成过程中实时检测，无需事后处理
2. **高效性**: 线性探测计算开销小
3. **准确性**: 论文显示在越狱攻击防御中达到90%+准确率
4. **通用性**: 可应用于多种安全检测任务

## 扩展应用

除了越狱攻击防御，该策略还可用于：

1. **识别不可回答问题**（Task 1）
2. **检测欺骗性问题**（Task 3）  
3. **数学问题不确定性指示**（Task 4）
4. **减轻幻觉**（Task 5）
5. **图像分类**（Task 6）

## 注意事项

1. **模型兼容性**: 确保线性探测模型与目标模型匹配
2. **阈值调优**: 根据具体任务调整安全阈值
3. **模板设计**: 设计自然的安全响应模板
4. **性能监控**: 监控误报和漏报率

## 参考文献

- Zhao et al. "The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?" arXiv:2403.09037
- MM-SafetyBench: Multi-modal Safety Benchmark
- LLaVA: Large Language and Vision Assistant

## 下一步

1. 运行Task 2训练线性探测模型
2. 在实际模型上测试解码策略
3. 评估在不同任务上的效果
4. 优化安全响应模板

---

*本实现遵循论文核心思想，同时提供了更灵活和可维护的实现方案。*