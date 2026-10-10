# nn 复合实现算子：沿 MaxPool2d 的调用顺序往下看

本文接在 [nn模块](nn模块.md)之后。已经学过的 AvgPool2d 帮我们认识了 Python 绑定、ATen 接口、Dispatcher 和 NPU wrapper。现在沿一次 `MaxPool2d` 调用继续往下走，重点观察中间新增的复合实现，以及它怎样再次调用一个 ATen 算子。

这是同目录源码汇编采用的本地源码根目录。正文所列行号来自实际读取的本地文件；本地缺失的生成文件使用真实生成器说明，不把推导出的代码当作已取得的构建产物。

本例主要沿普通 dense NPU 张量、eager 模式、`return_indices=False` 的前向路径。输入 `x` 已在 NPU 上。涉及 Autograd 的地方保留必要说明，反向 Engine 留给后续学习。

## Part1. 先了解整体结构

### 最上层：用户通过 return_indices 决定返回形式

MaxPool2d 可以只返回池化结果，也可以同时返回最大值位置。先明确这一点，后面 functional 为什么分两条路就有了原因。

```python
pool = nn.MaxPool2d(2, stride=2, return_indices=False)
y = pool(x)

pool_with_indices = nn.MaxPool2d(2, stride=2, return_indices=True)
y, indices = pool_with_indices(x)
```

这里真正决定返回形式的是 `return_indices`。执行 `y, indices = pool_with_indices(x)` 时，先调用右边，拿到 tuple，再由左边解包。函数不会根据左边有几个变量来选分支；写成 `result = pool_with_indices(x)`，拿到的仍然是 tuple。

`MaxPool2d` 继承`PT/torch/nn/modules/pooling.py:57` 的 [`_MaxPoolNd.__init__`](#src-maxpool-init)构造函数。默认 `return_indices=False`。

### 随后步骤：

我们的整体逻辑没有任何改变：调用forward方法，该方法再调用F.max_pool2d, 可以完全参考[nn模块中的Part1的第一、二步]()。但是F.max_pool2d的函数内容有一些不同，我们后续再说。

### 不同的改变
我们现在回到F.max_pool2d，他使用`return_indices`做了一个简单的选择，选择不同的包装好的C binding的python函数。示意图可以参考下面
```text
F.max_pool2d
  ├─ False → _max_pool2d → torch.max_pool2d
  └─ True  → max_pool2d_with_indices → torch._C._nn.max_pool2d_with_indices
```
这两个Python绑定函数是使用ymal生成的，同样可以参考[nn模块中的Part2]()。
```
torch.max_pool2d
```
他会从我在[nn模块中的Part2和Part3最初一部分]()一样，找到自己的aten算子，找到at::_ops对应算子句柄的call方法，定位Dispatcher::OperatorDef的那一条operatorentry，先提取有效 keyset，再执行 `lookup`，最后调用选中的 `KernelFunction`。整体都是没区别的。
| 名字 | 本次承担的作用 |
|---|---|
| `at::max_pool2d` | C++ 调用接口，把请求交给 Dispatcher |
| `aten::max_pool2d` | Dispatcher 查找这个算子使用的名称 |
| `at::native::max_pool2d` | 选中通用实现后实际执行的 C++ 函数 |

但是我们细看会发现，`max_pool2d` 没有独立注册的 NPU wrapper，为什么仍然有可调用的实现？这就是我们后续需要关注的重点了。至于`max_pool2d_with_indices`, 他有对应的 NPU wrapper，具体操作和前面nn模块的后续步骤一模一样。

## 区别一：F.max_pool2d 根据布尔参数选择具体的 Python 函数

接下来打开 `PT/torch/nn/functional.py:840`，见 [max_pool2d 的完整绑定赋值](#src-maxpool-boolean-binding)。

这里不是像 AvgPool2d 那样直接给 C binding 挂文档，而是调用 `boolean_dispatch`，得到一个新的 Python 可调用对象，并将它赋给 `max_pool2d`。用一个通俗的话来讲，这个函数会根据传入的`return_indices`确定我们需要把`max_pool2d`转发给哪个具体实现

赋值代码指定：

- 看哪个参数：`arg_name="return_indices"`。
- 位置参数在哪：`arg_index=6`，按从 0 开始的下标计算。
- 没传时用什么：`default=False`。
- 参数为真调用谁：`max_pool2d_with_indices`。
- 参数为假调用谁：`_max_pool2d`。

`boolean_dispatch` 的函数体在 `PT/torch/_jit_internal.py:612`，见[完整函数](#src-boolean-dispatch)。其中内部函数 `fn` 先读 kwargs，再按位置读 args，最后调用 `if_true` 或 `if_false`。

因此，本次 `False` 配置会把请求交给 `_max_pool2d`。这是一次 Python 参数分支选择；稍后才会进入按 dispatch keys 选 kernel 的 C++ Dispatcher。

`_max_pool2d` 位于 `PT/torch/nn/functional.py:814`，见[完整函数](#src-functional-maxpool)。它先处理 Tensor subclass 的 `__torch_function__` 接管情况；普通 Tensor 沿正常路径继续。stride 为 None 时会整理成空列表，然后调用 `torch.max_pool2d`。

若一开始选择 `return_indices=True`，则进入同文件 `:761` 的 [max_pool2d_with_indices](#src-functional-maxpool-indices)，最后调用 `torch._C._nn.max_pool2d_with_indices`。


## 区别二. CompositeImplicitAutograd 怎样为 max_pool2d 提供实现

标题有点难懂，我们从yaml讲起：

打开 `PT/aten/src/ATen/native/native_functions.yaml:3888` 的 [max_pool2d 完整声明](#src-maxpool-schema)，其中有：

```yaml
dispatch:
  CompositeImplicitAutograd: max_pool2d
  MPS: mps_max_pool2d
```

`CompositeImplicitAutograd: max_pool2d` 表示：以这个 key 注册通用实现，函数位于 `PT/aten/src/ATen/native/Pooling.cpp:135`，即 `at::native::max_pool2d`。MPS 可以另外提供专用实现；本文继续沿普通 NPU 输入的通用路径。

此处几个相近名字分工不同：

| 名字 | 本次承担的作用 |
|---|---|
| `at::max_pool2d` | C++ 调用接口，把请求交给 Dispatcher |
| `aten::max_pool2d` | Dispatcher 查找这个算子使用的名称 |
| `at::native::max_pool2d` | 选中通用实现后实际执行的 C++ 函数 |

所以后续的操作也很好理解了，注册其实和nn模块主链路一致，构建生成代码、加载时登记，以及 `Library::_impl → registerImpl → registerKernel` 的共同过程，见 [nn模块的注册链条](nn模块.md#nn-register-chain)。相关注册模板仍在 `PT/aten/src/ATen/templates/RegisterDispatchKey.cpp` 与 `RegisterDispatchDefinitions.ini`，见[完整模板](nn模块源码.md#src-backend-template)和[注册块](nn模块源码.md#src-dispatch-def-template)。

同样，对于m.impl的注册，我们的NPU wrapper现在是什么呢？`torchgen/dest/register_dispatch_key.py:489`生成了`wrapper_CompositeImplicitAutograd__max_pool2d`，这个wrapper本质是转发请求给`at::native::max_pool2d`

本例需要区别的是登记内容：AvgPool2d 的 NPU wrapper 挂在 PrivateUse1；这里讲的通用函数挂在 CompositeImplicitAutograd。`registerKernel` 会保存这个注册记录，再更新 `max_pool2d` 的有效调度表。

#### 认识CompositeImplicitAutograd这个dispatch key ：alias key 让一组运行时槽位重新选择实现

这次更新不能理解成“只给 CompositeImplicitAutograd 分配一个普通数组槽位”。它是 **alias key**，会影响一组运行时 keys,例如cuda privateuse1 xpu等。

具体代码在 `PT/aten/src/ATen/core/dispatch/OperatorEntry.cpp:493` 的 [updateDispatchTable_](nn模块源码.md#src-update-table)：它遍历 `getRuntimeDispatchKeySet(dispatch_key)`，逐个更新受影响的槽位。

继续进入 `PT/c10/core/DispatchKeySet.cpp:65` 的 [getRuntimeDispatchKeySet](nn模块源码.md#src-runtime-keys)，对应的真实代码节选是：

```cpp
case DispatchKey::CompositeImplicitAutograd:
  return math_dispatch_keyset;
```

因此，登记通用函数后，系统会重新计算这组运行时 keys 应当使用什么实现。**张量不需要增加一个 CompositeImplicitAutograd 标签；变化发生在算子的调度表里。** 例如PrivateUse1 对应的位置 → 通用复合 wrapper 的 KernelFunction

`math_dispatch_keyset` 在同文件 `:48` 开始定义，他对应了各种dispatchkey ：后端与 Autograd 并加入 NestedTensor 的特殊支持。本例只关注的其中与普通 NPU 后端及其 Autograd 调度相关的槽位。

### 第四步：通用实现怎样被选入有效表

把结果写入 `dispatchTable_[dispatch_ix]` 的共用过程，见 [nn模块 Part3](nn模块.md#nn-dispatch)；

for循环利用`getDispatchTableIndexForDispatchKey`找到`CompositeImplicitAutograd`对应的所有`dispatchkey`对应的数组下标`dispatch_ix`, 使用`computeDispatchTableEntry`计算`dispatcher`, `dispatch_key`对应的`KernelFunction`填入数组

我们发现，其实`computeDispatchTableEntry`不会因为循环由 CompositeImplicitAutograd 注册触发，就无条件填入复合函数。他调用了 `PT/aten/src/ATen/core/dispatch/OperatorEntry.cpp:352` 的 [computeDispatchTableEntryWithDebug](nn模块源码.md#src-compute-entry-debug)。它先检查直接注册的privateuse1等key对应的kernelfunction，再检查适用的 alias 注册的复合函数。

在隐式复合分支中，它确认当前 key 是否适用 CompositeImplicitAutograd，取得注册的通用函数，并检查相关后端实现等条件；满足条件时返回该实现，来源标签为 `math kernel`。

于是，“注册记录没有独立的 PrivateUse1 wrapper”和“运行时槽位没有函数可调用”是两件事。`max_pool2d` 的通用函数可以按这些规则成为有效入口。相关 Autograd 槽位也按自己的条件计算，不能机械地认为所有执行都先经过同一套功能层。

这段准备发生在调用之前。回到 Part3：`lookup` 此时读取已经填好的有效表，取得对应 `KernelFunction`，然后继续执行。

## 区别三. 执行 Pooling.cpp 中的 max_pool2d 函数


从 `KernelFunction::call` 经适配入口和函数对象，到真正 C++ 实现的过程，见 [nn模块的调用链条](nn模块.md#nn-kernel-call)。共用入口仍在 `PT/aten/src/ATen/core/boxing/KernelFunction_impl.h:145`，见 [KernelFunction::call](nn模块源码.md#src-kernel-call)。

本次选中的是通用实现，所以这条包装链的KernelFunction会最后进入 `at::native::max_pool2d`。

现在打开 `PT/aten/src/ATen/native/Pooling.cpp:135`，见[完整函数](#src-native-maxpool)。它处理 quantized、MKLDNN 等特殊情况后，普通 NPU Tensor 继续到 `:157–159`：

```cpp
auto output_and_indices = at::max_pool2d_with_indices(
    self, kernel_size, stride, padding, dilation, ceil_mode);
return std::get<0>(std::move(output_and_indices));
```

先执行第一句：把原来的 `self` 和池化配置交给 `at::max_pool2d_with_indices`。这里调用的是公开 ATen 接口，所以会重新进入 Dispatcher；它没有直接跳到某个 NPU 计算函数。

这就是本例复合实现的具体含义：`max_pool2d` 通过另一个已有算子完成工作，再整理返回值。选择 `max_pool2d_with_indices` 是函数体写明的动作，CompositeImplicitAutograd 负责让调用有机会选中这份通用函数。

`self` 仍是 NPU Tensor。执行主机上的 C++ 胶水函数，不等于把池化数据拷到 CPU 计算。

## 源码索引与阅读方式

下面收录本例新增的真实源码。Python 方法和函数按完整定义收录，YAML 与绑定赋值按完整条目收录；C++ native 函数及 op_api 文件保留真实内容。正文使用的短片段均可在这里或已有源码汇编中核对。

本次没有新增 NPU 运行实验。关于 PrivateUse1 注册的原运行证据来自学习资料目录中的 `nn/runtime_evidence.md`；它与当前本地源码快照有版本差异，不把原生成文件行号直接当成本地行号。

- [nn 复合实现算子：沿 MaxPool2d 的调用顺序往下看](#nn-复合实现算子沿-maxpool2d-的调用顺序往下看)
  - [Part1. 先了解整体结构](#part1-先了解整体结构)
    - [最上层：用户通过 return\_indices 决定返回形式](#最上层用户通过-return_indices-决定返回形式)
    - [随后步骤：](#随后步骤)
    - [不同的改变](#不同的改变)
  - [区别一：F.max\_pool2d 根据布尔参数选择具体的 Python 函数](#区别一fmax_pool2d-根据布尔参数选择具体的-python-函数)
  - [区别二. CompositeImplicitAutograd 怎样为 max\_pool2d 提供实现](#区别二-compositeimplicitautograd-怎样为-max_pool2d-提供实现)
      - [认识CompositeImplicitAutograd这个dispatch key ：alias key 让一组运行时槽位重新选择实现](#认识compositeimplicitautograd这个dispatch-key-alias-key-让一组运行时槽位重新选择实现)
    - [第四步：通用实现怎样被选入有效表](#第四步通用实现怎样被选入有效表)
  - [区别三. 执行 Pooling.cpp 中的 max\_pool2d 函数](#区别三-执行-poolingcpp-中的-max_pool2d-函数)
  - [源码索引与阅读方式](#源码索引与阅读方式)
  - [src-maxpool-init](#src-maxpool-init)
  - [src-maxpool-forward](#src-maxpool-forward)
  - [src-maxpool-boolean-binding](#src-maxpool-boolean-binding)
  - [src-boolean-dispatch](#src-boolean-dispatch)
  - [src-functional-maxpool](#src-functional-maxpool)
  - [src-functional-maxpool-indices](#src-functional-maxpool-indices)
  - [src-maxpool-schema](#src-maxpool-schema)
  - [src-maxpool-indices-schema](#src-maxpool-indices-schema)
  - [src-maxpool-binding-category](#src-maxpool-binding-category)
  - [src-maxpool-binding-output](#src-maxpool-binding-output)
  - [src-maxpool-python-template](#src-maxpool-python-template)
  - [src-native-maxpool](#src-native-maxpool)
  - [src-maxpool-plugin-config](#src-maxpool-plugin-config)
  - [src-maxpool-opapi](#src-maxpool-opapi)

## src-maxpool-init

**_MaxPoolNd.__init__：完整方法**

````python
# 来源：/Users/yuanchengyao/graduation-project/pytorch/torch/nn/modules/pooling.py
# 起始行：55

    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: _size_any_t | None = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
````

## src-maxpool-forward

**MaxPool2d.forward：完整方法**

````python
# 来源：/Users/yuanchengyao/graduation-project/pytorch/torch/nn/modules/pooling.py
# 起始行：222

    def forward(self, input: Tensor):
        """Runs the forward pass."""
        return F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
````

## src-maxpool-boolean-binding

**functional.max_pool2d：完整绑定赋值**

````python
# 来源：/Users/yuanchengyao/graduation-project/pytorch/torch/nn/functional.py
# 起始行：840

max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool2d_with_indices,
    if_false=_max_pool2d,
    module_name=__name__,
    func_name="max_pool2d",
)
````

## src-boolean-dispatch

**boolean_dispatch：完整函数**

````python
# 来源：/Users/yuanchengyao/graduation-project/pytorch/torch/_jit_internal.py
# 起始行：612

def boolean_dispatch(
    arg_name,
    arg_index,
    default,
    if_true,
    if_false,
    module_name,
    func_name,
):
    """
    Dispatches to either of 2 script functions based on a boolean argument.
    In TorchScript, the boolean argument must be constant so that the correct
    function to use can be determined at compile time.
    """

    def fn(*args, **kwargs):
        dispatch_flag = default
        if arg_name in kwargs:
            dispatch_flag = kwargs[arg_name]
        elif arg_index < len(args):
            dispatch_flag = args[arg_index]

        if dispatch_flag:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)

    if if_true.__doc__ is None and if_false.__doc__ is not None:
        doc = if_false.__doc__
        if_true.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is not None:
        doc = if_true.__doc__
        if_false.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is None:
        # neither function has a docstring
        doc = None
    else:
        raise RuntimeError("only one function can have a docstring")
    fn.__doc__ = doc

    if module_name is not None:
        fn.__module__ = module_name
    if func_name is not None:
        fn.__name__ = func_name

    boolean_dispatched[fn] = {
        "if_true": if_true,
        "if_false": if_false,
        "index": arg_index,
        "default": default,
        "arg_name": arg_name,
    }
    return fn
````

## src-functional-maxpool

**_max_pool2d：完整函数**

````python
# 来源：/Users/yuanchengyao/graduation-project/pytorch/torch/nn/functional.py
# 起始行：814

def _max_pool2d(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,  # noqa: UP045
    padding: BroadcastingList2[int] = 0,
    dilation: BroadcastingList2[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool2d,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(list[int], [])
    return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
````

## src-functional-maxpool-indices

**max_pool2d_with_indices：完整函数**

````python
# 来源：/Users/yuanchengyao/graduation-project/pytorch/torch/nn/functional.py
# 起始行：761

def max_pool2d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,  # noqa: UP045
    padding: BroadcastingList2[int] = 0,
    dilation: BroadcastingList2[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""
    max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

    Applies a 2D max pooling over an input signal composed of several input
    planes.

    .. note::
        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from
        what seen in :class:`~torch.nn.MaxPool2d`, and will change in a future release.

    See :class:`~torch.nn.MaxPool2d` for details.

    Args:
        input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`, minibatch dim optional.
        kernel_size: size of the pooling region. Can be a single number or a
            tuple `(kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
            tuple `(sH, sW)`. Default: :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.functional.max_unpool2d` later
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool2d_with_indices,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(list[int], [])
    return torch._C._nn.max_pool2d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )
````

## src-maxpool-schema

**max_pool2d：完整 YAML 条目**

````yaml
# 来源：/Users/yuanchengyao/graduation-project/pytorch/aten/src/ATen/native/native_functions.yaml
# 起始行：3888

- func: max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
  dispatch:
    CompositeImplicitAutograd: max_pool2d
    MPS: mps_max_pool2d
````

## src-maxpool-indices-schema

**max_pool2d_with_indices：完整 YAML 条目**

````yaml
# 来源：/Users/yuanchengyao/graduation-project/pytorch/aten/src/ATen/native/native_functions.yaml
# 起始行：12631

- func: max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
  python_module: nn
  structured_delegate: max_pool2d_with_indices.out
  tags: core
````

## src-maxpool-binding-category

**torch 与 nn 绑定分类：两个完整函数**

````python
# 来源：/Users/yuanchengyao/graduation-project/pytorch/tools/autograd/gen_python_functions.py
# 起始行：236

def is_py_torch_function(f: NativeFunction) -> bool:
    return f.python_module is None and Variant.function in f.variants


def is_py_nn_function(f: NativeFunction) -> bool:
    return f.python_module == "nn"
````

## src-maxpool-binding-output

**绑定生成入口：真实源码片段**

这是 gen 函数中的相关片段，完整解析和单个绑定生成机制见已有 method_impl 源码。

````python
# 来源：/Users/yuanchengyao/graduation-project/pytorch/tools/autograd/gen_python_functions.py
# 起始行：298

    #       torch/csrc/autograd/python_torch_functions_manual.cpp
    functions = load_signatures(native_functions, deprecated_yaml_path, method=False)
    create_python_bindings_sharded(
        fm,
        functions,
        is_py_torch_function,
        "torch",
        "python_torch_functions.cpp",
        method=False,
        num_shards=3,
        symint=symint,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_nn_function,
        "torch.nn",
        "python_nn_functions.cpp",
        method=False,
        symint=symint,
    )
````

## src-maxpool-python-template

**torch Python 绑定：完整模板文件**

模板中的 $ 占位符是生成模板原文；本地未取得 THPVariable_max_pool2d 的生成函数体。

````cpp
// 来源：/Users/yuanchengyao/graduation-project/pytorch/tools/autograd/templates/python_torch_functions.cpp
// 起始行：1

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

// Python bindings for torch.* functions implemented through ATen.
//
// The functions are bound as static methods on a class
// torch._C._VariableFunctions which is also aliased as Variable._torch
// and also copied into 'torch' module.

#include <Python.h>

// Undefine the copysign macro so that at::copysign works as intended with MSVC
// https://github.com/python/cpython/blob/c60394c7fc9cc09b16e9675a3eeb5844b6d8523f/PC/pyconfig.h#L196
#ifdef _MSC_VER
#undef copysign
#endif // _MSC_VER

#include "torch/csrc/autograd/python_torch_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/Dtype.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/out_types.h"
#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/tensor_layouts.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/jit/frontend/tracer.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/utils/device_lazy_init.h"
#include "torch/csrc/autograd/generated/python_return_types.h"

#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
$ops_headers
#endif

#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <utility>

using at::Tensor;
using at::Device;
using at::Layout;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::Generator;
using at::TensorList;
using at::ArrayRef;

using torch::utils::check_out_type_matches;
using namespace torch::autograd::utils;

// NOTE: See [Sharded File] comment in VariableType

namespace torch::autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef torch_functions_shard[] = {
  ${py_method_defs}
};

void gatherTorchFunctions${shard_id}(std::vector<PyMethodDef> &torch_functions) {
  constexpr size_t num_functions = sizeof(torch_functions_shard) / sizeof(torch_functions_shard[0]);
  torch_functions.insert(
    torch_functions.end(),
    torch_functions_shard,
    torch_functions_shard + num_functions);
}

// generated methods start here

${py_methods}

} // namespace torch::autograd
````

## src-native-maxpool

**at::native::max_pool2d：完整函数**

````cpp
// 来源：/Users/yuanchengyao/graduation-project/pytorch/aten/src/ATen/native/Pooling.cpp
// 起始行：135

Tensor max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (self.is_quantized()) {
    return at::quantized_max_pool2d(self, kernel_size, stride, padding,
                                    dilation, ceil_mode);
  }
  if (self.is_mkldnn()) {
    return at::mkldnn_max_pool2d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
#if defined(C10_MOBILE)
  if(xnnpack::use_max_pool2d(self, kernel_size, padding, stride,
                             dilation, ceil_mode)) {
    return xnnpack::max_pool2d(
        self, kernel_size, padding, stride, dilation, ceil_mode);
  }
#endif
  auto output_and_indices = at::max_pool2d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
  return std::get<0>(std::move(output_and_indices));
}
````

## src-maxpool-plugin-config

**max_pool2d_with_indices：完整 op-plugin 配置条目**

````yaml
# 来源：/Users/yuanchengyao/graduation-project/torchnpu/pytorch/third_party/op-plugin/op_plugin/config/op_plugin_functions.yaml
# 起始行：3444

  - func: max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
    acl_op: all_version
    op_api: all_version
````

## src-maxpool-opapi

**MaxPool2dWithIndicesKernelNpuOpApi.cpp：完整文件**

````cpp
// 来源：/Users/yuanchengyao/graduation-project/torchnpu/pytorch/third_party/op-plugin/op_plugin/ops/opapi/MaxPool2dWithIndicesKernelNpuOpApi.cpp
// 起始行：1

// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/native/Pool.h>

#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void max_pool2d_with_indices_parameter_check(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple of two ints",
      OPS_ERROR(ErrCode::PARAM))
  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
      OPS_ERROR(ErrCode::PARAM))
  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple of two ints",
      OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of two ints",
      OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      (self.ndimension() == 3 || self.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input",
      OPS_ERROR(ErrCode::PARAM));
}

std::tuple<at::Tensor, at::Tensor> exec_max_pool2d_with_indices(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  const int64_t DIM_SIZE = 4;
  max_pool2d_with_indices_parameter_check(self, kernel_size, stride, padding, dilation);
#if !VERSION_BETWEEN(V2R13, VERSION_NEWEST)
  at::DimnameList maybe_names = self.has_names() ? self.names() : at::DimnameList{};
#else
  at::DimnameList maybe_names{};
#endif

  const int k_H = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
  const int k_W = kernel_size.size() == 1 ? k_H : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
  c10::SmallVector<int64_t, SIZE> ksizes = {k_H, k_W};
  at::IntArrayRef kernel_sizes = at::IntArrayRef(ksizes);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  const int d_H = stride.empty() ? k_H : at::native::safe_downcast<int, int64_t>(stride[0]);
  const int d_W = stride.empty() ? k_W : stride.size() == 1 ? d_H : at::native::safe_downcast<int, int64_t>(stride[1]);
  c10::SmallVector<int64_t, SIZE> stride_size = {d_H, d_W};
  at::IntArrayRef strides = at::IntArrayRef(stride_size);

  const int pad_H = at::native::safe_downcast<int, int64_t>(padding[0]);
  const int pad_W = padding.size() == 1 ? pad_H : at::native::safe_downcast<int, int64_t>(padding[1]);
  c10::SmallVector<int64_t, SIZE> padding_size = {pad_H, pad_W};
  at::IntArrayRef paddings = at::IntArrayRef(padding_size);

  const int dilation_H = at::native::safe_downcast<int, int64_t>(dilation[0]);
  const int dilation_W = dilation.size() == 1 ? dilation_H : at::native::safe_downcast<int, int64_t>(dilation[1]);
  c10::SmallVector<int64_t, SIZE> dilation_size = {dilation_H, dilation_W};
  at::IntArrayRef dilations = at::IntArrayRef(dilation_size);

  const int64_t n_batch = self.ndimension() == 4 ? self.size(-4) : 1;
  const int64_t n_input_plane = self.size(-3);
  const int64_t input_height = self.size(-2);
  const int64_t input_width = self.size(-1);

  const int64_t output_height =
      at::native::pooling_output_shape<int64_t>(input_height, k_H, pad_H, d_H, dilation_H, ceil_mode);
  const int64_t output_width =
      at::native::pooling_output_shape<int64_t>(input_width, k_W, pad_W, d_W, dilation_W, ceil_mode);

  at::native::pool2d_shape_check(
      self,
      k_H,
      k_W,
      d_H,
      d_W,
      pad_H,
      pad_W,
      dilation_H,
      dilation_W,
      n_input_plane,
      input_height,
      input_width,
      output_height,
      output_width,
      self.suggest_memory_format());

  c10::SmallVector<int64_t, SIZE> output_size = self.ndimension() == 4
      ? c10::SmallVector<int64_t, SIZE>({n_batch, n_input_plane, output_height, output_width})
      : c10::SmallVector<int64_t, SIZE>({n_input_plane, output_height, output_width});

  const int64_t BLOCKSIZE = 16;
  int64_t mask_H = kernel_sizes[0] * kernel_sizes[1];
  int64_t mask_W = (op_infer::CeilDiv(output_height * output_width, BLOCKSIZE) + 1);
  c10::SmallVector<int64_t, SIZE> indices_size;
  at::Tensor indices;

  at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, self.options());
  static auto maxpool2d_sc = at_npu::native::env::CheckCompatibleImpl();
  if (!c10_npu::IsAclnnOnly() && !maxpool2d_sc) {
    // 5HD format return the indices with mask, dtype int8,size should be muls 16*2
    indices_size = self.ndimension() == DIM_SIZE
        ? c10::SmallVector<int64_t, SIZE>({n_batch, n_input_plane, mask_H, mask_W * 32})
        : c10::SmallVector<int64_t, SIZE>({n_input_plane, mask_H, mask_W * 32});
    indices = npu_preparation::apply_tensor_without_format(indices_size, self.options().dtype(at::kChar));
    EXEC_NPU_CMD(aclnnMaxPool2dWithMask, self, kernel_sizes, strides, paddings, dilations, ceil_mode, output, indices);
  } else {
    indices_size = output_size;
    indices = npu_preparation::apply_tensor_without_format(indices_size, self.options().dtype(at::kInt));
    EXEC_NPU_CMD(
        aclnnMaxPool2dWithIndices, self, kernel_sizes, strides, paddings, dilations, ceil_mode, output, indices);
  }
  at::namedinference::propagate_names_if_nonempty(output, maybe_names);
  at::namedinference::propagate_names_if_nonempty(indices, maybe_names);
  return std::tuple<at::Tensor, at::Tensor>(output, indices);
}

std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  static auto maxpool2d_sc = at_npu::native::env::CheckCompatibleImpl();
  if (!c10_npu::IsAclnnOnly() && !maxpool2d_sc) {
    DO_COMPATIBILITY(
        aclnnMaxPool2dWithMask,
        acl_op::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode));
  } else {
    DO_COMPATIBILITY(
        aclnnMaxPool2dWithIndices,
        acl_op::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode));
  }

  return op_api::exec_max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<at::Tensor&, at::Tensor&> max_pool2d_with_indices_out(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    at::Tensor& output,
    at::Tensor& indices) {
  static auto maxpool2d_sc = at_npu::native::env::CheckCompatibleImpl();
  if (!c10_npu::IsAclnnOnly() && !maxpool2d_sc) {
    DO_COMPATIBILITY(
        aclnnMaxPool2dWithMask,
        acl_op::max_pool2d_with_indices_out(self, kernel_size, stride, padding, dilation, ceil_mode, output, indices));
    // execute 5HD format
    EXEC_NPU_CMD(aclnnMaxPool2dWithMask, self, kernel_size, stride, padding, dilation, ceil_mode, output, indices);
  } else {
    DO_COMPATIBILITY(
        aclnnMaxPool2dWithIndices,
        acl_op::max_pool2d_with_indices_out(self, kernel_size, stride, padding, dilation, ceil_mode, output, indices));
    EXEC_NPU_CMD(aclnnMaxPool2dWithIndices, self, kernel_size, stride, padding, dilation, ceil_mode, output, indices);
  }
  return std::tuple<at::Tensor&, at::Tensor&>(output, indices);
}
} // namespace op_api
````
