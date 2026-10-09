## 先讲一个简单的二维平均池化

本文用 `PT` 表示 PyTorch 源码根目录，`TN` 表示 torch_npu 仓库根目录。正文沿请求的传递和输出的返回讲调用链；C++ 语法与包装实现细节集中在[附录](nn模块附录.md)。完整源码保存在[源码汇编](nn模块源码.md)，对应代码块注明实际路径与行号。
- TODO
  - 深入到最后的细节
    - wrapper 如何进入具体的 op_plugin 实现；
    - 输出分配、workspace 和执行接口如何衔接；
    - 前向怎样建立反向关系，backward 又怎样走到 NPU。
  - 总览图很麻烦，要全改掉
  - 分支太多，容易迷失方向

> 文中的简化调用图与演示代码用于解释关系；[源码汇编](nn模块源码.md)中收录的是从本地提取的真实源码。构建生成文件若不存在，会明确说明，并提供实际生成器或完整模板。

- 原记录的测试入口为 `TestPoolingNNDeviceNPU::test_AvgPool2d_npu`，见[来源核对记录](nn模块源码.md#src-test-origin)。当前本地未找到该名称的完整定义。
- 目的：测试`nn.AvgPool2d` 前向 + 反传

整体的调用链如下：

```text
test_AvgPool2d_npu()
│
├─ nn.AvgPool2d.forward → F.avg_pool2d          Python API
├─ torch._C._nn.avg_pool2d                      yaml 生成的 C wrapper（THPVariable_avg_pool2d）
├─ at::avg_pool2d → _ops::avg_pool2d::call      findSchemaOrThrow("aten::avg_pool2d") 查表
├─ aten::avg_pool2d                             Operator Schema（yaml 声明,RegisterSchema 注册）
├─ Dispatcher                                   DispatchKeySet = {AutogradPrivateUse1, PrivateUse1, ...}
├─ (需要 grad 时) Autograd[alias] 生成的透传 kernel
├─ PrivateUse1 槽位 → wrapper_NPU__avg_pool2d   torch_npu 生成 wrapper
├─ op_plugin::avg_pool2d (op_api)               算子实现（算输出尺寸、造输出 tensor）
├─ EXEC_NPU_CMD(aclnnAvgPool2d, ...)            两段式：GetWorkspaceSize → 执行
├─ libopapi.so（闭源）→ libascendcl → 驱动      Open-source tracing boundary
└─ NPU                                          结果写回 NPU 内存中的输出 Tensor
```

### Part1. torch 侧对算子的python包装

这里追踪的算子通过 C++ 入口接入 PyTorch，再通过 Python 绑定作为 API 供模型调用，我们从最上层开始讲起

- 第一步：调用模块的 `forward` 方法
    - 测试调用 `nn.AvgPool2d` 类的 `forward` 方法 （[完整 AvgPool2d 类](nn模块源码.md#src-avgpool-class)）
        - 构造函数保存 `kernel_size`、`stride` 等参数；`forward` 把输入和这些参数交给下一层 functional 接口
- 第二步: 进行C binding
    - 第一步提到的计算交给的functional层为 `F.avg_pool2d`（[完整绑定赋值](nn模块源码.md#src-functional-binding)）
    - 为已绑定函数添加文档并赋名
        - `avg_pool2d = _add_docstr(torch._C._nn.avg_pool2d, docstring)`
        - Python方法 = 已有的绑定函数 + 文档
        - [`F`相关知识](nn模块附录.md#f)
        - [`_add_docstr`相关知识](nn模块附录.md#_add_docstr)
- 第三步：上游如何生成 C 函数
    - YAML 声明接口的参数和返回值，生成器据此生成下游包装层
      - 包装层的主要任务是参数转换、转调用代码
      - 我们的`avg_pool2d`的要求存放在[avg_pool2d](nn模块源码.md#src-native-schema)
    - 当我们build pytorch的时候，代码生成器会生成对应的 C++ 接口、绑定和注册代码
        - 生成C函数一部分看不懂没关系，后面的模块会展开讲解
        - 代码生成器 codegen 会依据yaml在build期间按照职责生成六类文件
            1. **Python 绑定：**
                1. **转换Python参数格式为CPP参数，调用CPP的接口**
                2. [Python 绑定生成器 method_impl](nn模块源码.md#src-python-binding-gen)
                3. 生成结果中包含 `torch._C._nn.avg_pool2d` 对应的 `THPVariable_avg_pool2d`；当前附的是生成器，名称怎样挂载见[附录](nn模块附录.md#python-binding-mount)
            2. **CPP接口**
                1. **提供CPP算子空间中的函数,给 Python 绑定调用的入口**
                2. [ATen C++ API 生成器 ComputeFunction](nn模块源码.md#src-cpp-api-gen)
            3. **调度入口**(part2)
                1. 用不准确但是易懂的方式来说，就是
                    1. **按名缓存 `op`，后续用 `op.call(...)` 调用所选实现；`op` 称为句柄**
                    2. **给后面的分发器(dispatcher)去选择具体的算子**
                2. [算子句柄与 call 生成器 ComputeOperators](nn模块源码.md#src-operator-gen)
                3. 同一生成器也生成 `call` 与 `redispatch` 的函数定义。
            4. **Schema注册**
                1. **登记算子叫什么、接收什么参数、返回什么**
                2. [Schema 注册](nn模块源码.md#src-schema-template)
            5. **后端注册与包装**
                1. **登记各种后端的实现入口，dispatcher选中后连接到具体计算代码。**
                2. [CPU/CUDA 后端注册](nn模块源码.md#src-backend-template)
                3. [完整内层注册模板](nn模块源码.md#src-dispatch-def-template)
            6. **自动微分代码**
                1. 结合求导规则[avg_pool2d](nn模块源码.md#src-derivatives)
                2. [VariableType](nn模块源码.md#src-variabletype-template)
                3. [自动微分 Functions.h](nn模块源码.md#src-functions-h-template)
                4. [自动微分 Functions.cpp](nn模块源码.md#src-functions-cpp-template)

### Part2. 包装好的CPP函数怎么找到对应的一类算子

#### Python 绑定部分

这一部分是build时候生成的，对应 Part1 第三步的“Python 绑定”，主要涉及的函数是`THPVariable_avg_pool2d`，这里我们还是要先做一个区分，与Part1的第二步相比

- 这里做的主要是让传入参数类型转换为能被CPP处理的参数
- 后者只是一个简单的命名+包装注释

`THPVariable_avg_pool2d` 的生成机制见 [method_impl](nn模块源码.md#src-python-binding-gen)，绑定容器见[完整 nn 绑定模板](nn模块源码.md#src-python-binding-template)。

- 第一步：利用 `parser` 按生成的参数签名解析本次 Python 调用传入的参数
- 第二步：调用局部函数 `dispatch_avg_pool2d`，在临时释放 GIL 后，把拆好的参数交给 `at::avg_pool2d`。名字的写法见[附录](nn模块附录.md#cpp-names)，lambda 与 GIL 的细节分别见[回调说明](nn模块附录.md#cpp-lambda)和 [Guard 说明](nn模块附录.md#cpp-guards)
- 第三步：用 `wrap` 将返回的 C++ Tensor 包装为 Python Tensor

#### `at::avg_pool2d`是怎么做的

以下为帮助理解的简化示意，真实生成逻辑见 [ComputeOperators](nn模块源码.md#src-operator-gen)。

```cpp
// 简化示意，不是本地已生成文件的原文
at::Tensor at::_ops::avg_pool2d::call(...) {
  static auto op = create_avg_pool2d_typed_handle();  // 首次成功初始化时查表，后续复用
  return op.call(self, kernel_size, ...);             // 之后每次:用缓存句柄进 dispatcher
}
static ... create_avg_pool2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(avg_pool2d::name, avg_pool2d::overload_name)  // "aten::avg_pool2d" + ""
      .typed<avg_pool2d::schema>();
}
```

- `at::avg_pool2d` 本体只有一行：`return at::_ops::avg_pool2d::call(...)`,位于[ATen C++ API 生成器 ComputeFunction](nn模块源码.md#src-cpp-api-gen)
- `at::_ops::avg_pool2d::call(...)`
    - 第一步，按 `avg_pool2d` 的名称找到算子记录，缓存其句柄。以后仍根据本次输入进行调度，缓存句柄不表示固定使用第一次选中的后端；[句柄的保存方式](nn模块附录.md#operator-handle)和 [static 写法](nn模块附录.md#cpp-static)见附录。
    - 第二步，调用句柄的`call`方法，分发器选中实现后，经[functor 包装代码](nn模块源码.md#src-unboxed-adapter)调用后端算子
    - [算子句柄与 call 生成器 ComputeOperators](nn模块源码.md#src-operator-gen)

### Part3. 如何通过dispatch key查询真正需要定位的算子

#### 定位算子的流程：

[OperatorEntry](nn模块源码.md#src-operator-entry)
上文的 `OperatorEntry` 保存名称、签名和后端实现，下面看它的成员

- `name_`：这个条目代表哪个算子重载，例如 `aten::avg_pool2d` 的默认重载。
- `schema_`：这个算子接收什么参数、返回什么结果等信息。
- `dispatchTable_`：各个运行时 dispatch key 对应的可调用实现数组。
- `dispatchKeyExtractor_`：帮助从本次参数及执行状态中提取调度所需的 keyset。

真正机制在后两者，我们分开聊，如果从真实的tensor计算时候调用的参数，他们配合逻辑是这样的：

- [Dispatcher::call](nn模块源码.md#src-dispatcher-call)
- 使用`dispatchKeyExtractor_`提取`dispatchKeySet`
- `lookup(keyset)` 返回表中选中的 `KernelFunction`
- 把原来的 Tensor 和池化参数传给选中的`KernelFunction`实现

#### 按照顺序，先讲`OperatorEntry`的`dispatchKeyExtractor_`

这里有 boxed 和 unboxed 两种传参入口。本例沿 unboxed 路径，直接接收前面解析出的 Tensor、整数列表和布尔值；提取器从适用的 Tensor 参数收集 key。Python Tensor 怎样取得 C++ 表示，以及 `Args...` 是什么，见[附录](nn模块附录.md#python-tensor-to-cpp)。boxed 入口的含义见[参数说明](nn模块附录.md#python-tensor-to-cpp)，参数包的语法见[模板说明](nn模块附录.md#cpp-templates)。

- 这个类包含几种参数
    - [DispatchKeyExtractor](nn模块源码.md#src-key-extractor)
    - `dispatch_arg_indices_reverse_`：预先存储 boxed 调用：参数栈里哪些位置需要检查
    - 其余三个字段：为 boxed/unboxed 调用记录：哪些层可以跳过，以及是否需要按后端区分
        - `nonFallthroughKeys_`、`nonFallthroughKeysPerBackend_`数组
        - `bool requiresBitsetPerBackend_`
        - 他们的用处会在后续详细讲解

**具体如何提取参数的运行步骤**

- 第一步：从 Tensor 收集后端、自动微分等 key（调度标记）
    - [DispatchKeyExtractor](nn模块源码.md#src-key-extractor) `getDispatchKeySetUnboxed`
    - 子第一步：用 `MultiDispatchKeySet` 收集参数中的 key，构造一个keyset。入口见 [multi_dispatch_key_set](nn模块源码.md#src-multi-key-fn)
        - 此处讲解的是较为简单的unbox方法
        - [MultiDispatchKeySet](nn模块源码.md#src-multi-key)
        - 将各个 Tensor 的 keyset 合并到 `ts`；同名处理函数的不同参数形式见[附录](nn模块附录.md#cpp-return)
        - `ts = ts | x.key_set();`
            - unboxed 方法通过重载处理各参数，合并参与调度的 Tensor 等对象的 `key_set()`
        - 此处的原始 keyset 主要来自输入张量 self。池化窗口大小会影响后续计算，但不会在这一步直接决定 CPU 或 NPU
        - 多张量算子则会合并多个输入的 keyset, 设备兼容性还需要相应实现检查
    - 子第二步：结合当前线程状态，并去掉可以跳过的层
        - [DispatchKeyExtractor](nn模块源码.md#src-key-extractor)
        - 将前面收集的 keyset 传给`getDispatchKeySetFromRawDispatchKeySet`
        - subsubstep1：按照当前线程状态增加、删减 keyset，见[提取器中的处理代码](nn模块源码.md#src-key-extractor)
            - `auto dispatch_keys = (ks | tls.included_) - tls.excluded_;`
            - 当前线程的加入与排除状态怎样影响调度，见[附录：TLS](nn模块附录.md#tls)
        - subsubstep2:返回`dispatch_keys & nonFallthroughKeys_ & key_mask`
            - `nonFallthroughKeys_`:
                - 过滤 fallthrough 层：这种实现直接放行，不做额外处理
                - 如果不同后端的 fallthrough 情况不同，使用`nonFallthroughKeysPerBackend_`
            - `key_mask`：额外允许保留哪些 key；这个调用路径默认传入 FULL

```text
self                  Tensor         → 收集 self.key_set()
kernel_size           整数列表        → 不贡献 keyset
stride                整数列表        → 不贡献 keyset
padding               整数列表        → 不贡献 keyset
ceil_mode             bool           → 不贡献 keyset
count_include_pad     bool           → 不贡献 keyset
divisor_override      可选整数        → 不贡献 keyset
```

#### 再讲`OperatorEntry`的`dispatchTable_`

- `dispatchTable_`: 可以存放各种算子后端实现的槽位的定长数组
    - 以`KernelFunction`作为元素类型
    - `num_runtime_entries`构建中确定的数组的槽位数量
    - 定义在[OperatorEntry](nn模块源码.md#src-operator-entry)
    - 本文重点关注两类槽位
        - CPU 槽位/`PrivateUse1` 槽位 (CPU 调用入口/NPU调用入口)
        - `AutogradCPU` 槽位/`AutogradPrivateUse1` 槽位(自动微分调用入口)

- `KernelFunction`
    - [KernelFunction](nn模块源码.md#src-kernel-class)
        - 这个对象的三个调用相关字段中，我们目前只关注 `unboxed_kernel_func_`：它保存 unboxed 适配函数入口的地址，不是 `KernelFunction` 对象的地址
- 这些实现何时填入调度表？注册时：
    - [Dispatcher::registerImpl](nn模块源码.md#src-register-impl)
    - 这里关注三项注册信息
        - 给哪个opname注册、
        - 注册在哪个 key 下、
        - 注册哪个kernelfunction？
    - 随后两步：（细节会在后面讲）
        - step1:`findOrRegisterName_`,根据算子名字找到已有记录，或者创建记录
        - step2：`op.operatorDef_->op.registerKernel` 保存该算子的 kernel 注册记录
            - `registerKernel()` 先保存注册记录，再更新调度表。
    - 注册后更新调度表；下面先看填一个数组槽位的步骤
        - 代码在[OperatorEntry::updateDispatchTableEntry_](nn模块源码.md#src-update-entry)
        - 利用`getDispatchTableIndexForDispatchKey`找到`dispatchkey`对应的数组下标`dispatch_ix`
        - 使用`computeDispatchTableEntry`计算`dispatcher, dispatch_key`对应的`KernelFunction`填入数组
            - 为什么要重新选择有效实现，见[附录](nn模块附录.md#effective-kernel)
    - key 怎样映射为数组下标，以及 `dispatch_ix == -1` 的含义，见[附录](nn模块附录.md#dispatch-table-index)
    - 重点：拿到数组元素以后，怎样真正调用函数？

### Part4. 如何通过privateuse1挂载，调用npu算子

两个框架交接的地方，如何从pytorch的`KernelFunction` 内部的 `unboxed_kernel_func_`到torchnpu的`NPU wrapper`

- [RegisterNPU.cpp 对应的生成逻辑](nn模块源码.md#src-npu-wrapper-gen)：该文件由构建生成，本地尚不存在；文件组装与注册块见 [gen_dispatcher_registrations](nn模块源码.md#src-npu-register-gen)。

#### 注册链条

先回到调用前的注册阶段。下面的 `W` 代表 `wrapper_NPU__avg_pool2d`；这是注册形状示意，完整生成器见 [NPU 注册文件生成器](nn模块源码.md#src-npu-register-gen)。

```cpp
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("avg_pool2d", TORCH_FN(W));
}
```

这段代码登记三件事：算子名是 `aten::avg_pool2d`，实现挂在 `PrivateUse1` 槽位，最终要执行的函数是 `W`。wrapper 和注册代码在构建时生成；运行时加载相关库，初始化过程才把它接入 Dispatcher。宏怎样生成初始化入口，见[附录](nn模块附录.md#registration-start)。

从 `m.impl(...)` 进入 [Library::impl](nn模块源码.md#src-library-impl)，它把传入的 wrapper 封装成 `CppFunction`。其构造过程调用 [makeFromUnboxedFunction](nn模块源码.md#src-make-function)，再由 [makeFromUnboxedFunctor](nn模块源码.md#src-make-functor)把适配函数入口和函数对象一起装入 `KernelFunction`。其中 `unboxed_kernel_func_` 保存适配函数的地址，这个适配函数后续会调用 wrapper。`TORCH_FN`、模板和对象创建的具体写法见[附录：wrapper 的装配](nn模块附录.md#registration-wrapper)。

随后，[Library::_impl](nn模块源码.md#src-library-_impl)把算子名、key 和准备好的实现交给 `Dispatcher::registerImpl`。它找到对应算子记录，再调用 [registerKernel](nn模块源码.md#src-register-kernel)保存注册信息，并通过 [updateDispatchTable_](nn模块源码.md#src-update-table)更新运行时入口。

注册阶段到这里结束：wrapper 已经通过包装层接入 `PrivateUse1` 的有效表项。下一阶段从一次真实调用的 `KernelFunction::call` 开始，沿保存的适配入口进入 wrapper。

#### 调用链条

Dispatcher 选中实现以后，在 [KernelFunction::call](nn模块源码.md#src-kernel-call)中取得先前保存的函数对象，并把它和当前计算参数交给 `callUnboxedKernelFunction`。

进入 [callUnboxedKernelFunction](nn模块源码.md#src-call-unboxed)，它执行保存的 `unboxed_kernel_func_` 入口。注册时，这个入口指向 `wrap_kernel_functor_unboxed_::call`，所以下一步进入[这个适配函数](nn模块源码.md#src-unboxed-adapter)。

适配函数调用注册时创建的函数对象。该对象的 [operator()](nn模块源码.md#src-wrap-function)再取得 `W` 的入口地址，带上当前输入和池化参数调用它，也就进入 `wrapper_NPU__avg_pool2d`。函数对象与指针的语法解释见[附录](nn模块附录.md#cpp-functors)。

这一阶段结束在 wrapper 的入口：输入与池化参数已经从 Dispatcher 传到 torch_npu。下一节从 wrapper 函数体中调用 `op_plugin::avg_pool2d(...)` 的语句继续追。

### Part5. 从 NPU wrapper 到 CANN 的 ACLNN 算子接口

上一节讲到，`m.impl` 将 `wrapper_NPU__avg_pool2d` 接入 Dispatcher；实际调用时，再经过适配器执行这个 wrapper。现在从它的函数体开始往下看。

#### wrapper 如何调用 op-plugin

`wrapper_NPU__avg_pool2d` 定义在构建生成的 `RegisterNPU.cpp` 中，见[本次找到的完整函数](nn模块源码.md#src-npu-wrapper-body)。它先做检查和环境准备，再把输入 Tensor 与池化参数交给 `op_plugin::avg_pool2d`。

**函数内部按顺序做什么？**

1. **记录并检查共同设备。** `check_and_update_common_device` 记录首次遇到的 Tensor 设备，后续 Tensor 与之比较，不同则报错。本例只有 `self`，因此主要记录 `self.device()`。可选值与未使用变量的写法见[附录](nn模块附录.md#cpp-types)。

2. **按开关检查输入存储是否安全。** `get_npu_data_unsafe_flag()` 为真时，调用 `check_npu_tensor_is_safe(self)`，检查 NPU allocator 对这块存储的安全标记；不安全则报错。它并不检查池化窗口大小或计算结果。

3. **设置设备上下文。** `OptionalDeviceGuard` 在这里使用 `self` 所在设备，退出时恢复原来的设备上下文。自动恢复的机制见[附录](nn模块附录.md#cpp-guards)。

4. **准备 profiler 记录。** 使用 `NPURecordFunction` 进行性能分析相关记录；该段是否编译进程序由构建标记决定，写法见[附录](nn模块附录.md#cpp-macros)。

5. **调用 op-plugin，并按需执行前后 hook。** 开启 `CheckOpHookEnable()`（要使用hook） 时，  
   1. 取得负责 hook 的对象，调用它的 PreHook，把输入和池化参数交给它，进行计算前的额外处理
   2. 再调用 `op_plugin::avg_pool2d`得到结果res
   3. 正常返回后执行 `PostHook(res)`，把res 交给计算后的额外处理入口，处理后返回结果；
   4. 未开启时直接调用并返回。

最后两个分支都只调用一次同一个函数：

```cpp
op_plugin::avg_pool2d(
    self, kernel_size, stride, padding,
    ceil_mode, count_include_pad, divisor_override);
```

这里的参数直接沿用 wrapper 收到的参数；开启 hook 时，用局部变量 `res` 接住返回的 Tensor，供 `PostHook` 使用。wrapper 本身没有分配池化输出，也没有计算平均值。

下一步要看的就是 **`op_plugin` 命名空间下的 `avg_pool2d` 函数体**。

#### op_plugin 如何选择池化实现

我们在生成文件 `OpInterface_19.cpp` 的第 582 行找到了这个函数，见[完整源码](nn模块源码.md#src-opinterface-avg-pool2d)。文件前面的 `namespace op_plugin {` 表明，这里的 `avg_pool2d` 就是 wrapper 调用的 `op_plugin::avg_pool2d`。

它负责选择下一层使用哪套池化实现。

这里先认识 **JIT（Just-In-Time，即时编译）**：可以理解为在程序运行期间，按需要进行编译，准备设备能够执行的算子程序。这里的开关控制的是 **CANN 算子的编译模式**，涉及在线编译与使用已经编译好的算子二进制文件，不能与 PyTorch 的 `torch.jit` 混为一谈。参见 [CANN 编译选项说明](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/API/appdevgapi/aclcppdevg_03_1371.html)。[附录：这里的 JIT](nn模块附录.md#jit)。

下面按执行顺序看这个函数：

1. **读取两个判断结果。**

   ```cpp
   bool is_jit_disable = at_npu::native::env::CheckJitDisable();
   bool self_base_format =
       at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
   ```

   `is_jit_disable` 记录是否关闭了这里的 JIT 编译模式；`self_base_format` 记录输入 `self` 是否使用基础的tensor在内存中的存储格式。

2. **记录调用信息。** `OP_EXEC_LOG` 和 `ASCEND_LOGI` 用于记录参数和状态，阅读调用主线时可以先跳过。

3. **两个条件都满足时，调用 `op_api` 实现。**

   ```cpp
   if (is_jit_disable && self_base_format) {
       return op_api::avg_pool2d(...);
   }
   ```

   关闭 JIT 且输入使用基础存储格式时，这个分支把输入和池化设置交给 `op_api::avg_pool2d`，再把输出返回 wrapper。条件与返回语法见[附录](nn模块附录.md#cpp-return)。

4. **条件不满足时，检查能否使用旧实现。**

   ```cpp
   else {
       if (c10_npu::IsAclnnOnly()) {
           TORCH_CHECK(false, ...);
       }
       return acl_op::avg_pool2d(...);
   }
   ```

   `IsAclnnOnly()` 判断当前设备是否只支持 ACLNN。如果是，`TORCH_CHECK(false, ...)` 会报错并中断，下面的调用就不会执行；否则调用旧的 `acl_op::avg_pool2d`。这里的省略号分别代表错误说明和函数参数，完整内容见源码。

因此，这几个同名函数的职责需要区分：

| 函数 | 当前这一层的职责 |
| --- | --- |
| `wrapper_NPU__avg_pool2d` | 检查输入、准备设备上下文，然后调用 op-plugin |
| `op_plugin::avg_pool2d` | 根据条件选择具体实现 |
| `op_api::avg_pool2d` | ACLNN 路径的实现入口 |
| `acl_op::avg_pool2d` | 旧路径的实现入口 |

仅凭这段代码，还不能确定某次运行一定走哪条分支，需要知道当时两个条件的值。下面分别看它们的函数体，我们下面会对一些辅助函数的处理减少讲解，他其实对整体的调用链条没有什么帮助，主要是细节的改变。

#### op_api：准备输出，然后调用 ACLNN

这个实现直接写在 op-plugin 的 `ops/opapi/AvgPool2dKernelNpuOpApi.cpp` 中，不需要通过生成器才能看到，见[完整 avg_pool2d](nn模块源码.md#src-pool-opapi)。

读这一层时，先区分三个名字：`self` 是输入 Tensor；`output_size` 是输出的形状，也就是每个维度多大；`result` 是准备用来存放计算结果的 Tensor。知道形状以后，就可以先分配存储空间，后面再把计算结果写进去。

1. **算出输出形状。**

   ```cpp
   output_size = calc_output_size_with_generalized_attrs(
       self, kernel_size, stride, padding,
       ceil_mode, count_include_pad, divisor_override);
   ```

   此处省略了变量的类型，完整声明见源码。[这个辅助函数](nn模块源码.md#src-pool-generalize)先把池化窗口、步长和填充解释成二维参数：例如窗口 `(2)` 按 `(2, 2)` 理解，未指定步长时使用窗口大小。它也检查步长不能为零、填充不能为负等条件，再交给[输出尺寸推导函数](nn模块源码.md#src-pool-output-size)。

   例如输入形状为 `[1, 1, 4, 4]`（一批有几张图片、每张图片有几个通道，图片高度，图片宽度），窗口和步长都是 `(2, 2)`，没有填充时，输出形状为 `[1, 1, 2, 2]`。这一步只确定输出需要装多少数字，还没有算出这些数字。

2. **分配输出 Tensor。**

   ```cpp
   at::Tensor result =
       at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
   ```

   [这次分配](nn模块源码.md#src-pool-output-allocation)使用输入的设备、数据类型等选项，按刚算出的形状创建输出存储。此时 `result` 已经有空间，但池化结果还没有写入。`without_format` 表示这里采用普通连续存储的创建路径，不要求沿用输入的 NPU 私有存储格式。

3. **确认当前环境提供 ACLNN 接口。**

   ```cpp
   DO_COMPATIBILITY(
       aclnnAvgPool2d,
       acl_op::avg_pool2d(self, kernel_size, stride, padding,
                          ceil_mode, count_include_pad, divisor_override));
   ```

   `DO_COMPATIBILITY` 是宏，可以先理解为一段放到当前位置执行的公共代码。它检查 `aclnnAvgPool2d` 和 `aclnnAvgPool2dGetWorkspaceSize` 两个接口是否能找到。接口齐全就继续往下；缺失时，如果设备允许旧实现，就调用 `acl_op::avg_pool2d` 并直接返回；只支持 ACLNN 的设备则报错。见[完整宏](nn模块源码.md#src-pool-compatibility)。

   这是调用前的兼容检查，不表示“ACLNN 计算出错后再尝试旧实现”。即使上一层选中了 `op_api`，这里仍可能转入 `acl_op`。

4. **把输出和参数交给执行辅助函数。**

   ```cpp
   avg_pool2d_out_npu_nocheck_opapi(
       result, self, kernel_size, stride, padding,
       ceil_mode, count_include_pad, divisor_override);
   return result;
   ```

   这里把已分配的 `result` 传进去，让后续计算使用这块输出存储。辅助函数的名字带有 `out`，表示接收已有输出 Tensor；它与另一个单独的 `avg_pool2d_out` 函数需要区分，这个函数用于处理传入参数中带有目标输出tensor的情况，在pytorch里的对应重载`aten::avg_pool2d.out`。

**辅助函数里怎样发起计算？**

[avg_pool2d_out_npu_nocheck_opapi](nn模块源码.md#src-pool-opapi-helper)先整理两个参数：`divisor_override` 是手动指定平均值的除数，未指定时用 `0` 表示采用默认规则，显式指定 `0` 则报错；`cube_math_type` 是传给底层的计算精度选项，这里先知道它作为参数继续传下去即可。

然后执行完整的调用语句：

```cpp
EXEC_NPU_CMD(
    aclnnAvgPool2d,
    self, kernel, stride, padding,
    ceil_mode, count_include_pad,
    s_divisor_override, cube_math_type, result);
```

`EXEC_NPU_CMD` 是负责调用 ACLNN 的公共宏，`aclnnAvgPool2d` 指定本次要调用的算子接口，其余内容是输入、池化参数和输出。它的内部调用步骤是后面要继续追的下一层。

前面的形状辅助函数使用自己整理出的二维参数推算尺寸；这里仍然接收原来的参数列表，不能把推算时补齐列表理解为原参数已经被修改。

#### acl_op：整理参数，再组装 AvgPoolV2 命令

旧实现直接写在 op-plugin 的 `ops/aclops/AvgPool2dKernelNpu.cpp` 中，见[完整 avg_pool2d](nn模块源码.md#src-pool-aclop)。它也要先准备输出，但参数整理和执行入口与 `op_api` 不同。

1. **检查参数。** [avg_pool2d_parameter_check](nn模块源码.md#src-pool-aclop-check)检查窗口、步长、填充的参数个数，以及输入是否为三维或四维 Tensor；本地这版代码还要求显式指定的 `divisor_override` 在 `[1, 255]` 内。

2. **把三维输入临时按四维处理。** 三维输入形状为 `[C, H, W]`，四维则为 `[N, C, H, W]`，分别表示通道、高度、宽度，以及额外的批次维度。输入是三维时，`unsqueeze(0)` 在开头加一个大小为 `1` 的维度，变成 `[1, C, H, W]`，方便后面的统一处理。这里的 `self_copy = self` 和增加维度共用原来的数据存储，并不复制整份输入数字。

3. **补齐二维参数，算输出形状并分配结果。** 窗口、步长、填充分别整理成两个数，未指定步长时采用窗口大小。然后同样调用 `avg_pool2d_npu_output_size`，再用 `npu_preparation::apply_tensor` 分配 `result`。这里的 `npu_preparation` 是 `OpPreparation` 的简称，分配实现见[完整函数](nn模块源码.md#src-pool-aclop-allocation)。

4. **调用 `avg_pool2d_out_nocheck` 执行辅助函数。** 它接收 `result`、整理后的输入和二维参数，继续组装执行命令，见[完整辅助函数](nn模块源码.md#src-pool-aclop-helper)。

5. **恢复三维输出。** 如果原输入是三维，返回前用 `squeeze(0)` 去掉刚才临时增加的批次维度，使输出继续保持 `[C, H_out, W_out]` 的形式。

**辅助函数里怎样组装命令？**

```cpp
at_npu::native::OpCommand cmd;
cmd.Name("AvgPoolV2")
   .Input(self)
   .Output(result)
   .Attr("ksize", kernel_size_new)
   .Attr("strides", strides_size_new);
```

这里展示了命令的前几项，完整属性见源码。`cmd` 可以理解为一份执行请求：

| 写法 | 在请求中记录什么 |
| --- | --- |
| `.Name("AvgPoolV2")` | 要执行的 CANN 算子名称 |
| `.Input(self)` | 使用哪个输入 Tensor |
| `.Output(result)` | 使用哪个输出 Tensor 存放结果 |
| `.Attr(名称, 值)` | 窗口大小、步长、填充等设置 |
| `.Run()` | 将组装好的请求交给执行框架 |

设置名称和属性时还没有计算池化结果，最后的 `.Run()` 才进入执行流程。连续调用的语法见[附录](nn模块附录.md#cpp-return)。

辅助函数把窗口从 `[k_h, k_w]` 整理成 `[1, 1, k_h, k_w]`，步长也扩成四维，前两个 `1` 对应批次和通道维度。[get_paddings](nn模块源码.md#src-pool-aclop-padding)则计算上、下、左、右四个填充值，并处理 `ceil_mode` 下的边界。

命令还设置是否把填充位置计入平均值。代码通常用 `exclusive = !count_include_pad` 表达相反的开关；本地实现对 `Half` 和 `Char` 类型会将 `exclusive` 设为 `true`。最后设置除数并执行：

```cpp
cmd.Attr("divisor_override", divisor_override_value).Run();
```

到这里，两条实现的下一步就明确了：`op_api` 进入 `EXEC_NPU_CMD(aclnnAvgPool2d, ...)`，`acl_op` 进入 `OpCommand::Run()`，执行它组装的 `AvgPoolV2` 请求。两条路径都使用 NPU 执行接口，`acl_op` 并不表示转到 CPU。

#### EXEC_NPU_CMD：把池化参数交给 CANN

上一层已经分配了输出 `result`，现在需要让 CANN 使用输入和池化参数，向这块输出存储写入结果。先沿 [V1 宏](nn模块源码.md#src-exec-npu-v1)中未命中执行器缓存的主线看，其他分支见[附录](nn模块附录.md#npu-execution-details)。

下面始终追踪同一次池化请求：它先在宏中被包装成 `acl_call`，传入执行函数后叫 `func`；如果需要排队，又会被保存到任务对象的 `customHandler` 中。名称和保存位置在变化，我们要看的是这份待执行的计算怎样传下去，最后在哪一句被调用。

这里会遇到三个新名字：

| 名字 | 用来做什么 |
| --- | --- |
| `workspace` | 计算过程中使用的临时设备内存；最终结果放在已有的 `result` 中 |
| `executor` | 第一段 CANN 接口准备好的执行信息，第二段接口用它发起计算 |
| `stream` | NPU 任务的执行通道，用来安排任务顺序 |

宏内部按顺序做以下几件事：

1. **找到两个 CANN 函数的地址。**

   ```cpp
   GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");
   GetOpApiFuncAddr(#aclnn_api);
   ```

   本例查找的是 `aclnnAvgPool2dGetWorkspaceSize` 和 `aclnnAvgPool2d`。地址由[查找函数](nn模块源码.md#src-opapi-symbol)返回，后面据此调用两个接口；宏怎样组成这两个名字，见[附录](nn模块附录.md#cpp-macros)。

2. **转换参数表示。**

   ```cpp
   auto converted_params =
       ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);
   ```

   [ConvertTypes](nn模块源码.md#src-opapi-convert-pack)接收输入、窗口、步长、填充、输出等参数，把它们整理成 CANN 接口需要的表示。可变参数宏的写法见[附录](nn模块附录.md#cpp-macros)。

   对普通 NPU Tensor，这一步创建描述对象，记录形状、数据类型和已有数据的存储地址。可以理解为给 CANN 一张说明卡，告诉它“数据是什么、在哪里”，见[Tensor 转换函数](nn模块源码.md#src-opapi-convert-tensor)。

3. **调用第一段接口，取得执行信息和临时空间大小。**

   ```cpp
   auto workspace_status = call(getWorkspaceSizeFunc, converted_params);
   ```

   这里的 `getWorkspaceSizeFunc` 指向 `aclnnAvgPool2dGetWorkspaceSize`。`call` 将整理后的参数交给它，CANN 把所需临时内存大小填回 `workspace_size`，把执行信息填回 `executor`。返回的 `workspace_status` 用来检查这次调用是否成功。

4. **按大小分配临时空间。** 如果 `workspace_size` 不为零，就调用 `unsafe_empty_workspace(workspace_size)` 分配设备内存，并取得它的地址 `workspace_addr`；为零则不需要分配。见[分配函数](nn模块源码.md#src-opapi-workspace)。

5. **提交计算。** 前面的参数、executor 和 workspace 准备好以后，宏执行：

   ```cpp
   OpCommand::RunOpApiV2("aclnnAvgPool2d", acl_call);
   ```

   这一句是当前计算的提交入口，两个参数分别表示：`"aclnnAvgPool2d"` 是本次操作的名称，用于日志、性能记录等；`acl_call` 是交给框架执行的函数，其中写着真正调用 CANN 的代码。计算动作由第二个参数提供，不能只根据第一个字符串理解为框架会再次按名字查找算子。

   接下来先看 `acl_call` 写了什么，再看 `RunOpApiV2` 怎样执行它。注意，V1 宏也调用这个名为 `RunOpApiV2` 的方法，两处版本号描述的是不同层。

##### acl_call：把 CANN 调用写成一个函数

`acl_call` 在 V1 宏中这样创建，下面省略了流资源设置和可选清理代码，完整内容见[宏源码](nn模块源码.md#src-exec-npu-v1)：

```cpp
auto acl_call =
    [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int {
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
        auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
        NPU_CHECK_ERROR(api_ret, "call " #aclnn_api " failed");
        ReleaseConvertTypes(converted_params);
        return api_ret;
    };
```

`acl_call` 保存了本次执行所需的信息，创建时还没有运行函数体。后面调用它，才会执行下面的 CANN 调用。lambda 的方括号、捕获与返回类型，统一见[附录](nn模块附录.md#cpp-lambda)。

函数体依次做以下工作：

1. **把地址解释成可调用的函数指针。**

   ```cpp
   OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
   ```

   这句准备好指向 `aclnnAvgPool2d` 的可调用入口，还没有执行池化。函数指针类型和转换语法见[附录](nn模块附录.md#cpp-dynamic-library)，完整类型定义见[源码](nn模块源码.md#src-opapi-function-call)。

2. **真正调用 CANN 的第二段接口。**

   ```cpp
   auto api_ret =
       opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
   ```

   这一行调用的是 `aclnnAvgPool2d`，传入临时空间的地址和大小、准备好的 executor，以及执行流。输入、输出和池化设置已经在第一段准备好，第二段据此提交设备计算。两段接口的对应关系见 [CANN 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnAvgPool2d.md)。

3. **检查返回码。** `NPU_CHECK_ERROR(api_ret, ...)` 检查 CANN 调用是否成功，出错时报告错误。`api_ret` 是状态码，不是计算出来的 Tensor；池化结果写入前面 `result` 对应的设备存储。

4. **正常返回时清理辅助资源。** `ReleaseConvertTypes(converted_params)` 释放转换得到的 ACL 描述等对象；完整函数还在可用时调用 `ReleaseHugeMem` 做清理。这里释放的是辅助资源，不表示把输入或输出 Tensor 的数据一起删掉。最后 `return api_ret` 把状态码交给执行框架。

写到这里，我们已经知道这份请求被包装成了什么：`acl_call` 保存执行所需的信息，函数体中写着 `aclnnAvgPool2d` 的调用。下一步回到宏的 `RunOpApiV2("aclnnAvgPool2d", acl_call)`，进入 `RunOpApiV2`，看它怎样接收并执行这个回调。

##### RunOpApiV2：决定怎样执行 acl_call

现在回到提交语句：

```cpp
OpCommand::RunOpApiV2("aclnnAvgPool2d", acl_call);
```

[这个方法](nn模块源码.md#src-opapi-run)的参数为 `op_name`、`func`、`sync`。传进来后，`op_name` 对应字符串 `"aclnnAvgPool2d"`，`func` 对应刚才的 `acl_call`；第三个参数 `sync` 没有填写，采用默认值 `false`。

`func` 保存传入的可执行回调，框架通过它调用 `acl_call`。`PROC_FUNC` 与 `std::function` 的具体写法见[附录](nn模块附录.md#cpp-lambda)，方法声明见[源码](nn模块源码.md#src-opapi-submit-types)。

方法先取得当前 NPU 流，再判断：

```cpp
if (!stream.isSyncLaunchStream() && OptionsManager::GetTaskQueueEnable()) {
    // 交给主机任务队列
} else {
    // 在当前调用路径执行
}
```

这里省略了 `OptionsManager` 前面的命名空间。条件表示当前流不是同步发起类型的流，并且开启了任务队列。两者同时满足才走队列，否则直接执行。这个判断选择的是 CANN 调用怎样发起，不能单凭它判断设备是否已经完成计算。

**先看直接执行的情况。**

进入 `RunOpApiV2` 的 `else` 后，我们在 [OpCommand.cpp:267](nn模块源码.md#src-opapi-run) 看到：

```cpp
OpCommandImpl::RunOpApi(op_name, func);
```

这句把名称和回调交给 `OpCommandImpl::RunOpApi`。这里的 `func` 仍然是前面准备的 `acl_call`，它只是换了一个参数名，没有变成另一个计算函数。

接下来打开 `OpParamMaker.cpp` 中的 `RunOpApi`，在 [274 行](nn模块源码.md#src-opapi-run-impl) 可以看到：

```cpp
ACL_REQUIRE_OK_OP(InnerRunOpApi(op_name, func), op_name.c_str());
```

这句先调用 `InnerRunOpApi(op_name, func)`，外面的宏负责检查返回码。不同编译条件下，同样的调用还会出现在 277 或 280 行。因此下一步要进入的就是 `InnerRunOpApi`，`func` 也跟着传了进去。

在 [OpParamMaker.cpp:392](nn模块源码.md#src-opapi-inner-run-impl)，这个函数又把一段代码交给 `RunWithDeterministicSnapshot`：

```cpp
return RunWithDeterministicSnapshot(snapshot, true, [&]() -> aclError {
    // 这里放着后面要执行的代码
});
```

上面只展示调用的外壳，完整函数体见链接。它将后面的执行代码交给 `RunWithDeterministicSnapshot`，回调的写法见[附录](nn模块附录.md#cpp-lambda)。我们继续打开这个辅助函数，在 [166 行](nn模块源码.md#src-execution-snapshot) 找到：

```cpp
return launch();
```

`launch` 就是刚传进来的这个小函数。执行 `launch()` 后，才开始运行 `InnerRunOpApi` 大括号中的代码，其中 [395 行](nn模块源码.md#src-opapi-inner-run-impl) 是：

```cpp
ret = func();
```

这时终于调用到了 `func`，也就是最初的 `acl_call`。这里有两个不同的小函数：`launch` 包着运行状态和重试逻辑，`func` 保存池化的 CANN 调用。沿着它们进入各自的函数体，就能接上每一步。

执行 `acl_call()` 后，我们回到 V1 宏中定义的 lambda，在 [op_api_common.h:394](nn模块源码.md#src-exec-npu-v1) 执行：

```cpp
opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
```

`opApiFunc` 指向 `aclnnAvgPool2d`，所以这一句跨入 CANN 的第二段接口，提交池化计算。

这条路径结束在 `acl_call` 内部的 `opApiFunc(...)`。沿途的 `RunOpApiV2`、`RunOpApi` 和 `InnerRunOpApi` 一直传递同一次计算的回调，直到 `func()` 把它执行起来，才进入 CANN。下面回到 `RunOpApiV2` 的另一条分支，看同一份回调需要排队时会被交到哪里。

**再看排队执行的情况，先追任务怎样保存。**

这次我们回到 `RunOpApiV2` 的 `if`。它不会先调用上面的 `RunOpApi`，而是在 [OpCommand.cpp:246](nn模块源码.md#src-opapi-run) 把回调的位置记录到请求里：

```cpp
execParams.customHandler = const_cast<PROC_FUNC*>(&func);
```

`func` 仍是 `acl_call`；这句只是让请求能够找到它。248 行再将这个请求装入 `params`，任务类型为 `EXECUTE_OPAPI_V2`。随后，[252 行](nn模块源码.md#src-opapi-run) 执行：

```cpp
c10_npu::enCurrentNPUStream(&params);
```

所以，从 `RunOpApiV2` 继续追排队路径时，下一处应该打开的是 `NPUStream.cpp` 中的 `enCurrentNPUStream`。它收到同一个任务，只是在函数内部把参数称为 `cur_paras`。

在 [NPUStream.cpp:804](nn模块源码.md#src-queue-en-current)，开启每流队列时会执行：

```cpp
ptr->repo->Enqueue(cur_paras);
```

未开启每流队列时，[810 行](nn模块源码.md#src-queue-en-current) 则使用 `default_streams[device_index].repo->Enqueue(cur_paras)`。两者选用的队列对象不同，但下一步都进入 `Repository::Enqueue`。这里的 `repo` 就是管理待办任务的队列对象。

打开 `NPUQueue.cpp` 中的 `Enqueue`，继续走到 [659 行](nn模块源码.md#src-queue-enqueue)：

```cpp
ret = WriteQueue(cur_paras);
```

这句进入同一个类的 `WriteQueue`。它要把任务放进队列的存储位置，因此在 [381 行](nn模块源码.md#src-queue-write) 调用：

```cpp
manager().Copy(data, write_idx.idx, cur_paras);
```

`data` 是队列的存储区域，`write_idx` 标记这次写到哪里，`cur_paras` 是我们一路传下来的任务。现在进入 `CallBackManager::Copy`，在 [75 行](nn模块源码.md#src-queue-callback-manager) 看到：

```cpp
return this->copyFunc(dstPtr, src);
```

这里没有直接写 `CopyFunc`，而是通过保存在 `copyFunc` 字段中的函数指针调用。要知道它指向谁，需要回头看 [OpParamMaker.cpp:863](nn模块源码.md#src-queue-callback-registration) 的注册语句：

```cpp
REGISTER_QUEUE_FUNC(AsncExecFunc, CopyFunc, ...)
```

这里节选了前两个参数。宏创建注册对象，其构造函数在 [NPUQueue.cpp:142](nn模块源码.md#src-queue-callback-registration) 和 [143 行](nn模块源码.md#src-queue-callback-registration) 分别执行 `manager().SetExec(execFunc)`、`manager().SetCopy(copyFunc)`，把 `AsncExecFunc` 和 `CopyFunc` 保存下来。这是在库加载时接好的关系。因此刚才的 `this->copyFunc(dstPtr, src)` 实际调用的是 `OpParamMaker.cpp` 中的 `CopyFunc`。

进入 `CopyFunc` 后，[707 行](nn模块源码.md#src-opapi-queue-copy) 将本次任务从 `EXECUTE_OPAPI_V2` 标记为队列保存时使用的 `EXECUTE_OPAPI`。接着 [716 行](nn模块源码.md#src-opapi-queue-copy) 调用目标任务对象的 `Copy(...)`，进入 `NPUDefine.cpp` 中的 `ExecuteParasOpApi::Copy`。它保存名称，并在 [56 行](nn模块源码.md#src-opapi-queue-copy) 执行：

```cpp
this->customHandler = std::move(*(other.customHandler));
```

`other.customHandler` 是最开始记录的 `func` 的位置；这句将回调转移到队列自己的任务对象中。走到这里，队列持有的 `customHandler` 已经保存了 `acl_call`。保存任务的过程完成了，但还没有执行这个回调；`std::move` 的含义见[附录](nn模块附录.md#cpp-ownership)。

这一阶段停在 `ExecuteParasOpApi::Copy`：请求已经从调用者传入的 `func` 转移到队列对象的 `customHandler`。下一阶段要换到负责读取队列的执行流程，从它的入口 `StartConsume` 开始，找到这份任务何时被取出并调用。

**任务保存好了，接着看谁把它取出来执行。**

队列初始化时会启动专门处理任务的线程：提交算子的代码负责放入任务，它负责取出任务。线程与主机队列的基础说明见[附录](nn模块附录.md#npu-execution-details)。

这个线程是从哪来的？使用每流队列时，`enCurrentNPUStream` 在 [NPUStream.cpp:801](nn模块源码.md#src-queue-en-current) 调用 `ptr->repo->InitRepo(device_index)`。使用默认队列时，则先由 [773 行](nn模块源码.md#src-queue-en-current) 的 `initNPUStreamsOnce()` 进入初始化流程，其中 [282 行](nn模块源码.md#src-stream-queue-initialization) 调用 `initGlobalStreamState()`，后者在 [245 行](nn模块源码.md#src-stream-queue-initialization) 调用默认队列的 `InitRepo(device_id)`。这是准备队列时做的工作，不会为每个池化调用重复创建线程。

现在打开 `InitRepo`，在 [NPUQueue.cpp:897](nn模块源码.md#src-queue-init) 可以看到启动语句：

```cpp
std::thread cur_consumer(StartConsume, this, device_id);
```

它让新线程从 `StartConsume` 开始执行，并把当前队列对象交给它。我们继续读 `StartConsume`，在 [878 行](nn模块源码.md#src-queue-consume) 的循环中找到：

```cpp
repo->Dequeue();
```

这句进入 `Repository::Dequeue`。`Dequeue` 再在 [744 行](nn模块源码.md#src-queue-dequeue) 调用 `ReadQueue()`，读取当前待执行任务。于是我们继续打开 `ReadQueue`，在 [457 行](nn模块源码.md#src-queue-read) 找到：

```cpp
auto ret = manager().Call(data, read_idx.idx);
```

前面保存任务时使用 `write_idx`，这里执行任务时使用 `read_idx`，它标记当前读哪个位置。接下来进入 `CallBackManager::Call`，在 [69 行](nn模块源码.md#src-queue-callback-manager) 执行：

```cpp
return this->execFunc(dstPtr);
```

还记得前面注册的两个函数吗？`copyFunc` 用于保存任务，`execFunc` 用于执行任务。后者已被设为 `AsncExecFunc`，所以这里进入 `OpParamMaker.cpp` 的 `AsncExecFunc`。

这个函数读出任务类型，再在 [833 行](nn模块源码.md#src-opapi-queue-dispatch) 执行：

```cpp
auto ret = funcMap[type](queueParam, stream);
```

`funcMap` 是“任务类型对应哪个执行函数”的映射。本次保存后的类型是 `EXECUTE_OPAPI`，源码中对应 `{EXECUTE_OPAPI, ExecFuncOpApi}`，所以下一步进入 `ExecFuncOpApi`。

终于，在这个函数的 [558 行](nn模块源码.md#src-opapi-queue-exec) 可以看到：

```cpp
ret = cur_paras->customHandler();
```

`cur_paras` 是刚从队列取出的任务对象，它的 `customHandler` 就是保存进去的 `acl_call`。执行这一句，才开始运行那个 lambda；随后回到 [op_api_common.h:394](nn模块源码.md#src-exec-npu-v1)，调用 `opApiFunc(...)`，也就是 `aclnnAvgPool2d(...)`。

因此，`RunOpApiV2` 的 252 行与 `ExecFuncOpApi` 的 558 行之间，是先保存任务、再由另一条执行流程取出并调用回调。连接两边的是队列中的任务对象，其中保存着同一个 `acl_call`。

取任务的这一阶段结束在 `ExecFuncOpApi` 的 `customHandler()`：它执行保存的 `acl_call`，接着由回调中的 `opApiFunc(...)` 进入 CANN 的 `aclnnAvgPool2d`。也就是说，直接执行和排队执行走了不同的主机代码，但最后执行的都是先前包装好的池化调用。

**等待与记录。** `RunOpApiV2` 还包含 profiler、日志和可选的流同步。队列分支统计的时间是入队耗时，不能当成 NPU 池化耗时；直接分支在阻塞配置允许时等待流完成，方法末尾还会在 `sync == true` 时等待。本次调用的 `sync` 默认是 `false`，设备任务是否已完成需要结合具体路径和同步行为判断。

至此，`op_api` 路径已追到 `aclnnAvgPool2d` 这个 CANN 主机侧接口。接下来回到 `op_plugin` 选择的另一条 `acl_op` 路径：它把计算包装成带名称、输入输出和属性的 `OpCommand` 请求，我们从这个请求的 `.Run()` 继续追。

#### OpCommand::Run：沿请求追到 AvgPoolV2 的 CANN 调用

旧路径在 `AvgPool2dKernelNpu.cpp` 中用 `.Name("AvgPoolV2")`、`.Input(self)`、`.Output(result)` 和 `.Attr(...)` 保存了名称、输入、输出和属性，最后执行 `.Run()`。现在从这一次调用继续往下读。

`OpCommand` 内部用 `aclCmd` 保存请求。它在[构造函数](nn模块源码.md#src-command-constructor)中取得这个对象，并将 `customHandler` 设为空。本例后续没有设置回调，因此保存下来的是 `AvgPoolV2` 的名称和计算参数。

这次的主线是看这些信息怎样传递：直接执行时，它们随 `execParam` 进入 `InnerRun`；排队时，先导出到 `ExecuteParas` 任务中，随后由 `ExecFunc` 取出。两种情况下都要继续追到使用这些信息调用 CANN 的那一句。

**先从直接执行分支开始。**

打开 `OpCommand::Run`，进入它的 `else`，在 [OpCommand.cpp:172](nn模块源码.md#src-aclop-command-run) 看到：

```cpp
aclCmd->Run(sync, sync_index, outputTensor);
```

`aclCmd` 是 `OpCommandImpl` 对象，所以这一句进入的是 `OpParamMaker.cpp` 中的 `OpCommandImpl::Run`。两个函数都叫 `Run`，但分属不同的类：外面的 `OpCommand::Run` 决定怎样提交，里面的 `OpCommandImpl::Run` 继续处理保存的请求。

继续读里面的 `Run`，在 [257 行](nn模块源码.md#src-aclop-impl-run) 找到：

```cpp
ACL_REQUIRE_OK_OP(
    InnerRun(opName, execParam, sync, sync_index, outputTensor),
    opName.c_str());
```

这句先调用 `InnerRun`，然后检查返回码。不同编译条件下，同样的调用也可能位于 260 或 263 行。传入的 `opName` 是 `"AvgPoolV2"`，`execParam` 则是前面保存的输入、输出和属性，所以这些信息会一起进入 `InnerRun`。

在 [OpParamMaker.cpp:300](nn模块源码.md#src-aclop-impl-run)，`InnerRun` 将后面的计算代码交给 `RunWithDeterministicSnapshot`。这个辅助函数与前面相同，应用运行配置后，在 [166 行](nn模块源码.md#src-execution-snapshot) 执行 `launch()`，开始运行刚传入的代码。

本例的 `sync` 初始值是 `false`，池化辅助函数也没有调用 `.Sync()` 修改它。因此这段代码进入 `if (!sync)`，在 [335 行](nn模块源码.md#src-aclop-impl-run) 调用：

```cpp
ret = aclopCompileAndExecute(
    name.c_str(),
    inputSize, params.inDesc.data(), params.inBuffer.data(),
    outputSize, params.outDesc.data(), params.outBuffer.data(),
    params.attr,
    ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr, stream);
```

这里的 `name` 仍是 `"AvgPoolV2"`。输入、输出分别包含描述和数据位置，`params.attr` 保存窗口、步长、填充等属性，`stream` 指定执行流。走到这一句，请求才进入 CANN 的通用单算子接口。它通过名称和这些参数处理本次计算，接口约定见 [CANN 文档](https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC2/apiref/appdevgapi/aclcppdevg_03_0243.html)。

这条直接执行路径停在 `InnerRun` 的 `aclopCompileAndExecute(...)`：`OpCommand` 保存的请求已经沿 `aclCmd → execParam → params` 交到这里，下一层就是 CANN。下面回到 `OpCommand::Run` 的排队分支，看这份请求怎样先存入任务对象，再被执行。

**如果 Run 选择了排队分支，请求又怎样往下传？**

我们回到 `OpCommand::Run` 的 `if`，先看 [OpCommand.cpp:159](nn模块源码.md#src-aclop-command-run)：

```cpp
aclCmd->ExportParams(execParams);
```

这句进入 `OpParamMaker.h` 的 [ExportParams](nn模块源码.md#src-aclop-export)，把内部请求的名称、输入输出和属性整理到 `execParams` 中。随后 160 行将它装入类型为 `COMPILE_AND_EXECUTE` 的队列任务，再在 [161 行](nn模块源码.md#src-aclop-command-run) 执行：

```cpp
c10_npu::enCurrentNPUStream(&params);
```

这次又进入了前面读过的 `enCurrentNPUStream`。在 [NPUStream.cpp:804](nn模块源码.md#src-queue-en-current) 或 [810 行](nn模块源码.md#src-queue-en-current)，它调用对应队列的 `Enqueue`；`Enqueue` 在 [NPUQueue.cpp:659](nn模块源码.md#src-queue-enqueue) 调用 `WriteQueue(cur_paras)`；`WriteQueue` 再在 [381 行](nn模块源码.md#src-queue-write) 调用 `manager().Copy(...)`。

进入 `CallBackManager::Copy` 后，[75 行](nn模块源码.md#src-queue-callback-manager) 的 `this->copyFunc(dstPtr, src)` 仍然指向 `CopyFunc`。但这次任务类型是 `COMPILE_AND_EXECUTE`，因此 `CopyFunc` 不走之前保存 `acl_call` 的分支，而是在 [OpParamMaker.cpp:722](nn模块源码.md#src-opapi-queue-copy) 调用 `ExecuteParas::Copy(...)`。这个 [Copy 函数](nn模块源码.md#src-aclop-export)把 `AvgPoolV2` 名称、输入输出和属性等请求信息保存到队列对象中，任务类型也保持为 `COMPILE_AND_EXECUTE`。

到这里停在 `ExecuteParas::Copy`，名称和参数已经由原来的 `aclCmd` 导出并存入队列任务。接下来仍从 `StartConsume` 开始取任务；这次要关注的是 `COMPILE_AND_EXECUTE` 类型最终交给哪个执行函数。

请求保存以后，负责队列的线程继续运行前面读过的那一段：`StartConsume` 在 [NPUQueue.cpp:878](nn模块源码.md#src-queue-consume) 调用 `Dequeue()`，`Dequeue` 在 [744 行](nn模块源码.md#src-queue-dequeue) 调用 `ReadQueue()`。`ReadQueue` 再于 [457 行](nn模块源码.md#src-queue-read) 调用 `manager().Call(...)`，进入 `CallBackManager::Call`。其 [69 行](nn模块源码.md#src-queue-callback-manager) 调用保存的 `execFunc`，也就是 `AsncExecFunc`。

现在又走到了 [OpParamMaker.cpp:833](nn模块源码.md#src-opapi-queue-dispatch)：

```cpp
auto ret = funcMap[type](queueParam, stream);
```

这一次，`type` 是 `COMPILE_AND_EXECUTE`，映射表中的对应项为 `{COMPILE_AND_EXECUTE, ExecFunc}`。所以从这一句进入的执行函数变成了 `ExecFunc`，而不是之前的 `ExecFuncOpApi`。

进入 `ExecFunc` 后，本例的 `customHandler` 为空，会继续走普通旧接口请求的流程。在 [514 行](nn模块源码.md#src-aclop-queued-exec)，它根据队列任务中的名称、输入输出和属性，调用 `aclopCompileAndExecute(...)`。其中名称来自 `cur_paras->opType`，值还是 `"AvgPoolV2"`。

这样，旧路径也接到了 CANN：直接执行时，调用位于 `InnerRun` 的 335 行；排队执行时，调用位于 `ExecFunc` 的 514 行。两边使用的名称和计算参数来自同一份 `OpCommand` 请求。

到这里，`op_api` 通过回调中的 `aclnnAvgPool2d` 提交计算，`acl_op` 通过通用接口 `aclopCompileAndExecute` 提交名为 `AvgPoolV2` 的请求。设备计算完成时，结果写入此前准备的输出存储；接口返回与设备完成可能是两个时间点，等待行为见[附录](nn模块附录.md#npu-execution-details)。

这一节结束时，执行请求已从 torch_npu 交到 CANN：`op_api` 的进入点是 `acl_call` 调用的 `aclnnAvgPool2d`；`acl_op` 的进入点是 `InnerRun` 或 `ExecFunc` 调用的 `aclopCompileAndExecute`。本节核对到的是这些主机接口，还没有定位其内部具体的设备 kernel 调用链。下面先说明 `aclnnAvgPool2d` 的函数地址从哪里取得，再转到返回方向，追踪已有的输出 `result` 怎样回到 Python。

#### libopapi.so：aclnnAvgPool2d 的接口在哪里

前面在 `acl_call` 中真正调用 CANN 的语句是：

```cpp
auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
```

`opApiFunc` 是函数指针，它指向的地址由 `GetOpApiFuncAddr("aclnnAvgPool2d")` 查到。现在补上这次查找与 `libopapi.so` 的关系。

`libopapi.so` 是 CANN 提供的算子 API 动态库。可以先把动态库理解为“已经编译好、供其他程序调用的一组函数”：torch_npu 在运行时加载它，再按函数名寻找入口地址。在这里，`op_plugin` 准备池化参数和输出，CANN 的算子接口则接收这些信息，处理对应的计算请求。CANN 的[单算子 API 文档](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/appdevgapi/aclcppdevg_03_0160.html)列出了这类接口与库的关系。

**从哪句代码开始加载这个库？** 在 [op_api_common.cpp:874](nn模块源码.md#src-opapi-symbol)，`GetOpApiFuncAddr` 有这一句：

```cpp
static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
```

先调用 [op_api_common.h:53](nn模块源码.md#src-opapi-symbol) 的 `GetOpApiLibName()`，它返回字符串 `"libopapi.so"`。这个名字再传给 `GetOpApiLibHandler`，后者在 [70 行](nn模块源码.md#src-opapi-symbol) 执行：

```cpp
auto handler = dlopen(libName, RTLD_LAZY);
```

这里的 `dlopen` 用来加载动态库，返回的 `handler` 用于后续访问这个库。此时只是取得库的访问入口，还没有执行池化。

回到 `GetOpApiFuncAddr`，[876 行](nn模块源码.md#src-opapi-symbol) 将库的 `handler` 和要找的 `apiName` 一起交给 `GetOpApiFuncAddrInLib`。进入后，在 [op_api_common.h:62](nn模块源码.md#src-opapi-symbol) 执行：

```cpp
auto funcAddr = dlsym(handler, apiName);
```

`dlsym` 的作用是按名字寻找库提供的函数。这里的名字为 `"aclnnAvgPool2d"`，找到的地址随 `return funcAddr` 返回给宏，保存为 `opApiFuncAddr`，再由 `acl_call` 转成函数指针并调用。第一段接口 `"aclnnAvgPool2dGetWorkspaceSize"` 也通过同样的查找过程取得地址。

完整的 `GetOpApiFuncAddr` 会先检查自定义库和已加载的算子库；没有找到时才继续检查 `libopapi.so`，随后还可能查询功能子库。因此，上面追的是它查询 `libopapi.so` 的这一条路径，实际命中哪个库取决于运行环境。

**这个库为本例提供什么功能？** 对平均池化，我们关心的是成对的两个接口。

`aclnnAvgPool2dGetWorkspaceSize` 接收输入、输出和窗口、步长等设置，准备本次计算所需的执行信息，返回临时空间大小和 `executor`。`result` 的描述也在这一段传入，所以输出位置已经交给 CANN。

`aclnnAvgPool2d` 接收准备好的 `executor`、临时空间和执行流，据此发起计算。它返回的是调用状态码，计算结果使用前面指定的输出存储。这样的“先准备、再执行”是 CANN 的[两段式接口约定](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/appdevg/aclcppdevg/aclcppdevg_000019.html)，宏负责把两段衔接起来。

配套教程的远端运行记录也查看了该库的动态符号，完整摘录见[符号表记录](nn模块源码.md#src-opapi-library-evidence)：

```text
00000000018aad48 T aclnnAvgPool2d
00000000018af1f8 T aclnnAvgPool2dGetWorkspaceSize
```

这说明那次运行环境中的 `libopapi.so` 提供了这两个函数；前面的十六进制数字是符号值，`T` 表示已定义的代码符号。符号表能确认接口存在，但不会展示函数内部的计算步骤。该记录来自已有的远端实验，本次没有重新运行它。

版本也会影响库的组织方式：新版 CANN 文档已改为推荐 `libopapi_nn.so` 等按领域划分的子库，见[算子库文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta3/API/aolapi/operatorlist_00001.html)。这里的 `libopapi.so` 对应我们正在阅读的源码和配套运行记录。

到这里，提交方向停在 `acl_call` 中的 `opApiFunc(...)`，即 CANN 的 `aclnnAvgPool2d` 入口。已有材料没有给出这个环境下接口内部到设备 kernel 的逐句调用链。接下来回到 torch_npu：沿池化函数的 `return result`，看输出 Tensor 怎样返回调用者。

#### 请求提交后，result 怎样回到 Python

这一段换一个追踪对象：前面看的是计算请求怎样往下传，现在看的是输出 `result` 怎样往上返回。先记住，`result` 在提交计算之前就已经创建好了，它包含输出的形状、类型以及对应的设备存储。

**先从池化辅助函数回到 op_plugin。** 在 [AvgPool2dKernelNpuOpApi.cpp:52](nn模块源码.md#src-pool-opapi-helper)，`avg_pool2d_out_npu_nocheck_opapi` 在执行宏之后写着：

```cpp
return result;
```

辅助函数返回已有的输出 `result`。外层 `op_api::avg_pool2d` 在 [134 行](nn模块源码.md#src-pool-opapi)调用它，但没有用新变量接收返回值：外层原本就持有这个输出，并在 [137 行](nn模块源码.md#src-pool-opapi)执行 `return result;`。返回引用与返回 Tensor 的语法区别见[附录](nn模块附录.md#cpp-types)。

这里不要把 `acl_call` 的返回值混进来：`return api_ret` 把整数状态码交给执行框架；`return result` 才把输出 Tensor 交给池化函数的调用者。排队时，外层还可能在回调被取出执行之前就返回 Tensor，因此返回 `result` 本身不保证设备计算已经完成。

另一条 `acl_op` 路径也返回先前创建的 `result`。如果输入原本是三维，函数先在 [AvgPool2dKernelNpu.cpp:248](nn模块源码.md#src-pool-aclop)用 `squeeze(0)` 去掉为了旧接口增加的批次维度，然后在 [250 行](nn模块源码.md#src-pool-aclop)返回输出。

现在回到生成的 `op_plugin::avg_pool2d`。在 [OpInterface_19.cpp:592](nn模块源码.md#src-opinterface-avg-pool2d)，它写的是：

```cpp
return op_api::avg_pool2d(self, kernel_size, stride, padding,
                        ceil_mode, count_include_pad, divisor_override);
```

输出由这一句返回给 `op_plugin::avg_pool2d` 的调用者；走旧路径时，[599 行](nn模块源码.md#src-opinterface-avg-pool2d)的 `return acl_op::avg_pool2d(...)` 做同样的事。嵌套调用的返回语法见[附录](nn模块附录.md#cpp-return)。

再回到 [wrapper_NPU__avg_pool2d](nn模块源码.md#src-npu-wrapper-body)，普通路径的末尾是 `return op_plugin::avg_pool2d(...)`，因此输出继续返回给调用这个 wrapper 的调度包装层。如果开启了 OpHook，wrapper 会先用 `res` 接住输出，执行 `PostHook(res)`，再 `return res;`。

这一阶段结束在 `wrapper_NPU__avg_pool2d` 的返回语句：NPU 实现已经把输出 Tensor 交回调度包装层。下一步沿这些包装层的 `return`，返回 ATen 的 C++ 接口。

**调度包装层怎样把 Tensor 交回 ATen？** 前面调用 wrapper 的函数对象，在 [WrapFunctionIntoFunctor.h:17](nn模块源码.md#src-wrap-function)写着：

```cpp
return (*FuncPtr::func_ptr())(std::forward<Parameters>(args)...);
```

其中的函数指针对应已注册的 wrapper；wrapper 返回后，这句把 Tensor 返回给外面的适配函数。适配函数在 [make_boxed_from_unboxed_functor.h:577](nn模块源码.md#src-unboxed-adapter)用 `return (*functor_)(...)` 继续返回。

再向外，是 [KernelFunction_impl.h:108](nn模块源码.md#src-call-unboxed) 的 `callUnboxedKernelFunction`，它用 `return (*func)(...)` 返回；随后 [177 行](nn模块源码.md#src-kernel-call) 的 `KernelFunction::call` 返回这个调用结果。前面按 key 选中实现的 `Dispatcher::call`，在普通路径的 [Dispatcher.h:836](nn模块源码.md#src-dispatcher-call)写着 `return kernel.template call<Return, Args...>(...)`，因此输出又回到了 Dispatcher 的调用者。

调用者是句柄的 `TypedOperatorHandle::call`，它在 [Dispatcher.h:624](nn模块源码.md#src-typed-handle-return)返回 `Dispatcher::call` 的结果。接着，`at::_ops::avg_pool2d::call` 中的 `return op.call(...)`、`at::avg_pool2d` 中的 `return at::_ops::avg_pool2d::call(...)` 逐层把输出交回 Python 绑定。两处 ATen 入口的返回写法分别来自[调用入口生成器](nn模块源码.md#src-operator-gen)和 [C++ API 生成器](nn模块源码.md#src-cpp-api-gen)。

这些 `return` 是在结束此前进入的函数调用，输出沿调用关系向外传递。需要自动微分时，还会经过对应的自动微分包装层；建立梯度关系的细节留到反向部分展开。

到这里，输出已从 `at::avg_pool2d` 返回 Python 绑定中的 `dispatch_avg_pool2d`。下一阶段从绑定里的 `wrap(...)` 开始，把 C++ Tensor 包装成 Python 能使用的对象。

**wrap 怎样把 C++ Tensor 变成 Python Tensor？** 本地没有生成的 `THPVariable_avg_pool2d`，所以这里仍以真实生成器为依据。在 [gen_python_functions.py:1406](nn模块源码.md#src-python-dispatch-return)，生成器会写出 lambda 内的 `return {dispatch_callee}(...)`，而 [1408 行](nn模块源码.md#src-python-dispatch-return)写出外面的 `return wrap(...)`。对于本例，调用关系可以简化为：

```cpp
// 按生成器说明的简化示意，不是本地生成文件的原文
return wrap(dispatch_avg_pool2d(...));
```

这句先取得 `dispatch_avg_pool2d(...)` 返回的 C++ Tensor，再交给 `wrap`。Tensor 对应的 [wrap](nn模块源码.md#src-python-wrap-tensor)调用 `THPVariable_Wrap`，下一步进入这个函数；C++ 重载的写法见[附录](nn模块附录.md#cpp-return)。

进入 [python_variable.cpp:447–452](nn模块源码.md#src-python-tensor-object)，`THPVariable_Wrap` 又把 Tensor 交给 `THPVariable_WrapWithType`。这个函数在 [418 行](nn模块源码.md#src-python-tensor-object)取得原来的 `TensorImpl`，随后通过 `get_or_init` 获取或建立与它关联的 Python 对象。需要新建时，[429 行](nn模块源码.md#src-python-tensor-object)分配 Python 对象，再在 [437 行](nn模块源码.md#src-python-tensor-object)执行：

```cpp
new (&v->cdata) Tensor(std::forward<T>(var));
```

这句把传进来的 C++ Tensor 保存到 Python 对象的 `cdata` 中。包装后仍关联原来的 TensorImpl 和设备存储，不会把池化数据复制到 CPU；在指定位置构造对象的写法见[附录](nn模块附录.md#cpp-ownership)。

最后，[444 行](nn模块源码.md#src-python-tensor-object)的 `return obj.release()` 返回 Python 对象，由 `THPVariable_Wrap → wrap → THPVariable_avg_pool2d` 逐层交回 Python 调用方。`F.avg_pool2d` 就是这个绑定函数的 Python 名称，因此它现在返回了 Python Tensor。`release()` 的资源管理含义见[附录](nn模块附录.md#cpp-ownership)。

返回方向的最后一站是 [pooling.py:779](nn模块源码.md#src-avgpool-class) 的 `AvgPool2d.forward`：原来的 `return F.avg_pool2d(...)` 把这个 Python Tensor 再返回给模块的调用者。于是用户写下的 `y = pool(x)`，最终让 `y` 接住这个输出。

这一条前向主线到此结束：提交方向进入 CANN 的 `aclnnAvgPool2d` 或 `aclopCompileAndExecute`；返回方向经 `THPVariable_Wrap` 回到 Python，并在 `AvgPool2d.forward` 返回输出。此时用户已拿到输出 Tensor，设备计算是否完成仍取决于执行与同步状态。若继续研究设备计算，从 CANN 接口内部接着追；若继续研究训练流程，则从这个输出参与的 `backward()` 开始，进入自动微分阶段。
