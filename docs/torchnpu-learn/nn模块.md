## 先讲一个简单的二维平均池化

本文用 `PT` 表示 PyTorch 源码根目录，`TN` 表示 torch_npu 仓库根目录。正文中的源码引用均跳转到独立的[源码汇编](nn模块源码.md)；实际路径与行号写在对应代码块顶部。

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

- 第一步：调用无状态函数
    - 测试调用 `nn.AvgPool2d` 类的 `forward` 方法 （[完整 AvgPool2d 类](nn模块源码.md#src-avgpool-class)）
        - 构造函数保存 `kernel_size`、`stride` 等参数；`forward` 把输入和这些参数交给下一层 functional 接口
- 第二步: 进行C binding
    - 第一步提到的计算交给的functional层为 `F.avg_pool2d`（[完整绑定赋值](nn模块源码.md#src-functional-binding)）
    - 绑定的方法是通过赋值实现的，本质是把C函数暴露成Python可调用的名字
        - `avg_pool2d = _add_docstr(torch._C._nn.avg_pool2d, docstring)`
        - Python方法 = 算子的C实现 + 注释
        - `F`相关知识
        - `_add_docstr`相关知识
- 第三步：上游如何生成 C 函数
    - Pytorch规定了一个Python侧 API 对下游 CPP 接口的要求，这是一堆yaml文件
        - 我们的`avg_pool2d`的要求存放在[avg_pool2d](nn模块源码.md#src-native-schema)
    - 当我们build pytorch的时候，代码生成器会生成对应的 C++ 接口、绑定和注册代码
        - 生成C函数一部分看不懂没关系，后面的模块会展开讲解
        - 代码生成器 codegen 会依据yaml在build期间按照职责生成六类文件
            1. **Python 绑定：**
                1. **转换Python参数格式为CPP参数，调用CPP的接口**
                2. [Python 绑定生成器 method_impl](nn模块源码.md#src-python-binding-gen)
                3. 生成结果中包含 `torch._C._nn.avg_pool2d` 对应的 `THPVariable_avg_pool2d`；当前附的是生成器，挂载结构见[完整 nn 绑定模板](nn模块源码.md#src-python-binding-template)
            2. **CPP接口**
                1. **提供CPP算子空间中的函数,给 Python 绑定调用的入口**
                2. [ATen C++ API 生成器 ComputeFunction](nn模块源码.md#src-cpp-api-gen)
            3. **调度入口**(part2)
                1. 用不准确但是易懂的方式来说，就是
                    1. **通过CPP接口找到哪一类算子会被调用，缓存下来**
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

这一部分是build时候生成的，对应了Part1第三步的a，主要涉及的函数是`THPVariable_avg_pool2d`，这里我们还是要先做一个区分，与Part1的第二步相比

- 这里做的主要是让传入参数类型转换为能被CPP处理的参数
- 后者只是一个简单的命名+包装注释

`THPVariable_avg_pool2d` 的生成机制见 [method_impl](nn模块源码.md#src-python-binding-gen)，绑定容器见[完整 nn 绑定模板](nn模块源码.md#src-python-binding-template)。

- 第一步：利用 `parser` 按生成的参数签名解析本次 Python 调用传入的参数
- 第二步：定义CPP的lambda函数，函数内容如下：
    -  释放GIL：Python 的全局锁（同一时刻只许一个线程跑 Python 代码）
    -  调用`at::avg_pool2d`，传入拆好的参数
        -  at的相关介绍请看备注
- 第三步：用wrap包装结果函数返回值

#### `at::avg_pool2d`是怎么做的

以下为帮助理解的简化示意，真实生成逻辑见 [ComputeOperators](nn模块源码.md#src-operator-gen)。

```cpp
// 简化示意，不是本地已生成文件的原文
at::Tensor avg_pool2d::call(...) {
  static auto op = create_avg_pool2d_typed_handle();  // 首次调用:按名字查表并缓存
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
    - 第一步，按照`avg_pool2d`的名字来查找算子句柄，使用`static` 变量缓存下来
        - 在pytorch中已登记算子存放的格式是一个链表，在`dispatcher`中的`std::list<OperatorDef> operators_;`
            - [Dispatcher](nn模块源码.md#src-dispatcher-class)
        - 算子句柄持有指向这条链表中 `OperatorDef` 节点的迭代器，可用它访问对应记录
            - [Dispatcher](nn模块源码.md#src-dispatcher-class)
            - 存放了`impl::OperatorEntry`和生命周期管理信息
        - 通过 `operatorLookupTable_` 哈希表快速查询算子句柄；见 [findSchemaOrThrow](nn模块源码.md#src-find-schema) 与 [findOrRegisterName_](nn模块源码.md#src-find-name)
        - 具体细节请查看备注
    - 第二步，调用句柄的`call`方法，调用 tensor 计算运行时后端的算子
    - [算子句柄与 call 生成器 ComputeOperators](nn模块源码.md#src-operator-gen)

### Part3. 如何通过dispatch key查询真正需要定位的算子

#### 定位算子的流程：

[OperatorEntry](nn模块源码.md#src-operator-entry)
书接上文，`OperatorEntry`对象有几个关键的成员

- `name_`：这个条目代表哪个算子重载，例如 `aten::avg_pool2d` 的默认重载。
- `schema_`：这个算子接收什么参数、返回什么结果等信息。
- `dispatchTable_`：各个运行时 dispatch key 对应的可调用实现数组。
- `dispatchKeyExtractor_`：帮助从本次参数及执行状态中提取调度所需的 keyset。

真正机制在后两者，我们分开聊，如果从真实的tensor计算时候调用的参数，他们配合逻辑是这样的：

- [Dispatcher::call](nn模块源码.md#src-dispatcher-call)
- 使用`dispatchKeyExtractor_`提取`dispatchKeySet`
- 利用KeySet查找dispatchTable，找到`KernelFunction`
- 把参数传给function计算

#### 按照顺序，先讲`OperatorEntry`的`dispatchKeyExtractor_`

- 在讲如何提取参数前，先讲一下需要提取的参数分为两类：boxed 和 unboxed
    - unboxed 的 `Args...` 是 C++ 模板参数包，不是名为 `Args` 的类；其中 Tensor 等受支持参数参与 keyset 提取，普通整数、布尔等参数通常不贡献 key
        - 他是用来处理编译时已知调用签名的代码，例如生成的 ATen 接口
        - `Args`的具体实现我们放在附录
    - boxed参数存放在`IValue`栈,计算位置提取Tensor
        - [DispatchKeyExtractor](nn模块源码.md#src-key-extractor)
        - 需要用统一接口调用各种不同签名算子的代码，具体细节不在此处展开，（custom_operator 入门 boxed、在 jit 看完整应用场景）
        - 读取 schema，将需要参与调度的参数位置记成位集合（不用太理解，了解即可）
- 这个类包含几种参数
    - [DispatchKeyExtractor](nn模块源码.md#src-key-extractor)
    - `dispatch_arg_indices_reverse_`：预先存储 boxed 调用：参数栈里哪些位置需要检查
    - 其余三个字段：为 boxed/unboxed 调用记录：哪些层可以跳过，以及是否需要按后端区分
        - `nonFallthroughKeys_`、`nonFallthroughKeysPerBackend_`数组
        - `bool requiresBitsetPerBackend_`
        - 他们的用处会在后续详细讲解

**具体如何提取参数的运行步骤**

- 第一步：从参数中收集key
    - [DispatchKeyExtractor](nn模块源码.md#src-key-extractor) `getDispatchKeySetUnboxed`
    - 子第一步：用 `MultiDispatchKeySet` 收集参数中的 key，入口见 [multi_dispatch_key_set](nn模块源码.md#src-multi-key-fn)
        - 此处讲解的是较为简单的unbox方法
        - [MultiDispatchKeySet](nn模块源码.md#src-multi-key)
        - 类中有一个 `DispatchKeySet ts` 成员和对应的 `operator()` 重载处理参数,使用并集迭代收集所有tensor的keyset
        - `ts = ts | x.key_set();`
            - unboxed 方法通过重载处理各参数，合并参与调度的 Tensor 等对象的 `key_set()`
        - 此处的原始 keyset 主要来自输入张量 self。池化窗口大小会影响后续计算，但不会在这一步直接决定 CPU 或 NPU
        - 多张量算子则会合并多个输入的 keyset, 设备兼容性还需要相应实现检查
    - 子第二步：结合当前线程状态，并去掉可以跳过的层
        - [DispatchKeyExtractor](nn模块源码.md#src-key-extractor)
        - 利用我们前面得到的`MultiDispatchKeySet`类，调用`getDispatchKeySetFromRawDispatchKeySet`
        - subsubstep1：按照当前线程状态增加、删减 keyset，见 [computeDispatchKeySet](nn模块源码.md#src-compute-key)
            - `auto dispatch_keys = (ks | tls.included_) - tls.excluded_;`
            - tls: Thread Local Storage，线程局部存储,参考附录
        - subsubstep2:返回`dispatch_keys & nonFallthroughKeys_ & key_mask`
            - `nonFallthroughKeys_`:
                - 过滤已知的 fallthrough 层
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
- 在何处填入了这些地址？：`registerImpl()`
    - [Dispatcher::registerImpl](nn模块源码.md#src-register-impl)
    - 他传入的参数有三个
        - 给哪个opname注册、
        - 注册在哪个 key 下、
        - 注册哪个kernelfunction？
    - 随后两步：（细节会在后面讲）
        - step1:`findOrRegisterName_`,根据算子名字找到已有记录，或者创建记录
        - step2：`op.operatorDef_->op.registerKernel` 保存该算子的 kernel 注册记录
            - `registerKernel()` 先保存注册记录，再更新调度表。
    - 最终填数组
        - 代码在[OperatorEntry::updateDispatchTableEntry_](nn模块源码.md#src-update-entry)
        - 利用`getDispatchTableIndexForDispatchKey`找到`dispatchkey`对应的数组下标`dispatch_ix`
        - 使用`computeDispatchTableEntry`计算`dispatcher, dispatch_key`对应的`KernelFunction`填入数组
            - 此处为什么要计算，而非直接用之前注册的kernel function，请参考附录
    - 关于如何使用`dispatchkey`变成数组下标，查到对应kernel，也请参考附录
    - `dispatch_ix == -1` 表示 key 没有可用的表项下标；它与“有槽位但未注册实现”不同，细节留到后续讨论
    - 重点：拿到数组元素以后，怎样真正调用函数？

### Part4. 如何通过privateuse1挂载，调用npu算子

两个框架交接的地方，如何从pytorch的`KernelFunction` 内部的 `unboxed_kernel_func_`到torchnpu的`NPU wrapper`

- [RegisterNPU.cpp 对应的生成逻辑](nn模块源码.md#src-npu-wrapper-gen)：该文件由构建生成，本地尚不存在；文件组装与注册块见 [gen_dispatcher_registrations](nn模块源码.md#src-npu-register-gen)。

#### 注册链条

这里要反着看，从已经在torchnpu包装好的`avg_pool2d`算子（NPU wrapper）讲起，他是怎么一步步挂载到
`unboxed_kernel_func_`的
下面是注册形状示意，`W` 代表 wrapper；完整生成器见 [NPU 注册文件生成器](nn模块源码.md#src-npu-register-gen)。

```cpp
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("avg_pool2d", TORCH_FN(W));
}
```

- 我们的需要注册的wrapper是`wrapper_NPU__avg_pool2d`,他的函数定义是torchnpugen在构建阶段生成的
- `TORCH_LIBRARY_IMPL`这个宏定义帮助我们注册了他（这个宏定义的具体机制看附录）
    - [TORCH_LIBRARY_IMPL](nn模块源码.md#src-library-macro)、[_TORCH_LIBRARY_IMPL](nn模块源码.md#src-library-macro-expand)
    - 生成：一个包含上述注册语句的初始化函数。一个静态 `TorchLibraryInit` 对象。
    - 加载相关动态库、执行静态对象初始化时，会调用 `TorchLibraryInit` 的构造函数
        - [TorchLibraryInit](nn模块源码.md#src-library-init)
        - 这里关键是一个`fn(lib_)`
        - `lib_` 已携带 aten 和 `PrivateUse1`
        - fn是一个执行注册的函数，执行`m.impl(...)`，具体内容参考前面的宏定义注册附录
- 在构建完成后，我们就会拥有一个已经定义编译好的NPU wrapper和`TorchLibraryInit`注册初始化函数，下一步进入运行阶段的import，正式开始注册
    - [TORCH_FN_TYPE](nn模块源码.md#src-torch-fn-type)、[TORCH_FN](nn模块源码.md#src-torch-fn)、[CompileTimeFunctionPointer](nn模块源码.md#src-compile-pointer)
    - 这里定义了`TORCH_FN` 和 `TORCH_FN_TYPE`：后者表示绑定函数的类型，前者创建该类型的对象,我们通俗的理解一下`TORCH_FN(W)`干了什么：

        ```cpp
        // 定义一个类型：它绑定了 add_one 这个函数。
        using AddOneToken =
           c10::CompileTimeFunctionPointer<int(int), &add_one>;
        // 创建这个类型的对象。
        AddOneToken token;
        ```

        - 初始化这个struct生成的对象：`CompileTimeFunctionPointer` 长什么样
            - `FuncType_` = W 的函数类型
            - `func_ptr_` = W 的函数地址
            - 提供返回`func_ptr_`的函数
        - 这个`TORCH_FN(W)`就是后续的`raw_f`
- impl定义在[Library::impl](nn模块源码.md#src-library-impl)，我们考虑非 mobile 分支
    - 他可以分为几步
        - `CppFunction f(std::forward<Func>(raw_f));`
            - `raw_f`就是`TORCH_FN(W)`
            - `CppFunction`定义了一个变量f，`f.func_` 将使用`std::forward<Func>(raw_f)`让自己变成一个能调用 W 的 `KernelFunction`
        - `_impl`

**`CppFunction`**

- 先讲`CppFunction` 构造函数
    - 类定义在[CppFunction](nn模块源码.md#src-cppfunction)，非mobile分支构造函数在[CppFunction](nn模块源码.md#src-cppfunction)
    - 类有很多成员：
        - `func_`：                      一个 `c10::KernelFunction` 成员
        - `cpp_signature_`：             C++ 函数签名信息
        - `schema_`：                    schema 信息
    - 构造函数对应最重要的是初始化`func_(c10::KernelFunction::makeFromUnboxedFunction(f))`
        - 这里的f指的就是我们前面的`std::forward<Func>(raw_f)`
- `makeFromUnboxedFunction`
    - 定义在：[KernelFunction::makeFromUnboxedFunction](nn模块源码.md#src-make-function),（尖括号指定编译期模板参数，圆括号传入实际函数参数）
    - 非 mobile 分支代码在[KernelFunction::makeFromUnboxedFunction](nn模块源码.md#src-make-function)
    - 这部分比较复杂，我们结合代码讲讲：
        - `AllowLegacyTypes`
            - 一个不重要的参数，定义把他定义为false
        - `typename impl::WrapFunctionIntoFunctor<FuncPtr>::type`:
            - 整个表达式没有创建对象，也没有调用 wrapper
            - 他把 `FuncPtr` 交给 `WrapFunctionIntoFunctor` 这个模板确定类型，取实例内部名为 type 的类型别名`ConcreteFunctor`
            - 这里的`typename`：告诉编译器，“这里是一个类型名”
            - 这个 type 指向带有 `operator()` 的 `WrapFunctionIntoFunctor_` 类
        - `detail::make_unique_base<OperatorKernel, ConcreteFunctor>()`
            - `make_unique_base<以什么基类持有, 实际创建什么类>()`
            - 创建 `ConcreteFunctor` 对象，返回管理它的 `unique_ptr<OperatorKernel>`
            - 通俗的理解，就是把我们把npu wrapper包装好的对象的指针传出去，调用这个对象，他的operator就会调用npu wrapper
            - `detail::make_unique_base`位于：[make_unique_base](nn模块源码.md#src-make-unique)
        - 整体：

            ```cpp
            makeFromUnboxedFunctor<
               AllowLegacyTypes,    // 编译期设置
               ConcreteFunctor     // 编译期确定的具体类类型
            >(
               std::move(object)   // 运行时传入创建好的对象，转移所有权
            );
            ```

            - 尖括号告诉编译器生成哪个版本；圆括号给这个版本传实际参数。
    - `WrapFunctionIntoFunctor`
        - [WrapFunctionIntoFunctor](nn模块源码.md#src-wrap-type)
        - 给他 `FuncPtr`，他从中取得函数的返回类型、参数类型列表，再把这些信息交给 `WrapFunctionIntoFunctor_`，把得到的类类型命名为 type
        - [WrapFunctionIntoFunctor_ 的完整偏特化](nn模块源码.md#src-wrap-function) 中，`operator()` 返回 `(*FuncPtr::func_ptr())(std::forward<Parameters>(args)...)`
            - `FuncPtr::func_ptr()`取得 `&npu_wrapper`
    - `makeFromUnboxedFunctor`（[完整函数](nn模块源码.md#src-make-functor)）
        - 模板参数 `KernelFunctor` 指定具体类；函数参数 `kernelFunctor` 是管理该类对象的 `unique_ptr<OperatorKernel>`
        - 它取得 unboxed 与 boxed 适配函数的地址，把它们与 functor 对象一起装入 `KernelFunction`；完整实现见下方源码引用
        - 构造函数在[KernelFunction](nn模块源码.md#src-kernel-ctor)
        - unboxed 入口指向 `wrap_kernel_functor_unboxed_::call`；该静态函数调用具体 functor 的 `operator()`，后者再通过 `FuncPtr::func_ptr()` 调用 wrapper
- 现在我们就可以说，通过一系列操作，我们把`TORCH_FN(W)`变成了`KernelFunction`放在了`CppFunction::func_`

**`_impl`**

- [Library::_impl](nn模块源码.md#src-library-_impl)
- `c10::Dispatcher::singleton().registerImpl()`
    - 这就接到我们上面提到的查找算子，随后注册的的故事了
    - 他首先通过`findOrRegisterName_`找到`OperatorEntry`
    - `registerKernel`操作该算子的 `kernels_`，然后更新 `dispatchTable_`
        - 把 `KernelFunction` 放入 `kernels_` 的注册记录
            - `kernels_`保存各个 dispatch key 下的注册记录
            - [OperatorEntry::registerKernel](nn模块源码.md#src-register-kernel)
            - 取得`kernels_`容器，把新记录放在最前面
        - 随后`updateDispatchTable_(dispatcher, *dispatch_key);`
            - [OperatorEntry::updateDispatchTable_](nn模块源码.md#src-update-table)
            - 它先通过 [getRuntimeDispatchKeySet](nn模块源码.md#src-runtime-keys) 展开受影响的 key，再逐个调用 `updateDispatchTableEntry_`。对于 `PrivateUse1`，先更新自身槽位，再更新 `AutogradPrivateUse1` 槽位；后者是重新选择实现，不是复制同一个 NPU wrapper
                - 第一步：把 key 映射成数组下标。
                - 第二步：决定该槽位该用哪个 kernel，然后赋值到dispatchtable。
                    - 这里涉及了 `computeDispatchTableEntry` 函数；直接注册了 `PrivateUse1` 实现时，使用对应注册记录
                        - [OperatorEntry::updateDispatchTableEntry_](nn模块源码.md#src-update-entry)
                        - 它会利用 [getKernelForDispatchKey](nn模块源码.md#src-get-kernel) 取出 `kernels_` 的最新注册记录；完整选择过程见 [computeDispatchTableEntry](nn模块源码.md#src-compute-entry) 与 [computeDispatchTableEntryWithDebug](nn模块源码.md#src-compute-entry-debug)
                    - 如果涉及fallback，我们暂时不讨论

```text
构建阶段
    Python 代码生成器生成 C++：
        · NPU wrapper 的函数定义
        · TORCH_LIBRARY_IMPL 注册代码

    C++ 编译、链接：
        · wrapper 成为编译后的函数代码
        · 注册初始化函数也成为编译后的函数代码
        · 生成包含这些代码的库

运行阶段
    加载相关动态库，执行静态对象初始化：
        · 构造 registration_object
        · 构造函数执行 fn(lib_)
        · 即 register_npu_ops(lib_)
        · 执行 m.impl(..., TORCH_FN(npu_wrapper))
        · 把 wrapper 接入 Dispatcher

后续算子调用
    Dispatcher 选中对应 kernel
        · 经过之前讨论的适配函数
        · 真正执行 NPU wrapper
```

#### 调用链条

```text
① KernelFunction::call
   将 unboxed_kernel_func_ 转成函数指针并调用
        ↓
② unboxed 适配器
   wrap_kernel_functor_unboxed_::call(...)
        ↓
③ 函数对象适配器
   WrapFunctionIntoFunctor_::operator()(...)
        │
        ├── ④ 编译期函数指针封装
        │      func_ptr() 返回 NPU wrapper 的地址
        │
        └── 调用取得的地址
               ↓
⑤ NPU wrapper（构建阶段生成）
   wrapper_NPU__avg_pool2d(...)
```

- step1: 调用kernelfuntion的call功能：
    - [KernelFunction::call](nn模块源码.md#src-kernel-call)：调用 `unboxed_kernel_func_` 保存的地址
    - [KernelFunction::call](nn模块源码.md#src-kernel-call)
    - `functor = boxed_kernel_func_.getFunctor()`，见 [getFunctor 完整函数](nn模块源码.md#src-boxed-get-functor)
    - `callUnboxedKernelFunction`
- step2 `callUnboxedKernelFunction`
    - [callUnboxedKernelFunction](nn模块源码.md#src-call-unboxed)
    - 这个地方把void类型的`unboxed_kernel_func`变成一个确定类型的func去执行对应参数的运算
- step3 func是什么
    - [KernelFunction::makeFromUnboxedFunctor](nn模块源码.md#src-make-functor)点明了`unboxed_fn`是指向`wrap_kernel_functor_unboxed`的一个call方法
    - 这个struct的call方法在[wrap_kernel_functor_unboxed_](nn模块源码.md#src-unboxed-adapter)
    - 用处是把基类指针转回具体类指针，对象本身早已存在
    - 将 `OperatorKernel*` 转回实际的 `KernelFunctor`*。
    - `std::forward` 只负责转发参数、保留相应的值类别
    - 相当于`functor_->operator()(std::forward<ParameterTypes>(args)...);`
    - 调用该对象的 `operator()`
    - 当前这个适配器分支不向 wrapper 转发 `DispatchKeySet`
    - `KernelFunctor` 是模板类型参数，不是在这里定义的某个固定类
