# nn 模块附录

[返回调用链正文](nn模块.md)。这里集中解释阅读正文需要的 Python、C++ 和调度基础；真实函数与完整模板继续放在[源码汇编](nn模块源码.md)。下面的小例子用于说明语法，不是仓库中生成文件的原文。

可以按遇到的问题跳转：

- Python 入口：[F](#f)、[_add_docstr](#_add_docstr)、[绑定如何挂载](#python-binding-mount)、[Python Tensor 如何交给 C++](#python-tensor-to-cpp)。
- 调度基础：[线程局部状态](#tls)、[算子句柄](#operator-handle)、[有效 kernel 的选择](#effective-kernel)、[key 怎样变成数组下标](#dispatch-table-index)。
- C++ 阅读：[名字与访问](#cpp-names)、[类型、引用与指针](#cpp-types)、[模板](#cpp-templates)、[函数对象](#cpp-functors)、[static 与匿名命名空间](#cpp-static)、[lambda 与回调](#cpp-lambda)、[对象和资源的转移](#cpp-ownership)、[宏](#cpp-macros)、[自动恢复与 GIL](#cpp-guards)、[返回值](#cpp-return)、[动态库](#cpp-dynamic-library)。
- 注册细节：[宏怎样开始注册](#registration-start)、[wrapper 怎样装入 KernelFunction](#registration-wrapper)。
- NPU 执行：[JIT](#jit)、[队列、缓存与等待](#npu-execution-details)。

## F

`F` 是代码通过 `import torch.nn.functional as F` 取的简称。它指向 `torch.nn.functional` 模块，因此 `F.avg_pool2d` 就是这个模块中的 `avg_pool2d`。

`nn.AvgPool2d` 是一个类：创建对象时保存窗口、步长等配置，调用 `forward` 时读取它们。functional 接口接收本次计算需要的输入和设置。在这个例子中，两者最后使用同一个池化入口，见 [AvgPool2d.forward](nn模块源码.md#src-avgpool-class) 和 [functional 绑定](nn模块源码.md#src-functional-binding)。

`F` 不是 PyTorch 强制规定的全局名字。另一个 Python 文件若也想写 `F.avg_pool2d`，需要自己导入并使用这个别名。

<a id="_add_docstr"></a>
## _add_docstr 的作用

`_add_docstr` 给已有对象添加文档说明。对于本例：

```python
# 简化示意，完整赋值见源码汇编
avg_pool2d = _add_docstr(torch._C._nn.avg_pool2d, docstring)
```

先前已经有 `torch._C._nn.avg_pool2d` 这个绑定函数，`_add_docstr` 给它添加文档，再返回同一个对象。赋值以后，functional 模块中的 `avg_pool2d` 名字也指向它。它没有另建一套池化计算，也没有在调用时增加一个 Python 池化函数层。

本地实现为 [THPModule_addDocStr](nn模块源码.md#src-add-docstr)：它设置对象的文档字段，最后用 `Py_NewRef(obj)` 返回原对象。这个写法增加 Python 引用计数，不是复制函数。已有文档或不支持的对象类型会进入对应的报错分支。

<a id="python-binding-mount"></a>
## torch._C._nn.avg_pool2d 如何挂载 THPVariable_avg_pool2d

这里有两个名字：Python 用 `avg_pool2d` 调用；C++ 绑定函数叫 `THPVariable_avg_pool2d`。它们靠 Python 扩展模块的方法表连接。

构建时，[get_pycname 与 method_def](nn模块源码.md#src-python-method-entry) 根据算子名生成 C++ 函数名和方法表条目。条目记录“Python 名称、C++ 函数入口、参数接收方式”等信息。对本例，可以把生成的关系理解为：

```text
Python 名字 "avg_pool2d" → C++ 入口 THPVariable_avg_pool2d
```

[python_nn_functions.cpp 模板](nn模块源码.md#src-python-binding-template)把这些条目放进 `nn_functions`。初始化函数 `initNNFunctions` 用这张表创建 `torch._C._nn` 模块，并通过 `PyModule_AddObject(module, "_nn", nn)` 把它挂到上层模块。

所以，Python 读取 `torch._C._nn.avg_pool2d` 时，取得的是方法表登记的函数。当前本地没有已生成的 `python_nn_functions.cpp`，这里依据生成器和完整模板说明挂载过程；不是把模板当成已经生成的文件。

<a id="python-tensor-to-cpp"></a>
## Args 怎样取得拥有 key_set() 的 Tensor

schema 是参数协议：它说明某个位置应当接收 Tensor。它不会把一段声明文字变成真实 Tensor，也不会在调用时凭空添加 `key_set()`。

用户传入的 Python Tensor 已经包装着 C++ Tensor。参数解析器先按协议检查本次输入，再通过 `_r.tensor(0)` 这样的调用取得第一个 Tensor 参数。

在 [PythonArgs::tensor](nn模块源码.md#src-python-tensor-unpack)中，普通 Tensor 的快速路径调用 `THPVariable_Unpack(args[i])`。后者取出 Python 对象中保存的 `cdata`，也就是原来的 C++ Tensor；特殊情况由 `tensor_slow` 处理。

C++ Tensor 关联着 TensorImpl，调度标记属于这套已有的张量信息。后面 `x.key_set()` 读取的是张量携带的标记。输入绑定和输出包装正好对应：输入用 `Unpack` 取出 C++ Tensor，输出用 [THPVariable_Wrap](nn模块源码.md#src-python-tensor-object)提供 Python 对象。

正文中的 `Args...` 表示 C++ 调用的参数类型集合；实际的 `args...` 是已经取出的 Tensor、整数列表、布尔值等参数。只有适用的 Tensor 等参数参与 key 收集，普通窗口整数不贡献设备 key。

这里还要区分两种传参方式。unboxed 入口按已知的 C++ 函数签名接收 Tensor、整数等参数；boxed 入口则把值放入统一的 `IValue` 栈，通过相同的容器处理不同签名。`IValue` 可以承载多种值，提取器按预先记录的 Tensor 参数位置读取其中的 Tensor。它们是调度调用的两种表示，并不是把 NPU 数据装箱或复制一次，见 [DispatchKeyExtractor](nn模块源码.md#src-key-extractor)。

<a id="operator-handle"></a>
## 算子句柄缓存的是什么

`findSchemaOrThrow("aten::avg_pool2d", "")` 找到的是一个算子条目，见[完整函数](nn模块源码.md#src-find-schema)。句柄使调用方能够继续访问这条记录；它不是某次计算的结果，也不是固定的 CPU 或 NPU 入口。

[Dispatcher](nn模块源码.md#src-dispatcher-class)用 `std::list<OperatorDef>` 保存算子定义，用查询表按名字找到对应记录。`OperatorHandle` 持有指向记录的迭代器，可以理解为“在已登记的算子记录中定位这一条”，无需复制整份定义。记录中包含 schema、提取器和调度表。

`static auto op = ...` 缓存这个句柄后，后续调用省去按名字查找，但每次 `op.call(...)` 仍然根据本次 Tensor 的 keyset 选用实现。句柄缓存与 NPU 执行器缓存、JIT 编译缓存属于不同阶段。

<a id="tls"></a>
## TLS：线程局部存储

线程可以先理解为同一个程序中的一条执行流程。TLS 是 Thread Local Storage，意思是某些状态按线程分别保存：一条线程临时修改自己的状态，不等于同时改掉其他线程的状态。

PyTorch 调度除了读取 Tensor 上的 key，还要考虑当前线程启用或暂时排除哪些调度层。比如某层已经处理完当前算子，继续调用下层时，需要避免再次进入自己。

[DispatchKeyExtractor](nn模块源码.md#src-key-extractor)中可以看到：

```cpp
auto dispatch_keys = (ks | tls.included_) - tls.excluded_;
```

这里先合并张量的标记与当前线程要求加入的标记，再移除要求排除的标记。后续还会与 fallthrough 掩码及 `key_mask` 取交集，跳过本次不需要处理的层。

这几个符号是在操作 DispatchKeySet：`|` 合并集合，`-` 移除集合中的元素，`&` 保留交集。不要按普通整数的加减来理解这段调度逻辑。TLS 也不是把任务保存到 NPU 队列的地方。

<a id="effective-kernel"></a>
## KernelFunction 为什么要“计算”，而不是直接赋值刚注册的 kernel

`kernels_` 保存注册记录，`dispatchTable_` 保存运行时可直接选用的有效入口。某个运行时 key 的入口可能来自直接注册，也可能来自适用的复合实现或后端 fallback；还可能根本没有可用实现。

因此，更新表项时，需要根据当前所有相关注册记录判断“这个槽位现在应该用谁”。这里的“计算”是选择实现，不是运行池化，也不是当场编译一个设备程序。

完整选择规则在 [computeDispatchTableEntryWithDebug](nn模块源码.md#src-compute-entry-debug)中。对本例的 `PrivateUse1`，存在直接注册的 wrapper 时使用对应记录。更新其自动微分槽位时则会重新判断该槽位的有效入口，不能把它理解为直接复制同一个 NPU wrapper。

<a id="dispatch-table-index"></a>
## dispatch key 怎样变成调度表数组下标

注册时，[updateDispatchTableEntry_](nn模块源码.md#src-update-entry)先用 `getDispatchTableIndexForDispatchKey(dispatch_key)` 得到 `dispatch_ix`，再把选好的实现写入 `dispatchTable_[dispatch_ix]`。

调用时，[OperatorEntry::lookup](nn模块源码.md#src-operator-entry)则从本次 keyset 得到下标，再读取对应元素。两边使用一致的映射，才能让运行时读到先前写入的入口。

普通构建中的[下标计算函数](nn模块源码.md#src-dispatch-table-index)先定位最高优先级的功能部分，再根据后端部分得到位置，最后返回 `offset_and_mask.offset + backend_idx`。因此它不只是把 `PrivateUse1` 的枚举数值直接当成数组下标，也不需要每次遍历整张表寻找字符串 `"npu"`。

源码的移动端裁剪分支使用较小的映射表，未保留的 key 可以返回 `-1`。这表示没有对应的表项位置；“位置存在，但尚无可用实现”是另一种情况，会在查到表项之后处理。

<a id="cpp-names"></a>
## C++ 的名字、命名空间与成员访问

`at::avg_pool2d` 中的 `::` 用来限定名字属于哪里。`at` 是 ATen 使用的命名空间，`op_plugin`、`op_api`、`acl_op` 也是本文遇到的命名空间。`OpCommand::RunOpApiV2` 则表示 `OpCommand` 类中的方法；看到 `::`，不能一律认为左边都是命名空间。

`namespace op_plugin { ... }` 把里面的声明放进这个命名空间。它组织名字，不会因为多包了一层大括号，就在运行时增加一层函数调用。

对象用点访问成员，比如 `cmd.Run()`；通过对象指针访问成员时通常写箭头，比如 `aclCmd->Run()`。这里的 `this` 指向正在执行成员方法的当前对象，因此 `this->customHandler` 就是当前任务对象保存的回调。

`std::` 表示 C++ 标准库中的名字，本文的 `std::optional`、`std::function`、`std::move` 和 `std::thread` 分别涉及可选值、可调用对象、移动语义和线程。

<a id="cpp-types"></a>
## 类型、auto、引用与指针

`at::Tensor result` 声明一个叫 `result` 的 C++ Tensor。`auto result = ...` 则让编译器根据右边推导类型，类型仍在编译时确定，并不是 Python 那样可以随时换成另一种类型。

`const at::Tensor& self` 中，`&` 表示引用，可以先理解为通过另一个名字访问传进来的对象；`const` 限制通过这个引用进行修改。对于 Tensor，复制 C++ 包装对象也通常共享底层存储，不能把“传引用”和“避免复制整块 NPU 数据”简单画成等号。

函数返回类型 `at::Tensor&` 表示返回对象的引用。池化辅助函数的 `return result` 返回的是已有输出；外层返回类型 `at::Tensor` 则按值返回 Tensor 包装对象，并不因此把设备数组复制到 CPU。

`ExecuteParas* p` 中的 `*` 表示指针，保存对象的位置。表达式 `&func` 取得 `func` 的地址；`*p` 访问指针指向的对象。同一个 `&` 出现在类型声明、取地址表达式或集合交集中，含义要结合位置判断。

`std::optional<Device>` 表示“可能有设备值，也可能没有”。`std::nullopt` 表示当前没有值；它与表示空指针的 `nullptr` 不是同一种类型。wrapper 初始还没有记录共同设备，所以把 `common_device` 设为 `std::nullopt`。

对可选值，`has_value()` 检查是否有值，`value()` 取得已保存的值。例如池化除数没有提供时，代码使用默认处理；提供了才读取 `divisor_override.value()`。

`at::IntArrayRef` 用于向 C++ 函数传递整数列表，例如窗口和步长。`int64_t` 是整数类型，`bool` 表示真假。`(void)common_device` 是把表达式转为 `void` 来抑制未使用变量的编译警告，不会执行设备切换。

<a id="cpp-templates"></a>
## 模板、尖括号、typename 与参数包

模板让编译器根据类型或编译期设置生成合适的代码。可以先把 `SomeFunction<T>(x)` 分成两部分：尖括号说明要使用什么类型的版本，圆括号给这个版本传入实际参数。

例如正文注册代码中的简化关系：

```cpp
// 语法示意
makeFromUnboxedFunctor<AllowLegacyTypes, ConcreteFunctor>(std::move(object));
```

`AllowLegacyTypes` 是编译期设置，`ConcreteFunctor` 是具体的函数对象类型，`object` 是运行时已经创建的对象。决定类型与创建对象是两件事。

`using ConcreteFunctor = ...;` 给类型取别名。`typename impl::WrapFunctionIntoFunctor<FuncPtr>::type` 指的是模板实例中名为 `type` 的类型；`typename` 告诉编译器这里需要按类型名理解。这个表达式本身没有创建对象，也没有调用 wrapper。

`Args...`、`ParameterTypes...` 是模板参数包，表示一组类型；`args...` 表示对应的一组实参。源码中的这些省略号是 C++ 语法，不是笔记删掉了代码。

`std::forward<Args>(args)...` 将这组参数继续转交给下一层，并保留相应的引用和值类别。这是避免包装层改变传参方式的写法，不负责选择 NPU，也不负责搬运设备数据。生成模板中的 `{name}`、`${py_methods}` 等则是生成器占位符，构建时会被替换，不能当成最终 C++ 程序中的参数包。

`kernel.template call<Return, Args...>(...)` 中间的 `template` 则提示编译器：后面的 `call` 是模板成员调用。阅读主线时，把它理解为“调用 kernel 的 call 方法，并指定返回值和参数类型”即可。

<a id="cpp-functors"></a>
## class、struct、构造函数与 operator()

类把数据和处理这些数据的方法组织在一起。本文中 `OpCommand` 保存执行请求，`KernelFunction` 保存调用入口和包装对象。C++ 的 `class` 与 `struct` 都可以包含成员和方法，它们的默认访问权限不同；正文无需靠这个区别理解调用链。

创建对象时执行的函数叫构造函数。例如 `CppFunction f(raw_f)` 创建 `f`，同时调用适用的构造函数；构造函数中的 `func_(...)` 是初始化成员 `func_`，不是重新声明一个函数。

有的类定义了 `operator()`，于是对象也可以像函数一样被调用：`functor(args...)` 会进入它的 `operator()`。这类对象常叫 functor，中文可以理解为“能被调用的对象”。

[WrapFunctionIntoFunctor_](nn模块源码.md#src-wrap-function)就是本文的例子：它的 `operator()` 取得注册 wrapper 的地址，再调用 wrapper。包装对象在注册时创建，运行调用时使用它，不是在每次池化时重新定义一个类。

一个静态成员方法可以通过 `Class::method()` 调用，无需绑定某个实例。例如 `OpCommand::RunOpApiV2` 是统一的提交入口；普通的 `cmd.Run()` 则使用 `cmd` 保存的请求信息。`static_cast<KernelFunctor*>(functor)` 在适配器中把指针还原为实际对象类型，转换本身不会新建一个 functor。

<a id="cpp-static"></a>
## static 与匿名命名空间

函数内部的 `static auto op = ...` 只在控制流第一次成功到达声明并完成初始化时建立，后续调用继续使用它。本文用这个方式缓存算子句柄或函数地址；它不会缓存某次池化的输出。

函数外部的静态注册对象则具有静态存储期，相关库加载并进行初始化时会执行其构造函数。它与“第一次调用池化时才初始化局部 static”属于不同位置的用法。

类中的静态方法不依赖某一个对象实例，见前一节。`inline` 则主要涉及函数定义的组织与编译规则，并不保证编译器一定把函数体展开到调用位置；函数是否 inline 不会改变本文的调用语义。

wrapper 外面的 `namespace { ... }` 是匿名命名空间。其中的函数名在当前翻译单元内部使用；这里可以把翻译单元理解为这个 `.cpp` 连同它包含的头文件。这样可以避免与其他文件中同名的内部函数发生链接冲突。

同一翻译单元里的注册代码仍能引用 wrapper，把可调用入口接入 Dispatcher。之后 Dispatcher 使用已注册的入口调用它，不要求别的文件能直接按名字查到这个内部函数。匿名命名空间不是类，也不会额外执行一层逻辑。

<a id="cpp-lambda"></a>
## lambda、捕获与回调

lambda 是在当前位置定义的小函数，可以保存到变量里，再传给别的函数。正文的 `acl_call` 用它保存准备好的 CANN 调用。先看一个独立的语法例子：

```cpp
// 语法示意
int number = 3;
auto task = [number]() -> int {
    return number + 1;
};
int answer = task();
```

`auto task = ...` 保存函数对象；方括号 `[number]` 保存外部变量的值，称为捕获；圆括号 `()` 说明调用时不再接收参数；`-> int` 说明返回整数；大括号里是实际执行的代码。

执行定义语句时，`task` 已保存需要的值，但函数体还没有运行。最后的 `task()` 才执行函数体。正文因此可以先建立 `acl_call`，再把它放进队列，稍后由 `customHandler()` 调用。

按值捕获一个地址，只保存地址值，不复制地址指向的整块设备内存。`acl_call` 保存的 `workspace_addr` 和 `executor` 属于这种情况；`converted_params` 则保存转换后的参数集合。

`[&]` 表示默认按引用访问使用到的外部变量。正文 `InnerRunOpApi` 交给 `RunWithDeterministicSnapshot` 的另一段 lambda 使用这种写法；其中再调用原来的 `func()`。这是包住原回调的另一个函数，不能把两者当成同一个变量。

“回调”描述的是用途：先把可调用的对象交给另一段代码，由那段代码决定何时调用。本文的 `PROC_FUNC` 定义为 `std::function<int()>`，它能保存“无参数、返回整数”的可调用对象，包括 `acl_call`。`func` 和队列中的 `customHandler` 可以使用这个容器保存同一次调用。

<a id="cpp-ownership"></a>
## std::move、unique_ptr、release 与已有内存

`std::move(x)` 本身不执行复制或移动，它让后续操作可以使用移动语义。在队列保存回调时，`this->customHandler = std::move(*(other.customHandler))` 由赋值操作把回调内容转交给目标任务对象；这里转移的是主机侧回调对象，不是搬动 NPU Tensor 的数据。

`unique_ptr<T>` 管理一个对象的生存期：它负责在适当时机销毁所管理的对象，通常由一个拥有者持有。注册包装代码创建具体 functor，再通过 `unique_ptr<OperatorKernel>` 管理它。实际对象仍是具体 functor，不会因为用基类指针保存就变成另一个基类对象。

返回 Python 对象时，`THPObjectPtr` 也负责管理引用。`obj.release()` 把它管理的 Python 对象交给调用者，不再由这个局部管理器负责释放；它不是删除对象，更不是释放 NPU 输出数据。

[python_variable.cpp:437](nn模块源码.md#src-python-tensor-object)中的 `new (&v->cdata) Tensor(std::forward<T>(var))` 是 placement new：在指定位置构造一个 Tensor 包装对象。这里的位置是 Python 对象内部的 `cdata`。它仍关联原来的 TensorImpl 与存储，所以这次构造不等于复制整块池化数据。

<a id="cpp-macros"></a>
## 宏、字符串拼接与条件编译

宏由 C++ 预处理器展开，通常发生在真正编译之前。`EXEC_NPU_CMD(...)` 虽然写起来像函数调用，但构建时会按定义展开出代码；展开出的代码才在运行时准备和提交计算。

宏中的 `#aclnn_api` 把传进来的名字变成字符串。本例传入 `aclnnAvgPool2d`，因此得到 `"aclnnAvgPool2d"`。相邻字符串会连接，`#aclnn_api "GetWorkspaceSize"` 就得到 `"aclnnAvgPool2dGetWorkspaceSize"`。这也是兼容性检查能查询第一段接口的原因，见 [DO_COMPATIBILITY](nn模块源码.md#src-pool-compatibility)。

`__VA_ARGS__` 表示可变参数宏中收到的其余参数。本例就是算子名后面的输入、窗口、步长、填充、开关和输出。它与 C++ 模板的 `Args...` 是两种不同层次的语法。

`##` 在注册宏中用于拼接名字，帮助生成不会互相冲突的初始化函数和对象名；`TORCH_LIBRARY_IMPL` 的完整展开见[注册宏源码](nn模块源码.md#src-library-macro-expand)。宏定义末尾的反斜杠表示下一行仍属于同一个宏。

`#ifndef BUILD_LIBTORCH ... #endif` 根据构建标记决定这段代码是否被编译进去。普通 `if (...)` 则在运行时根据当前条件决定执行哪一支。`C10_UNLIKELY(condition)` 给编译器分支预测提示，不改变条件本身的真假。

<a id="cpp-guards"></a>
## Guard、构造与析构：为什么退出函数会自动恢复

对象创建时执行构造函数，结束生存期时执行析构函数。C++ 中局部对象离开作用域时会自动析构；正常返回、提前 `return` 或异常退出，都能利用这个机制做清理。通过对象生存期管理资源或状态的做法常叫 RAII。

正文的 `OptionalDeviceGuard` 在适用时切换到输入设备，退出当前作用域时恢复之前的设备上下文。它不是把 Tensor 搬到新设备，也不是某种锁。

Python 绑定中的 `pybind11::gil_scoped_release no_gil` 则在进入局部调用代码时释放 GIL，离开时恢复。GIL 可以先理解为传统 CPython 执行 Python 代码时使用的全局锁：进入 C++ 计算路径后临时放开它，其他 Python 线程就有机会运行；回到需要操作 Python 对象的 `wrap` 前恢复它。

这里不用掌握锁的内部实现。只需分清 Guard 各自管理的状态：设备 Guard 管设备上下文，GIL Guard 管解释器锁的持有状态；两个对象不是在做同一件事。绑定中的使用位置见[返回模板](nn模块源码.md#src-python-dispatch-return)。

<a id="cpp-return"></a>
## return、重载和连续的方法调用

`return inner(args...)` 先调用内层，等内层返回后，把它的返回值交给当前函数的调用者。函数调用因此可以一层层进入，也可以沿这些返回语句一层层退出。

`if (...)` 按条件选择分支，`else` 处理条件不满足的情况。`&&` 表示两边条件都满足，`!` 反转条件；例如 `!stream.isSyncLaunchStream()` 表示当前流不是同步发起类型的流。它们与前面 DispatchKeySet 上的集合运算是不同用法。

本文需要分清返回值的类型：`acl_call` 的 `return api_ret` 返回整数状态码；池化函数的 `return result` 返回 Tensor；Python 绑定的 `return wrap(...)` 返回 Python 对象。成功提交和设备完成也不是同一个时间点，见[执行与等待](#npu-execution-details)。

同名函数可以接收不同类型的参数，这叫 C++ 重载。`wrap` 有整数、Tensor 等不同版本，编译器根据参数类型选用适合的版本。Tensor 的两个版本又可以接收不同形式的引用，最后都调用 `THPVariable_Wrap`。

这与 Dispatcher 的 schema 重载是不同层次的概念：`aten::avg_pool2d` 和 `aten::avg_pool2d.out` 是分别登记的算子条目。一个池化辅助函数名字中带 `out`，不会自动让当前 schema 改成 `.out`。

`cmd.Name(...).Input(...).Output(...).Run()` 是连续的方法调用。前面的设置方法返回当前命令对象，后一个点就可以继续调用其方法。各次设置把信息保存进请求，直到 `Run()` 才进入提交流程。

<a id="cpp-dynamic-library"></a>
## 动态库、函数指针与类型转换

动态库是已经编译好的函数集合，程序运行时可以加载并调用。正文中 `dlopen("libopapi.so", ...)` 取得库的访问入口，`dlsym(handler, "aclnnAvgPool2d")` 再按函数名取得地址。

函数指针保存可调用函数的位置。它需要有对应的参数和返回类型，才能按正确的签名调用。本文的 `OpApiFunc` 规定第二段接口的参数类型，见[函数指针类型](nn模块源码.md#src-opapi-function-call)。

`reinterpret_cast<OpApiFunc>(opApiFuncAddr)` 把查到的地址按这种函数指针类型解释。转换不执行函数；后面的 `opApiFunc(workspace_addr, workspace_size, executor, acl_stream)` 才真正调用接口。

`static_cast` 和 `reinterpret_cast` 是不同的类型转换写法。正文适配器用前者把已知具体类型的对象指针还原，动态符号调用用后者解释函数地址。不能把任何一次转换都理解成重新创建对象或启动计算。

符号表中的 `T aclnnAvgPool2d` 表示库定义了这个代码符号。它可以证明接口存在，但不展示函数体，也不足以证明某次调用具体启动了哪个设备 kernel，见[已有符号表记录](nn模块源码.md#src-opapi-library-evidence)。

<a id="registration-start"></a>
## TORCH_LIBRARY_IMPL 怎样开始注册

注册宏写在函数外。构建时，[TORCH_LIBRARY_IMPL](nn模块源码.md#src-library-macro)与[内层展开](nn模块源码.md#src-library-macro-expand)生成注册函数和静态 `TorchLibraryInit` 对象；注册函数里放着正文看到的 `m.impl(...)`。

运行时加载相关库并进行静态初始化，[TorchLibraryInit 构造函数](nn模块源码.md#src-library-init)创建携带命名空间和 dispatch key 的 `Library` 对象 `lib_`，随后执行 `fn(lib_)`。这个 `fn` 就是生成的注册函数，传进去的 `lib_` 在函数体里使用参数名 `m`。

所以 `m.impl("avg_pool2d", ...)` 能知道它为 `aten` 命名空间、`PrivateUse1` key 注册实现。完成注册以后，每次池化调用使用已经接好的调度入口，不需要重新执行整段注册宏。

<a id="registration-wrapper"></a>
## TORCH_FN 怎样把 wrapper 装入 KernelFunction

[TORCH_FN 与 CompileTimeFunctionPointer](nn模块源码.md#src-compile-pointer)在编译期把 wrapper 的函数类型与地址关联起来。`TORCH_FN(W)` 创建可供注册代码使用的封装对象，不会现场执行 `W`。可以用下面的独立示意理解类型与对象的区别：

```cpp
// 假设 add_one 函数已定义；这里只说明封装语法
using AddOneToken = c10::CompileTimeFunctionPointer<int(int), &add_one>;
AddOneToken token;
```

对应地，`TORCH_FN_TYPE(W)` 表示这类封装的类型，`TORCH_FN(W)` 则创建该类型的对象。类型是对象遵守的定义，对象才是实际传给 `m.impl` 的值。

[Library::impl](nn模块源码.md#src-library-impl)把这个封装对象作为 `raw_f` 接收，再创建 `CppFunction f(...)`。[CppFunction 构造函数](nn模块源码.md#src-cppfunction)初始化 `func_`，调用 `KernelFunction::makeFromUnboxedFunction(f)` 准备调度器需要的入口。

[makeFromUnboxedFunction](nn模块源码.md#src-make-function)通过 `WrapFunctionIntoFunctor<FuncPtr>::type` 选定具体函数对象类型。这个类型的[选择规则](nn模块源码.md#src-wrap-type)从注册函数取得返回类型和参数类型列表；具体的 [WrapFunctionIntoFunctor_](nn模块源码.md#src-wrap-function)类则提供能调用 wrapper 的 `operator()`。

接着，[make_unique_base](nn模块源码.md#src-make-unique)创建该具体对象，并通过 `unique_ptr<OperatorKernel>` 管理。`makeFromUnboxedFunctor` 接收它，取得适用于 boxed 和 unboxed 调用的适配函数地址，再把入口地址和对象一起装入 [KernelFunction](nn模块源码.md#src-kernel-ctor)。

其中，`unboxed_kernel_func_` 保存的是 `wrap_kernel_functor_unboxed_::call` 的入口地址，functor 对象另外保存。这解释了运行时为什么先调用适配函数，再由它使用函数对象调用 wrapper。完整装配过程见 [makeFromUnboxedFunctor](nn模块源码.md#src-make-functor)。

最后，[Library::_impl](nn模块源码.md#src-library-_impl)把这个 `CppFunction` 交给 `Dispatcher::registerImpl`。`registerKernel` 保存注册记录，`updateDispatchTable_` 更新受影响的槽位。`PrivateUse1` 的更新还会触发对应自动微分槽位的重新选择，规则见[有效 kernel](#effective-kernel)。

## JIT

本节解释正文选择池化实现时遇到的 CANN 编译开关，以及它与编译缓存的关系。

### 这里的 JIT 在编译什么？

“编译”可以先理解为把计算的描述转换成设备能够执行的程序。本文追踪的是平均池化算子，因此这里讨论的是 CANN 怎样准备算子的执行程序。

JIT 是 Just-In-Time 的缩写，中文称“即时编译”。它与使用预先编译好的程序，主要区别在于准备可执行程序的时机：

| 模式 | 可以怎样理解 |
| --- | --- |
| 在线编译 | 程序运行期间，根据需要编译算子，再执行计算 |
| 使用预编译二进制 | 使用已经编译好的算子程序，准备好输入和参数后执行 |

CANN 的 `ACL_OP_JIT_COMPILE` 选项用于选择这两种模式。这里讨论的是算子编译选项，与 `torch.jit` 处理 PyTorch 模型的功能属于不同层次。依据：[CANN 编译选项说明](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/API/appdevgapi/aclcppdevg_03_1371.html)。

### 怎样设置，代码又怎样读到它？

Python 侧可以通过下面的接口关闭 JIT 模式：

```python
import torch_npu

torch_npu.npu.set_compile_mode(jit_compile=False)
```

参数传 `True` 表示开启，传 `False` 表示关闭；具体支持情况和默认配置需要结合设备与版本查看 [set_compile_mode 接口文档](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu-npu/%28beta%29torch_npu-npu-set_compile_mode.md)。这里展示设置方法，没有在本次学习环境中执行它。

本地的[设置与查询源码](nn模块源码.md#src-jit-options)把 Python 的 `False` 转换成配置值 `"disable"`。配置处理代码会设置 CANN 的 `ACL_OP_JIT_COMPILE` 选项，并更新 torch_npu 保存的查询结果。`CheckJitDisable()` 读取这个结果，供 `op_plugin::avg_pool2d` 判断分支。

特别注意这两个名字问的问题相反：

| 名字 | 它表达的意思 | 关闭 JIT 时 |
| --- | --- | --- |
| `jit_compile` | 要不要开启 JIT？ | `False` |
| `CheckJitDisable()` 的返回值 | JIT 是否已关闭？ | `true` |

所以，设置 `jit_compile=False` 后，正文中的 `is_jit_disable` 为真；还需要 `self_base_format` 也为真，才能进入该函数的 `op_api` 分支。

### 编译缓存是什么，为什么第一次可能更慢？

在线编译需要时间。如果每次都为相同条件重新编译，就会重复做准备工作。编译缓存的作用是保存编译产物，在条件匹配时复用。这里缓存的是算子程序，输入变化后仍然要重新计算输出。

CANN 提供 `ACL_OP_COMPILER_CACHE_MODE` 控制算子编译磁盘缓存：`enable` 表示启用，`disable` 表示禁用，`force` 表示强制刷新。能否复用取决于编译参数、算子参数等条件是否匹配，具体约束见 [CANN 缓存选项说明](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/API/appdevgapi/aclcppdevg_03_1371.html)。

因此，在需要在线编译且没有可复用产物的情况下，首次调用可能包含编译开销，后续调用则可能复用已有产物。仅凭“第一次比较慢”还不能认定是 JIT 导致的，需要结合实际编译记录或性能分析结果判断。

## npu-execution-details

本节解释请求怎样安排执行、准备信息怎样复用，以及为什么接口返回不一定表示设备计算已经完成。

### 直接执行与排队执行

正文先沿直接执行分支讲调用链。框架也可以把请求放入主机任务队列，由负责处理任务的代码稍后发起 CANN 调用。这种队列保存的是待办请求；NPU stream 则安排设备任务的执行顺序，两个概念需要区分。

线程是同一个程序中的另一条执行流程。队列初始化中的 `std::thread cur_consumer(StartConsume, this, device_id)` 创建线程，让它调用 `StartConsume`，并传入当前队列对象和设备编号；之后它循环读取任务。主机线程负责发起 CANN 调用，不等于 NPU 中执行计算的设备 kernel。线程的建立位置见 [InitRepo](nn模块源码.md#src-queue-init)，读取入口见 [StartConsume](nn模块源码.md#src-queue-consume)。

在 `op_api` 路径中，[RunOpApiV2](nn模块源码.md#src-opapi-run)根据当前流和配置，选择直接执行 `acl_call`，或者将它交给任务队列。

在 `acl_op` 路径中，[OpCommand::Run](nn模块源码.md#src-aclop-command-run)可以把请求标为 `COMPILE_AND_EXECUTE` 后入队。后续的 [ExecFunc](nn模块源码.md#src-aclop-queued-exec)处理这个请求，仍会调用 `aclopCompileAndExecute`。

### V1、V2 和执行器缓存

`EXEC_NPU_CMD` 的[选择器](nn模块源码.md#src-exec-npu-select)在 `GetTaskQueueEnable() == 2` 时选择 V2，否则选择 V1。

| 情况 | 对正文调用顺序的影响 |
| --- | --- |
| V1，未命中缓存 | 第一段接口和 workspace 分配先完成，再把第二段调用放进 `acl_call` |
| V2，未命中缓存 | 先保存参数，把转换、第一段、workspace 分配和第二段都放进 `acl_call` |
| 执行器缓存命中 | 可以复用执行信息和 workspace 大小，省去重新准备第一段的步骤 |

这里的执行器缓存与 JIT 编译缓存作用不同，也不会缓存上一次的池化结果。完整代码见 [V1](nn模块源码.md#src-exec-npu-v1)、[V2](nn模块源码.md#src-exec-npu-v2)及 [V1 缓存命中路径](nn模块源码.md#src-opapi-hit-cache)。

### 接口返回与计算完成

“异步”可以先理解为：发起任务以后，调用者可以继续往下运行，设备任务稍后完成。因此，接口成功返回可能只表示提交成功，不能直接理解为输出数字已经全部算好。

“同步等待”则表示等相关设备任务完成后再继续。读取结果或进行耗时统计时，需要考虑这个等待过程。`aclopCompileAndExecute` 的异步约定见[官方接口文档](https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC2/apiref/appdevgapi/aclcppdevg_03_0243.html)；框架的阻塞配置或显式同步请求也会影响实际等待行为。
