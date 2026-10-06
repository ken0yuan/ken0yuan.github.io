#
## F知识
## `_add_docstr`相关知识
## `torch._C._nn.avg_pool2d`如何挂载上了`THPVariable_avg_pool2d`方法
## Args是如何把python的tensor从schema声明变成一个拥有.key_set()属性的对象的
## tls: Thread Local Storage，线程局部存储
## kernelfunction什么要“计算”，而不是直接赋值刚注册的 kernel？
因为一个槽位的有效实现可能来自：
- 直接注册到这个 key 的实现。
- 适用的复合实现。
- 后端 fallback。
- 都没有，则留下缺失实现状态。
这些规则在 [computeDispatchTableEntryWithDebug (line 352)](/Users/yuanchengyao/graduation-project/pytorch/aten/src/ATen/core/dispatch/OperatorEntry.cpp:352) 中处理