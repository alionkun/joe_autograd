from typing import List, NamedTuple, Callable, Optional, Union

import numpy as np

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray] # 重要！相邻两个tensor之间的依赖关系/转换关系，在整个计算中唯一确定的，因此对应的梯度关系也是唯一确定的，并且取决于该转换关系。这是反向传播的最直白的表示

Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class Tensor: # 这个 Tensor 使用 np.ndarray 来表示数据，然后维护依赖的上游关系，以及计算给每个上游反传梯度的函数
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data)) # 初始状态，tensor对应的梯度为全零

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                grad = Tensor(np.ones_like(self.data))
                # raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data += grad.data # 下游回传的梯度，直接累加

        for dependency in self.depends_on:
            if dependency.tensor.requires_grad:
                backward_grad = dependency.grad_fn(grad.data) # 增量计算依赖/输入的梯度，将结果通过 backward() 回传给上游，上游也会累加，并继续传递给上游的上游，直至某个上游没有入边，反向传播结束。
                dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> 'Tensor':
        return tensor_sum(self)
    
    def add(self, other) -> 'Tensor':
        return tensor_add(self, other)
        


def tensor_sum(t: Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements.
    """
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor, so each input element
            contributes that much
            """
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)] # OP 除了定义input->ouput的计算方法，也定义了 doutput / dinput，也就是梯度计算方法

    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)

def tensor_add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = False
    depends_on = []
    if t1.requires_grad:
        requires_grad = True
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t1.data)
        depends_on.append(Dependency(t1, grad_fn))
    if t1.requires_grad:
        requires_grad = True
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t2.data)
        depends_on.append(Dependency(t2, grad_fn))
    return Tensor(data,
                  requires_grad,
                  depends_on)
    

    