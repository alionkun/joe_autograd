import unittest

from autograd.tensor import Tensor

class TestTensorSum(unittest.TestCase):
    def test_simple_sum(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1.sum()

        t2.backward() # 不提供梯度，默认为 1

        assert t1.grad.data.tolist() == [1, 1, 1] # y=x1+x2+x3, 所以梯度都是 1

    def test_sum_with_grad(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1.sum()

        t2.backward(Tensor(3))

        assert t1.grad.data.tolist() == [3, 3, 3]


class TestTensorAdd(unittest.TestCase):
    def test_simple_add(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=False)
        t3 = t1.add(t2)
        
        t3.backward()
        
        assert t1.grad.data.tolist() == [1, 1, 1]
        assert t2.grad == None
        
    def test_add_with_grad(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=False)
        t3 = t1.add(t2)
        
        t3.backward(Tensor(3))
        
        assert t1.grad.data.tolist() == [3, 3, 3]
        assert t2.grad == None