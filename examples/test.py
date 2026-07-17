import libAI as la

x = la.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
w = la.Tensor.random_normal([3, 2])
y = x.matmul(w)

print ("Input Tensor x:", x)
print ("Input Shape:", x.shape)
print ("Weight Tensor w:", w)
print ("Weight Shape:", w.shape)
print ("Output Tensor y:", y)
print ("Output Shape:", y.shape)

hello = la.Tensor([[7.0, 8.0], [12, 122], [13, 14]])

print ("Hello Tensor:", hello)
print ("Hello Shape:", hello.shape)
print ("Hello Tensor with data:", hello.data)
print ("Hello Tensor with data:", hello[2])
print ("Hello Tensor with data:", hello[1, 1])
print ("Hello Tensor ndim:", hello.ndim)
