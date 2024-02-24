from .differentiable import Differentiable
from .sum import Sum
class Mul(Differentiable):
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def compute(self):
		return self.x.compute() * self.y.compute()
	
	# Product rule of differentiation
	# \frac{\mathrm{d} }{\mathrm{d} x}(f(x)*g(x)) = 
	#			g(x) * \frac{\mathrm{d} }{\mathrm{d} x}f(x) 
	#					+ 
	#			f(x) * \frac{\mathrm{d} }{\mathrm{d} x}g(x) 
	def backward(self, var):
		return Sum(
					Mul (self.x.backward(var), self.y),
					Mul (self.x, self.y.backward(var)) )

	def __repr__(self):
		return f'{self.x} * {self.y}'