from .differentiable import Differentiable
from .mul import Mul
from .sub import Sub

class Div(Differentiable):
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def compute(self):
		return self.x.compute() / self.y.compute()

	# Quotient rule of differentiation
	# \frac{\mathrm{d} }{\mathrm{d} x} \frac{f(x)}{g(x)} = 
	#			\frac{ 
	#					 g(x) * \frac{\mathrm{d} }{\mathrm{d} x}f(x) 
	#									-
	#					 f(x) * \frac{\mathrm{d} }{\mathrm{d} x}g(x) 
	#				 }
	#			{g(x) * g(x)}
	
	def backward(self, var):
		# return Div(
		# 			Sub( 
		# 				Mul(self.x.backward(var), self.y.compute()),
		# 				Mul(self.x.compute(), self.y.backward(var))
		# 			),
		# 			Mul(self.y.compute(), self.y.compute())
		# 		)
		return Div(Sub(Mul(self.y, self.x.backward(var)), Mul(self.x, self.y.backward(var))), Mul(self.y, self.y))  # This line assumes 'var' has a 'grad' attribute to accumulate gradients

	def __repr__(self):
		return f'{self.x} / {self.y}'