from .differentiable import Differentiable
class Sum(Differentiable):
	def __init__(self, x, y):
		self.x = x
		self.y = y
	# Addition operation's backward step passes on the value as is
	def backward(self, var):
		return Sum(self.x.backward(var), self.y.backward(var))

	def compute(self):
		return self.x.compute() + self.y.compute()
	
	def __repr__(self):
		return f'{self.x} + {self.y}'