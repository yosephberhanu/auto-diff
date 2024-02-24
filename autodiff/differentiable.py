from abc import ABC, abstractmethod

class Differentiable(ABC):
	@abstractmethod
	def backward(self, var):
		pass
	@abstractmethod
	def compute(self):
		pass
	@staticmethod
	def _to_symbolic(x):
		'''
		makes sure that x is a tree node by converting it
		into a constant node if necessary
		'''
		if not isinstance(x, Differentiable):
			return Const(x)
		else:
			return x
	def __add__(self, other):
		return Sum(self, self._to_symbolic(other))

	def __radd__(self, other):
		return Sum(self, self._to_symbolic(other))
	
	def __mul__(self, other):
		return Mul(self, self._to_symbolic(other))

	def __rmul__(self, other):
		return Mul(self, self._to_symbolic(other))
	
	def __rsub__(self, other):
		return Sub(self, self._to_symbolic(other))
	
	def __sub__(self, other):
		return Sub(self, self._to_symbolic(other))
	
	def __floordiv__(self, other):
		return Div(self, self._to_symbolic(other))
	
	def __truediv__(self, other):
		return Div(self, self._to_symbolic(other))
	
	def __neg__(self):
		return Mul(Const(-1), self)