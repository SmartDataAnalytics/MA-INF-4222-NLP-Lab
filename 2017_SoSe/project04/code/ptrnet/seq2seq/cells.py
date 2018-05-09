import recurrentshop
from recurrentshop.cells import *
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Activation
from keras.layers import add, multiply, concatenate
from keras.activations import tanh, softmax
from keras import backend as K
from keras import initializers
import theano
import theano.tensor as T

def _slice(x, dim, index):
    return x[:, index * dim: dim * (index + 1)]

def get_slices_custom(x, n, dim):
    # dim = int(K.int_shape(x)[1] / n)
    dim = int(dim/n)
    print dim,"**********************"
    return [Lambda(_slice, arguments={'dim': dim, 'index': i}, output_shape=lambda s: (s[0], dim))(x) for i in range(n)]

class LSTMDecoderCell(ExtendedRNNCell):
	'''
	This cell implements an LSTM which (IMO) is best described in http://colah.github.io/posts/2015-08-Understanding-LSTMs/.

	**As a black box**
		Visualize how does the LSTM Cell fits into the Recurrent Framework. 
		It outputs a vector (h(t)). This vector is then passed through an activation (sigmoid typically) OUTSIDE of the cell, within the entire mdoel.
		So h(t), h(t-1) ... are the outputs of the LSTM cell,
		but y(t), y(t-1) ... are the outputs of the network which can be compared to the true outputs. p
		(More on this on http://www.deeplearningbook.org/contents/rnn.html, pg 378, figure 10.3 )

		Now, for a particular cell
		__Inputs__:
		x (input); h(t-1) (hidden state); c(t-1) (memory)
		__Outputs__:
		h(t) (output of the cell), c(t) (memory of the cell), and [OPTIONAL] y(t) (See paragraph above)


	**Peeking In**
		An LSTM can be categorized by the following equations:

		_Forget Gate_:
		f(t) = sigmoid( W_f \times [ h(t-1), x(t) ] + b_f )

		_Input Gate_:
		i(t) = sigmoid( W_i \times [ h(t-1), x(t) ] + b_i )

		_Memory_:
		c_cap = tanh( W_c \times [ h(t-1), x(t) ] + b_c )

		c(t) = f(t)*c(t-1) + i(t)*c_cap

		_Output Gate_:
		o(t) = sigmoid( W_o \times [ h(t-1), x(t) ] + b_o )

		_Output (h(t))_:
		h(t) = o(t)*tanh(c(t))

		**Implementation**
		This is where things start going B.A.D.

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Implementation
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	'''
	def __init__(self, hidden_dim=None, **kwargs):
		print "here"
		if hidden_dim:
			self.hidden_dim = hidden_dim
		else:
			self.hidden_dim = self.output_dim
		super(LSTMDecoderCell, self).__init__(**kwargs)

	def build_model(self, input_shape):
		hidden_dim = self.hidden_dim
		output_dim = self.output_dim
		''' 	
			Input :- This returns a tensor.
			input_shape = (number_of_times_unfolded,dimension_of_each_ouptu)
		'''
		x = Input(batch_shape=input_shape)
		h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
		c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))

		W1 = Dense(hidden_dim * 4,
					 kernel_initializer=self.kernel_initializer,
					 kernel_regularizer=self.kernel_regularizer,
					 use_bias=False)
		W2 = Dense(output_dim,
					 kernel_initializer=self.kernel_initializer,
					 kernel_regularizer=self.kernel_regularizer,)
		U = Dense(hidden_dim * 4,
					kernel_initializer=self.kernel_initializer,
					kernel_regularizer=self.kernel_regularizer,)

		z = add([W1(x), U(h_tm1)])

		z0, z1, z2, z3 = get_slices(z, 4)
		i = Activation(self.recurrent_activation)(z0)
		f = Activation(self.recurrent_activation)(z1)
		c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
		o = Activation(self.recurrent_activation)(z3)
		h = multiply([o, Activation(self.activation)(c)])
		y = Activation(self.activation)(W2(h))

		return Model([x, h_tm1, c_tm1], [y, h, c]) #h_tm1 --> h(t-1) i.e h of previous timestep.
		# return Model([x, h_tm1, c_tm1], [y, h]) #h_tm1 --> h(t-1) i.e h of previous timestep.


class AttentionDecoderCell(ExtendedRNNCell):

	def __init__(self, hidden_dim=None, **kwargs):
		if hidden_dim:
			self.hidden_dim = hidden_dim
		else:
			self.hidden_dim = self.output_dim
		self.input_ndim = 3
		super(AttentionDecoderCell, self).__init__(**kwargs)


	def build_model(self, input_shape):
		#input shape in None,input_len,hidden_dimension

		input_dim = input_shape[-1]
		output_dim = self.output_dim
		input_length = input_shape[1]
		hidden_dim = self.hidden_dim

		x = Input(batch_shape=input_shape)
		h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
		c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
		
		W1 = Dense(hidden_dim * 4,
					 kernel_initializer=self.kernel_initializer,
					 kernel_regularizer=self.kernel_regularizer)
		W2 = Dense(output_dim,
					 kernel_initializer=self.kernel_initializer,
					 kernel_regularizer=self.kernel_regularizer)
		W3 = Dense(1,
					 kernel_initializer=self.kernel_initializer,
					 kernel_regularizer=self.kernel_regularizer)
		U = Dense(hidden_dim * 4,
					kernel_initializer=self.kernel_initializer,
					kernel_regularizer=self.kernel_regularizer)

		'''
			1. Lambda() returns a function
			2. It is a keras thing. It executes lambda expressions.
			**Parameters**
				>> output_shape: how do you want your output.
				>> masks...	

			lambda x: K.repeat(x, input_length)
			lambda: declaration
			x:y -> f(x) = y
			Inputlength: number of encoder unfoldings
			x = one (maybe the last one) encoder output.
		'''
		C = Lambda(lambda x: K.repeat(x, input_length), output_shape=(input_length, input_dim))(c_tm1)
		_xC = concatenate([x, C])
		_xC = Lambda(lambda x: K.reshape(x, (-1, input_dim + hidden_dim)), output_shape=(input_dim + hidden_dim,))(_xC) #essentially transpose

		''' 
			alpha is softmax over input length 
		'''
		alpha = W3(_xC)
		alpha = Lambda(lambda x: K.reshape(x, (-1, input_length)), output_shape=(input_length,))(alpha)
		alpha = Activation('softmax')(alpha)

		_x = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 1)), output_shape=(input_dim,))([alpha, x])

		z = add([W1(_x), U(h_tm1)])

		z0, z1, z2, z3 = get_slices(z, 4)

		i = Activation(self.recurrent_activation)(z0)
		f = Activation(self.recurrent_activation)(z0)

		c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
		o = Activation(self.recurrent_activation)(z3)
		h = multiply([o, Activation(self.activation)(c)])
		y = Activation(self.activation)(W2(h))

		return Model([x, h_tm1, c_tm1], [y, h, c])


class PointerDecoderCell(ExtendedRNNCell):
	def __init__(self, hidden_dim=None, **kwargs):
		if hidden_dim:
			self.hidden_dim = hidden_dim
		else:
			self.hidden_dim = self.output_dim
		self.input_ndim = 3
		super(PointerDecoderCell, self).__init__(**kwargs)

	def slice(self,x):
	    return x[:,-1,:]

	def custom_soft_max(self,alpha):
		e = K.exp(alpha - K.max(alpha, axis=-1, keepdims=True))
		s = K.sum(e, axis=-1, keepdims=True)
		return e/s
	def custom_flatten(self,x):
		return K.batch_flatten(x)

	def build_model(self, input_shape):
	
		input_dim = input_shape[-1]
		output_dim = self.output_dim
		input_length = input_shape[1]
		hidden_dim = self.hidden_dim
		print "the input shape is ", input_shape, "hidden shape ", hidden_dim

		# print input_shape
		# print hidden_dim
		# raw_input("Verify Shapes")

		# x = K.variable(np.random.rand(1,input_shape[1],input_shape[2]))

		x = Input(batch_shape=input_shape)

		# Slicing doesn't work
		# slice_layer = Lambda(self.slice,output_shape=(1,hidden_dim))
		# x_tm1 = slice_layer(x)

		#Transposing, forget it.
		# x_tm1 = K.transpose(x_tm1)				#Does not work!
		
		# Let's try flattening inputs instead
		x_tm1 = Lambda(self.custom_flatten, output_shape=(input_shape[0], input_length*hidden_dim))(x)
		# x_tm1 = K.batch_flatten(x)


		h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
		c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
		
		# h_tm1 = K.variable(np.random.rand(1,hidden_dim))
		# c_tm1 = K.variable(np.random.rand(1,hidden_dim))

		W1 = Dense(hidden_dim * 4,
					 kernel_initializer=self.kernel_initializer,
					 kernel_regularizer=self.kernel_regularizer,
					 use_bias=False,
					 input_shape=(hidden_dim*input_length,),
					 name="W1")
		W2 = Dense(output_dim,
					 kernel_initializer=self.kernel_initializer,
					 kernel_regularizer=self.kernel_regularizer)
		W3 = Dense(1,
					 kernel_initializer=self.kernel_initializer,
					 kernel_regularizer=self.kernel_regularizer,
					 use_bias=False,
					 name="W3")
		U = Dense(hidden_dim * 4,
					kernel_initializer=self.kernel_initializer,
					kernel_regularizer=self.kernel_regularizer,
					use_bias=False,
					name="U")

		# print K.eval(x).shape
		# print K.eval(x_tm1).shape
		# print K.eval(h_tm1).shape
		# raw_input('check the dimenbasipon f0r x and h')
		# print "x_tm1"
		# print K.eval(x_tm1)
		# print K.eval(x_tm1).shape
		# raw_input("Berry Berry Berrifyxxxx")
		# print "W1 dot x_tm1"
		# print K.eval(W1(x_tm1))
		# print K.eval(W1(x_tm1)).shape
		# raw_input("Berry Berry Berrify")

		z = add([W1(x_tm1), U(h_tm1)])	

		z0, z1, z2, z3 = get_slices_custom(z, 4, 4*hidden_dim)

		i = Activation(self.recurrent_activation)(z0)
		f = Activation(self.recurrent_activation)(z1)

		temp1 = multiply([f, c_tm1])
		temp2 = multiply([i, Activation(self.activation)(z2)])

		c = add([temp1, temp2])
		# c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
		o = Activation(self.recurrent_activation)(z3)
		h = multiply([o, Activation(self.activation)(c)])

		# #Treating h as d_i (wrt Pointer Network nomenclature https://arxiv.org/pdf/1506.03134.pdf)

		H = Lambda(lambda x: K.repeat(x, input_length), output_shape=(input_length, input_dim))(h)
		_xH = concatenate([x, H])
		_xH = Lambda(lambda x: K.reshape(x, (-1, input_dim + hidden_dim)), output_shape=(input_dim + hidden_dim,))(_xH)

		# print K.eval(_xH)
		# print K.eval(_xH).shape
		# raw_input("Verify Shapes _xH")

		alpha = W3(_xH)
		alpha = Lambda(lambda x: K.reshape(x, (-1, input_length)), output_shape=(input_length,))(alpha)			#Transpose
		
		alpha = W2(alpha)
		alpha = Activation('softmax')(alpha)
		

		# softer = Lambda(self.custom_soft_max,output_shape=(input_length,))
		# alphas = softer(alpha)
		return Model([x, h_tm1, c_tm1], [alpha, h, c])
		# alpha = softmax(alpha
		# alpha = K.softmax(alpha)
		# alpha = Activation('tanh')(alpha)

		# y = Activation(self.activation)(W2(alpha))
		# print K.eval(alpha)
		# print K.eval(alpha).shape
		# raw_input("Verify Shapes alpha")
		# print K.eval(h)
		# print K.eval(h).shape
		# raw_input("Verify Shapes h")
		# print K.eval(c)
		# print K.eval(c).shape
		# raw_input("Verify Shapes c")

