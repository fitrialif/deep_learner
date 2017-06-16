from keras.wrappers.scikit_learn import KerasClassifier

class KerasClassifierImpl(KerasClassifier):
	def fit(self, x, y, **kwargs):
		if not issparse(x):
			return super().fit(x, y, **kwargs)
		if self.build_fn is None:
			self.model = self.__call__(**self.filter_sk_params(self.__call__))
		elif not isinstance(self.build_fn, types.FunctionType):
			self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
		else:
			self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

		loss_name = self.model.loss
		if hasattr(loss_name, '__name__'):
			loss_name = loss_name.__name__
		if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
			y = to_categorical(y)
		### fit => fit_generator
		fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit_generator))
		fit_args.update(kwargs)
		############################################################
		self.model.fit_generator(self.get_batch(x, y, self.sk_params["batch_size"]), samples_per_epoch=x.shape[0], **fit_args)
		return self
	def get_batch(self, x, y=None, batch_size=32):
		index = np.arange(x.shape[0])
		start = 0
		while True:
			if start == 0 and y is not None:
				np.random.shuffle(index)
			batch = index[start:start+batch_size]
			if y is not None:
				yield x[batch].toarray(), y[batch]
			else:
				yield x[batch].toarray()
			start += batch_size
			if start >= x.shape[0]:
				start = 0

	def predict_proba(self, x):
		""" adds sparse matrix handling """
		if not issparse(x):
			return super().predict_proba(x)

		preds = self.model.predict_generator(self.get_batch(x, None, self.sk_params["batch_size"]), val_samples=x.shape[0])
		return preds