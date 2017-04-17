class Config:

	def __init__(self, default_source, override_source):
		self.__default_source = default_source
		self.__override_source = override_source

		self.use_dummy_data = False
		self.create_new_model = False
		self.max_timesteps = 10000
		self.skip_training = True
		self.training_frequency = 5
		self.model_save_frequency = 30
		self.training_window_length = 30 * 1440 # 30 days # this feature not implemented yet
		self.model_input_length = 1440
		self.prediction_latency = 60
		self.training_batch_size = 32
		self.max_input_value = 2000
		self.input_smoothing = 0.7

	def read(self):
		self.__readFromCCM(self.__default_source) if Config.__isCcmUrl(self.__default_source) else self.__readFromFile(self.__default_source)
		self.__readFromCCM(self.__override_source) if Config.__isCcmUrl(self.__override_source) else self.__readFromFile(self.__override_source)

	def __readFromFile(self, filepath):
		pass

	def __readFromCCM(self, ccmpath):
		# ishabh to implement
		pass

	def __isCcmUrl(source):
		# ishabh to implement
		return False