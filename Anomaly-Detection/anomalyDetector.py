class AnomalyDetector:
	# Interface

	def __init__(self, config):
		self.__config = config

	def create_new(self):
		pass

	def consume(self, y):
		results = {}
		results.anomaly_exists = False
		results.anomaly_started = False
		results.anomaly_endend = False
		results.current_anomaly_duration = 0
		results.current_anomaly_confidence = 0
		results.notes = 'Nothing interesting right now'
		return results

	def train(self, number_of_steps):
		pass

	def save(self, dir):
		# saves model and whatever data is to be saved in directory 'dir'
		pass

	def load(self, dir):
		# load model and needed data from directory 'dir'
		# Should raise exception if load failed
		pass

	def show_plots(self):
		pass