import numpy as np
import random
import sys
import math
import csv
import os.path
import keras.models
import dataLoader
from config import Config
from dataLoader import FileDataLoader, DummyDataLoader
from predictionBasedAnomalyDetector import PredictionBasedAnomalyDetector
from matplotlib import pyplot as plt


# two command line arguments to this program should be
# metric_name on which this detector runs

metric_name = 'so-opm'
default_config_source = 'Configs/default.ini'
override_config_source = 'Configs/{0}.ini'.format(metric_name)
config = Config(default_config_source, override_config_source)
config.read()

timesteps_since_beginning = 0

if config.use_dummy_data:
	dataLoader = DummyDataLoader()
else:
	dataLoader = FileDataLoader('Data/{0}.csv'.format(metric_name))

model = PredictionBasedAnomalyDetector(config)

if not config.create_new_model:
	try:
		print('Looking for saved model..')
		model.load('Saved Models/{0}'.format(metric_name))
	except Exception as e:
		print('Saved model could not be loaded.')
		print('Creating a new model..')
		model.create_new()
	else:
		print('Saved model successfully loaded')
else:
	print('Creating a new model..')
	model.create_new()
	
print('Model ready')


# alerter = Alerter(metric_name, config)

# the infinite loop:
while timesteps_since_beginning != config.max_timesteps:
	
	# wait on getting next point:
	y = dataLoader.getNextPoint()

	# then evaluate:
	results = model.consume(y)

	# optionally allow model to learn something based on some critera, such as learn every 5 minutes
	if not config.skip_training and timesteps_since_beginning % config.learning_frequency:
		model.train(1)

	# save model every few minutes
	if timesteps_since_beginning % config.model_save_frequency:
		model.save('Saved Models/{0}'.format(metric_name))

	# create/update/delete alert according to results
	# alerter.update(results)

	timesteps_since_beginning += 1
	# re read the config in case it has changed
	config.read()

model.show_plots()