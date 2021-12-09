import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from core.util import time_seq


plt.rcParams.update({'font.size': 13})
plt.rcParams['figure.figsize'] = 10, 8


class TensorBoardLogger(object):
	"""
    Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
	(Modified)
    """

	def __init__(self, log_dir):
		"""Create a summary writer logging to log_dir."""
		self.writer = SummaryWriter(log_dir=log_dir)

	def scalar_summary(self, tag, step, value):
		"""Log a scalar variable."""
		self.writer.add_scalar(tag, value, global_step=step)

	def image_summary(self, tag, images, step):
		"""Log a list of images."""
		self.writer.add_images(tag, images, global_step=step)

	def histo_summary(self, tag, values, step, bins=1000):
		"""Log a histogram of the tensor of values."""
		# Create a histogram using numpy
		counts, bin_edges = np.histogram(values, bins=bins)

		# Fill the fields of the histogram proto
		hist_min = float(np.min(values))
		hist_max = float(np.max(values))
		hist_num = int(np.prod(values.shape))
		hist_sum = float(np.sum(values))
		hist_sum_squares = float(np.sum(values ** 2))

		# Drop the start of the first bin
		bin_edges = bin_edges[1:]

		# add histogram
		self.writer.add_histogram_raw(
			tag, hist_min, hist_max, hist_num, hist_sum,
			hist_sum_squares, bin_edges, counts, global_step=step)
		self.writer.flush()


class Plot:
	def __init__(self, save_path):
		self.Y = []
		self.X = []
		self.ax = None
		self.fig = None
		self.save_path = save_path

	def save(self):
		# list and ',' = list[0]
		line, = self.ax.plot(self.X, self.Y, 'b')
		if self.ax.get_title() != '':
			name = self.ax.get_title().replace(' ', '_')
			self.fig.savefig(self.save_path + name + '.png')
		else:
			name = time_seq()
			self.fig.savefig(self.save_path + name + '.png')


class MatplotlibLogger:

	def __init__(self, save_path):
		self.save_path = save_path
		self.plot_dict = {}

		if self.save_path[-1] != '/':
			self.save_path += '/'

	def add_plot(self, tag: str, x_label, y_label, title=''):
		plot = Plot(self.save_path)
		plot.fig, plot.ax = plt.subplots()
		plot.ax.set_xlabel(x_label)
		plot.ax.set_ylabel(y_label)
		plot.ax.set_title(title)

		self.plot_dict[tag] = plot

	def scalar_summary(self, tag, x, y):
		self.plot_dict[tag].Y.append(y)
		self.plot_dict[tag].X.append(x)
		self.plot_dict[tag].save()







