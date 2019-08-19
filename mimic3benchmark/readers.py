from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import random
import math
import csv


class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=777):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)
    
    def limit_data(self, percent):
        listfile_path = os.path.join(self._dataset_dir, "listfile_{}.csv".format(str(int(percent))))
        
        if not os.path.exists(listfile_path):
            num_samples = int(math.ceil(self.get_number_of_examples() * (percent/100)))
            self.random_shuffle()
            self._data = self._data[0:num_samples]
            #Save Data Limited Samples
            with open(listfile_path, "w") as outfile:
                csv_outfile = csv.writer(outfile)
                outfile.write(self._listfile_header)
                for line in self._data:
                    csv_outfile.writerow(line)
        else:
            self.__init__(dataset_dir=self._dataset_dir, listfile=listfile_path)

#Reader Used for Discharge and Decompensation Task (AKA Prediction for Each Hour)
class HourlyReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=24.0):
        """ Reader for decompensation prediction task.
        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        :period_length:     Amount of prior data to use for prediction.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), int(y)) for (x, t, y) in self._data]
        self._period_length = period_length
        self._input_dim = None

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return (np.stack(ret), header)
    
    def get_input_dim(self):
        if self._input_dim is None:
            name = self._data[0][0]
            t = self._data[0][1]
            (X, header) = self._read_timeseries(name, t)
            self._input_dim = X.shape[1] - 1
        return self._input_dim

    def read_example(self, index):
        """ Read the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Directory with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            p : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                Label within next 24 hours.
            t : float
                Hour from which to make prediction.
            
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of examples (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name, t)
        
        p = self._period_length

        return {"X": X,
                "t": t,
                "y": y,
                "p": p,
                "header": header,
                "name": name}
    
#Reader used for In Hospital Mortality Task and Exended Length of Stay (AKA Predicted Based on first 24hrs)
class DayReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=24.0):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length
        self._input_dim = None

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)
    
    def get_input_dim(self):
        if self._input_dim is None:
            name = self._data[0][0]
            (X, header) = self._read_timeseries(name)
            self._input_dim = X.shape[1] - 1
        return self._input_dim

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data used for prediction = first t hours.
            y : int (0 or 1)
                In-hospital mortality or LOS > 7 Days.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}
    
class PatientEmbeddingReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=48.0, totalfile=None):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Double the Length of the period (in hours) from which the embedding is created.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(wt), int(w)) for (x, wt, w) in self._data]
        self._period_length = period_length
        self._input_dim = None
        
        if totalfile is None:
            totalfile_path = os.path.join(dataset_dir, 'totalfile.csv')
        else:
            totalfile_path = totalfile
        with open(totalfile_path, "r") as tfile:
            self._total = int(tfile.readlines()[0])
            
    def get_number_of_visits(self):
        return self._total
        
    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
            try:
                (np.stack(ret), header)
            except:
                print(ts_filename)
            return (np.stack(ret), header)
    
    def get_input_dim(self):
        if self._input_dim is None:
            name, norm, end_time = self._data[0]
            (X, header) = self._read_timeseries(name)
            self._input_dim = X.shape[1] - 1
        return self._input_dim  
        
    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            norm : int (0 or 1)
                Number of Windows for the Patient in the dataset.
            end_time : float
                last hour to include in the current window for embedding.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")
            
        name, norm, end_time = self._data[index]
        end_time += 48
        t = self._period_length
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "norm": norm,
                "end_time": end_time,
                "header": header,
                "name": name}