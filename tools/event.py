import os

import numpy as np
from PIL import Image
import tqdm
import tools.os_tools as os_tools
import tools.visualization_tools as visualization_tools


TIMESTAMP_COLUMN = 2
X_COLUMN = 0
Y_COLUMN = 1
POLARITY_COLUMN = 3


def save_events(events, file):
    """Save events to ".npy" file.

    In the "events" array columns correspond to: x, y, timestamp, polarity. 

    We store:
    (1) x,y coordinates with uint16 precision.
    (2) timestamp with float32 precision.
    (3) polarity with binary precision, by converting it to {0,1} representation.
    
    """
    if (0 > events[:, X_COLUMN]).any() or (events[:, X_COLUMN] > 2 ** 16 - 1).any():
        raise ValueError("Coordinates should be in [0; 2**16-1].")
    if (0 > events[:, Y_COLUMN]).any() or (events[:, Y_COLUMN] > 2 ** 16 - 1).any():
        raise ValueError("Coordinates should be in [0; 2**16-1].")
    if ((events[:, POLARITY_COLUMN] != -1) & (events[:, POLARITY_COLUMN] != 1)).any():
        raise ValueError("Polarity should be in {-1,1}.")
    events = np.copy(events)
    x, y, timestamp, polarity = np.hsplit(events, events.shape[1])
    polarity = (polarity + 1) / 2
    np.savez(file, x=x.astype(np.uint16), y=y.astype(np.uint16), timestamp=timestamp.astype(np.float32),
             polarity=polarity.astype(np.bool))


def load_events(file):
    """Load events to ".npz" file.
    See "save_events" function description.
    """
    tmp = np.load(file, allow_pickle=True)
    (x, y, timestamp, polarity) = (tmp["x"].astype(np.float64).reshape((-1,)), tmp["y"].astype(np.float64).reshape((-1,)),
        tmp["t"].astype(np.float64).reshape((-1,)), tmp["p"].astype(np.float32).reshape((-1,)) * 2 - 1)
    events = np.stack((x, y, timestamp, polarity), axis=-1)
    if events.shape[0] == 0:
        events = np.zeros([1, 4])
    return events


class EventSequence(object):
    """Stores events in oldes-first order."""

    def __init__(self, features, image_height, image_width, start_time=None, end_time=None):
        """Returns object of EventSequence class.
        Args:
            features: numpy array with events softed in oldest-first order. Inside,
                      rows correspond to individual events and columns to event
                      features (x, y, timestamp, polarity)

            image_height, image_width: widht and height of the event sensor. 
                                       Note, that it can not be inferred
                                       directly from the events, because
                                       events are spares.
            start_time, end_time: start and end times of the event sequence.
                                  If they are not provided, this function inferrs
                                  them from the events. Note, that it can not be
                                  inferred from the events when there is no motion.
        """
        self._features = features
        self._image_width = image_width
        self._image_height = image_height
        if features.shape[0] == 0:
            self._start_time = 0
            self._end_time = 0
        else:
            self._start_time = (start_time if start_time is not None else features[0, TIMESTAMP_COLUMN])
            self._end_time = (end_time if end_time is not None else features[-1, TIMESTAMP_COLUMN])

    def __len__(self):
        return self._features.shape[0]

    def is_self_consistent(self):
        '''
        判断是否合规
        :return:
        '''
        return (self.are_spatial_coordinates_within_range() and self.are_timestamps_ascending()
                and self.are_polarities_one_and_minus_one() and self.are_timestamps_within_range())

    def flip_horizontally(self):
        self._features[:, X_COLUMN] = (self._image_width - 1 - self._features[:, X_COLUMN])

    def flip_vertically(self):
        self._features[:, Y_COLUMN] = (self._image_height - 1 - self._features[:, Y_COLUMN])

    def reverse(self):
        """Reverse temporal direction of the event stream.
        Polarities of the events reversed.
                          (-)       (+) 
        --------|----------|---------|------------|----> time
           t_start        t_1       t_2        t_end

                          (+)       (-) 
        --------|----------|---------|------------|----> time
                0    (t_end-t_2) (t_end-t_1) (t_end-t_start)
        """
        if len(self) == 0:
            return
        self._features[:, TIMESTAMP_COLUMN] = (self._end_time - self._features[:, TIMESTAMP_COLUMN])
        self._features[:, POLARITY_COLUMN] = -self._features[:, POLARITY_COLUMN]
        self._start_time, self._end_time = 0, self._end_time - self._start_time
        # Flip rows of the 'features' matrix, since it is sorted in oldest first.
        self._features = np.copy(np.flipud(self._features))

    def filter_by_polarity(self, polarity, make_deep_copy=True):
        mask = self._features[:, POLARITY_COLUMN] == polarity
        return self.filter_by_mask(mask, make_deep_copy)

    def filter_by_timestamp(self, start_time, duration, make_deep_copy=True):
        """
        Returns event sequence filtered by the timestamp.
        The new sequence includes event in [start_time, start_time+duration).
        """
        end_time = start_time + duration
        mask = (start_time <= self._features[:, TIMESTAMP_COLUMN]) & (end_time > self._features[:, TIMESTAMP_COLUMN])
        event_sequence = self.filter_by_mask(mask, make_deep_copy)
        event_sequence._start_time = start_time
        event_sequence._end_time = start_time + duration
        return event_sequence

    def to_image(self, background=None):
        """Visualizes stream of event as a PIL image.

        The pixel is shown as red if dominant polarity of pixel's
        events is 1, as blue if dominant polarity of pixel's
        events is -1 and white if pixel does not recieve any events,
        or it's events does not have dominant polarity.

        Args:
            background: is PIL image.
        """
        polarity = self._features[:, POLARITY_COLUMN] == 1
        x_negative = self._features[~polarity, X_COLUMN].astype(np.int)
        y_negative = self._features[~polarity, Y_COLUMN].astype(np.int)
        x_positive = self._features[polarity, X_COLUMN].astype(np.int)
        y_positive = self._features[polarity, Y_COLUMN].astype(np.int)

        positive_histogram, _, _ = np.histogram2d(x_positive, y_positive, bins=(self._image_width, self._image_height),
                                                  range=[[0, self._image_width], [0, self._image_height]])
        negative_histogram, _, _ = np.histogram2d(x_negative, y_negative, bins=(self._image_width, self._image_height),
                                                  range=[[0, self._image_width], [0, self._image_height]])

        red = np.transpose(positive_histogram > negative_histogram)
        blue = np.transpose(positive_histogram < negative_histogram)

        if background is None:
            height, width = red.shape
            background = Image.fromarray(np.full((height, width, 3), 255, dtype=np.uint8))
        y, x = np.nonzero(red)
        points_on_background = visualization_tools.plot_points_on_background(y, x, background, [255, 0, 0])
        y, x = np.nonzero(blue)
        points_on_background = visualization_tools.plot_points_on_background(y, x, points_on_background, [0, 0, 255])
        return points_on_background

    def split_in_two(self, timestamp):
        """Returns two sequences from splitting the original sequence in two."""
        if not (self.start_time() <= timestamp <= self.end_time()):
            raise ValueError('"timestamps" should be between start and end of the sequence.')
        first_sequence_duration = timestamp - self.start_time()
        second_sequence_duration = self.end_time() - timestamp
        first_sequence = self.filter_by_timestamp(self.start_time(), first_sequence_duration)
        second_sequence = self.filter_by_timestamp(timestamp, second_sequence_duration)
        return first_sequence, second_sequence

    def make_iterator_over_splits(self, number_of_splits, ts_list=None):
        """Returns iterator over splits in two.
        E.g, if "number_of_splits" = 3, than the iterator will output
        (t_start->t_0, t_0->t_end)
        (t_start->t_1, t_1->t_end)
        (t_start->t_2, t_2->t_end)

        ---|------|------|------|------|--->
         t_start  t0     t1    t2     t_end

        t0 = (t_end - t_start) / (number_of_splits + 1), and ect.   
        """
        start_time = self.start_time()
        end_time = self.end_time()
        if ts_list is None:
            split_timestamps = np.linspace(start_time, end_time, number_of_splits + 2)[1:-1]
        else:
            split_timestamps = ts_list

        for split_timestamp in split_timestamps:
            left_events, right_events = self.split_in_two(split_timestamp)
            yield left_events, right_events

    def make_sequential_iterator(self, timestamps):
        """Returns iterator over sub-sequences of events.
        Args:
            timestamps: list of timestamps that specify bining of  events into the sub-sequences.
                        E.g. iterator will return events:
                        from timestamps[0] to timestamps[1],
                        from timestamps[1] to timestamps[2], and e.c.t.
        """
        if len(timestamps) < 2:
            raise ValueError("There should be at least two timestamps")
        start_timestamp = timestamps[0]
        start_index = self._advance_index_to_timestamp(start_timestamp)
        for end_timestamp in timestamps[1:]:
            end_index = self._advance_index_to_timestamp(end_timestamp, start_index)
            # self._features[start_index:end_index, :].size == 0
            yield EventSequence(features=np.copy(self._features[start_index:end_index, :]),
                                image_height=self._image_height, image_width=self._image_width,
                                start_time=start_timestamp, end_time=end_timestamp)
            start_index = end_index
            start_timestamp = end_timestamp

    def to_folder(self, folder, timestamps, event_file_template="{:06d}"):
        """Saves event sequences from to npz.
        Args:
            folder: folder where events will be saved in the files events_000000.npz,
                    events_000001.npz, etc.
            timestamps: iterator that outputs event sequences. 
        """
        event_iterator = self.make_sequential_iterator(timestamps)
        for sequence_index, sequence in enumerate(event_iterator):
            filename = os.path.join(folder, event_file_template.format(sequence_index))
            save_events(sequence._features, filename)

    @classmethod
    def from_folder(cls, folder, image_height, image_width, event_file_template="{:06d}.npz"):
        filename_iterator = os_tools.make_glob_filename_iterator(os.path.join(folder, event_file_template))
        filenames = [filename for filename in filename_iterator]
        return cls.from_npz_files(filenames, image_height, image_width)

    @classmethod
    def from_npz_files(cls, list_of_filenames, image_height, image_width, start_time=None, end_time=None,):
        """Reads event sequence from numpy file list."""
        if len(list_of_filenames) > 1:
            features_list = []
            for f in list_of_filenames:
                features_list += [load_events(f)]# for filename in list_of_filenames]
            features = np.concatenate(features_list)
        else:
            features = load_events(list_of_filenames[0])
        return EventSequence(features, image_height, image_width, start_time, end_time)

    @classmethod
    def from_npz_file(cls, filename, image_height, image_width, start_time=None, end_time=None,):
        """Reads event sequence from numpy file list."""
        return EventSequence(load_events(filename), image_height, image_width, start_time, end_time)

    @classmethod
    def from_npy_file(cls, filename, image_height, image_width, start_time=None, end_time=None,):
        """Reads event sequence from numpy file list."""
        tmp = np.load(filename)
        tmp[:, -1] = tmp[:, -1] * 2 - 1
        features = tmp[:, [1, 2, 0, 3]]  # t,x,y,p -> x,y,t,p
        return EventSequence(features, image_height, image_width, start_time, end_time)

    @classmethod
    def from_npy_files(cls, list_of_filenames, image_height, image_width, start_time=None, end_time=None,):
        if len(list_of_filenames) > 1:
            features_list = []
            for f in list_of_filenames:
                features_list += [np.load(f)]# for filename in list_of_filenames]
            features = np.concatenate(features_list)
        else:
            features = np.load(list_of_filenames[0])
        features[:, -1] = features[:, -1] * 2 - 1
        features = features[:, [1, 2, 0, 3]]  # t,x,y,p -> x,y,t,p
        return EventSequence(features, image_height, image_width, start_time, end_time)

    def are_spatial_coordinates_within_range(self):
        x = self._features[:, X_COLUMN]
        y = self._features[:, Y_COLUMN]
        return np.all((x >= 0) & (x < self._image_width)) and np.all((y >= 0) & (y < self._image_height))

    def are_timestamps_ascending(self):
        timestamp = self._features[:, TIMESTAMP_COLUMN]
        return np.all((timestamp[1:] - timestamp[:-1]) >= 0)

    def are_timestamps_within_range(self):
        timestamp = self._features[:, TIMESTAMP_COLUMN]
        return np.all((timestamp <= self.end_time()) & (timestamp >= self.start_time()))

    def are_polarities_one_and_minus_one(self):
        polarity = self._features[:, POLARITY_COLUMN]
        return np.all((polarity == -1) | (polarity == 1))

    def duration(self):
        return self.end_time() - self.start_time()

    def start_time(self):
        return self._start_time

    def end_time(self):
        return self._end_time

    def min_timestamp(self):
        return self._features[:, TIMESTAMP_COLUMN].min()

    def max_timestamp(self):
        return self._features[:, TIMESTAMP_COLUMN].max()

    def filter_by_mask(self, mask, make_deep_copy=True):
        if make_deep_copy:
            return EventSequence(features=np.copy(self._features[mask]), image_height=self._image_height,
                image_width=self._image_width, start_time=self._start_time, end_time=self._end_time)
        else:
            return EventSequence(features=self._features[mask], image_height=self._image_height,
                image_width=self._image_width, start_time=self._start_time, end_time=self._end_time)

    def copy(self):
        return EventSequence(features=np.copy(self._features), image_height=self._image_height,
                             image_width=self._image_width, start_time=self._start_time, end_time=self._end_time)

    def _advance_index_to_timestamp(self, timestamp, start_index=0):
        """Returns index of the first event with timestamp > "timestamp" from "start_index"."""
        index = start_index
        while index < len(self):
            if self._features[index, TIMESTAMP_COLUMN] >= timestamp:
                return index
            index += 1
        return index
