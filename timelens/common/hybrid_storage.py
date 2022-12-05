from timelens.common import event, image_sequence, iterator_modifiers
import os


class HybridStorage(object):
    """Template class that stores events and images."""

    def __init__(self, images, events):
        self._images = images
        self._events = events

    def get_image_size(self):
        return self._images._height, self._images._width

    def make_interframe_events_iterator(self, number_of_skips):
        timestamps = list(self.make_boundary_timestamps_iterator(number_of_skips))
        print(timestamps)
        return self._events.make_sequential_iterator(timestamps)

    def make_boundary_timestamps_iterator(self, number_of_skips):
        return iterator_modifiers.make_skip_iterator(iter(self._images._timestamps), number_of_skips)

    def make_boundary_frames_iterator(self, number_of_skips):
        return iterator_modifiers.make_iterator_over_groups(iterator_modifiers.make_skip_iterator(
                iter(self._images._images), number_of_skips), 2)

    @classmethod
    def from_folders(cls, event_folder, image_folder, event_file_template="{:06d}.npz", image_file_template="{:06d}.png",
            cropping_data=None, timestamps_file="timestamp.txt"):
        images = image_sequence.ImageSequence.from_folder(folder=image_folder, image_file_template=image_file_template,
                                                          timestamps_file=timestamps_file)
        # events = event.EventSequence.from_folder(folder=event_folder, image_height=images._height,
        #                                          image_width=images._width, event_file_template=event_file_template)
        events = event.EventSequence.from_npy_files(filename=os.path.join(event_folder, 'event_cut.npy'),
                                                    image_height=images._height, image_width=images._width)
        return cls(images, events)


if __name__ == '__main__':
    storage = HybridStorage.from_folders('/repository2/lisiqi/data/cup/cup1/final1/.',
                                         '/repository2/lisiqi/data/cup/cup1/final1/dvs_cut',
                                         image_file_template="*.jpg", timestamps_file='../ts_dvs.txt')
    # storage = HybridStorage.from_folders('/home2/lisiqi/timelens/data/images/', '/home2/lisiqi/timelens/data/events/',
    #                                      image_file_template='\d{6}.png')