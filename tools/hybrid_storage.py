from tools import event, image_sequence
import os


class HybridStorage(object):
    def __init__(self, events, gtImages):
        self._events = events
        self._gtImages = gtImages

    def get_image_size(self):
        return self._gtImages._height, self._gtImages._width

    @classmethod
    def from_folders(cls, event_folder, gt_image_folder, image_file_template="*.jpg", gt_img_timestamps_file="ts_frame.txt", event_name='event.npy'):

        gt_images = image_sequence.ImageSequence.from_folder(folder=gt_image_folder,
                                                             image_file_template=image_file_template,
                                                             timestamps_file=gt_img_timestamps_file)

        events = event.EventSequence.from_npy_file(filename=os.path.join(event_folder, event_name),
                                                    image_height=gt_images._height, image_width=gt_images._width)


        return cls(events, gt_images)


