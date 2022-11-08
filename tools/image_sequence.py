import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages/')
import os
import numpy as np
from PIL import Image
from tools import os_tools, iterator_modifiers
import tqdm
from torchvision import transforms


class ImageSequence(object):
    """Class that provides access to image sequences."""

    def __init__(self, images, timestamps):
        self._images = images
        self._timestamps = timestamps
        self._width, self._height = self[0].shape[2], self[0].shape[1]
        # self._width, self._height = self[0].size

        # self.timestamps = timestamps

    def __len__(self):
        return len(self._images)

    def skip_and_repeat(self, number_of_skips, number_of_frames_to_insert):
        images = list(iterator_modifiers.make_skip_and_repeat_iterator(iter(self._images), number_of_skips, number_of_frames_to_insert))
        timestamps = list(iterator_modifiers.make_skip_and_repeat_iterator(iter(self._timestamps), number_of_skips, number_of_frames_to_insert))
        return ImageSequence(images, timestamps)
        
    def make_frame_iterator(self, number_of_skips):
        return iter(self._images)
    
    def to_folder(self, folder, file_template="{:06d}.png", timestamps_file="timestamp.txt"):
        """Save images to image files"""
        folder = os.path.abspath(folder)
        for image_index, image in enumerate(self._images):
            filename = os.path.join(folder, "{:06d}.png".format(image_index))
            image.save(filename)
        os_tools.list_to_file(
        os.path.join(folder, "timestamp.txt"), [str(timestamp) for timestamp in self._timestamps])

    # def to_video(self, filename):
    #     """Saves to video."""
    #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #     video = cv2.VideoWriter(filename, fourcc, 30.0, (self._width, self._height))
    #     for image in self._images:
    #         video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    #     video.release()

    def __getitem__(self, index):
        """Return example by its index."""
        if index >= len(self):
            raise IndexError
        return self._images[index]

    @classmethod
    def from_folder(cls, folder, image_file_template="frame_{:010d}.png", timestamps_file="timestamps.txt", start_time=-1, end_time=-1):
        # filename_iterator = os_tools.make_glob_filename_iterator(os.path.join(folder, image_file_template))
        # filenames = [f for f in filename_iterator]
        import glob
        num = len(glob.glob(os.path.join(folder, image_file_template)))
        filenames = [os.path.join(folder, '%d'%f + image_file_template[-4:]) for f in range(num)]

        images = []
        for f in tqdm.tqdm(filenames):
            # images += [Image.open(f).convert("RGB")]
            images += [transforms.ToTensor()(Image.open(f))]

        timestamps = [float(number.split(" ")[-1]) for number in os_tools.file_to_list(os.path.join(folder, timestamps_file))]

        if start_time != -1:
            idx1 = (np.array(timestamps) >= start_time)
        if end_time != -1:
            idx2 = (np.array(timestamps) <= end_time)
        if start_time != -1 or end_time != -1:
            idx = idx1 & idx2
            timestamps = [timestamps[n] for n in range(len(timestamps)) if idx[n]]
            images = [images[n] for n in range(len(images)) if idx[n]]
        return cls(images, timestamps)

    # @classmethod
    # def from_video(cls, filename, fps):
    #     images = []
    #     capture = cv2.VideoCapture(filename)
    #     while capture.isOpened():
    #         success, frame = capture.read()
    #         if not success:
    #             break
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         images.append(Image.fromarray(frame))
    #     capture.release()
    #     timestamps = [float(index) / float(fps) for index in range(len(images))]
    #     return cls(images, timestamps)


if __name__ == '__main__':
    img = transforms.ToTensor()(Image.open('/home2/lisiqi/data/final/ball/ball1/frame/0.jpg').convert('L'))