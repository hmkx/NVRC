"""
Datasets
"""
from utils import *
from io_utils import *
import bisect


class VideoDatasetBase(torch.utils.data.Dataset):
    """
    The base dataset class for loading videos.
    It loads video frames in 3D patches.
    """
    def __init__(self, logger, video, channel_scale=None, channel_shift=None, is_subset=False):
        self.logger = logger
        self.video = video

        assert channel_scale is None or channel_shift is None or \
               (len(channel_scale) == self.video.get_num_channels() and \
                len(channel_shift) == self.video.get_num_channels())
        self.channel_scale, self.channel_shift = compute_scale_shift(self.video.get_num_channels(),
                                                                     self.video.get_bit_depth(),
                                                                     channel_scale, channel_shift)

        if not is_subset:
            self.logger.info(f'{type(self).__name__}:')
            self.logger.info(f'     Path: {self.video.path}')
            self.logger.info(f'     Video size: {self.video.get_video_size()}    Patch size: {self.video.get_patch_size()}    Padding: {self.video.get_padding()}')
            self.logger.info(f'     Channels: {self.video.get_num_channels()}    Bit depth: {self.video.get_bit_depth()}')
            self.logger.info(f'     Scale: {self.channel_scale}    Shift: {self.channel_shift}')

    def get_video_size(self):
        return self.video.get_video_size()

    def get_patch_size(self):
        return self.video.get_patch_size()

    def get_num_channels(self):
        return self.video.get_num_channels()

    def get_start_frame(self):
        return self.video.get_start_frame()

    def get_num_frames(self):
        return self.video.get_num_frames()

    def set_frames(self, start_frame, num_frames):
        self.video.set_frames(start_frame, num_frames)

    def create_cache(self):
        self.video.create_cache()

    def _get_patch(self, idx_thw):
        raise NotImplementedError

    def __len__(self):
        return math.prod(self.video.get_idx_max())

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        idx_max = self.video.get_idx_max()
        idx_thw = (idx // (idx_max[1] * idx_max[2]),
                   (idx % (idx_max[1] * idx_max[2])) // idx_max[2],
                   (idx % (idx_max[1] * idx_max[2])) % idx_max[2])
        return torch.tensor(idx_thw, dtype=int), self._get_patch(idx_thw)


class PNGVideoDataset(VideoDatasetBase):
    """
    The dataset class for loading videos (in PNG files). Each dataset instance loads all video frames in a folder.
    """
    def _get_patch(self, idx_thw):
        scale = self.channel_scale.view(self.video.get_num_channels(), 1, 1, 1)
        shift = self.channel_shift.view(self.video.get_num_channels(), 1, 1, 1)
        rgb_patch = torch.tensor(self.video.read_patch(idx_thw)).permute(3, 0, 1, 2).float() * scale + shift
        return torch.clip_(rgb_patch, 0., 1.)


class YUVVideoDataset(VideoDatasetBase):
    """
    The dataset class for loading videos (in YUV format). Each dataset instance loads all video frames from a yuv file.
    """
    def _get_patch(self, idx_thw):
        scale = self.channel_scale
        shift = self.channel_shift

        # List of [T, 1, H, W] tensors
        yuv420_patch = [torch.tensor(patch_i)[:, None].float() * scale[i] + shift[i] \
                        for i, patch_i in enumerate(self.video.read_patch(idx_thw))]
        # List of [T, 1, H, W] tensors -> [T, 3, H, W] tensor -> [3, T, H, W] tensor
        yuv444_patch = yuv420_to_yuv444(yuv420_patch, mode='nearest').permute(1, 0, 2, 3)

        return torch.clip_(yuv444_patch, 0., 1.)


class VideoSubset(torch.utils.data.Subset):
    def __init__(self, dataset, start_frame, num_frames):
        super().__init__(dataset, [])
        self.dataset = dataset
        self.indices = []
        dataset = self.dataset
        patches_per_frame = math.prod(dataset.video.get_idx_max()[1:])
        assert start_frame % dataset.get_patch_size()[0] == 0 and num_frames % dataset.get_patch_size()[0] == 0, \
            f'start_frame and num_frames should be multiples of patch_size[0]'
        start_patch = (start_frame // dataset.get_patch_size()[0]) * patches_per_frame
        num_patches = (num_frames // dataset.get_patch_size()[0]) * patches_per_frame
        self.indices += list(range(start_patch, start_patch + num_patches))

    def get_video_size(self):
        return self.dataset.get_video_size()

    def get_patch_size(self):
        return self.dataset.get_patch_size()


def create_overfit_dataset(args, logger, video, channel_scale=None, channel_shift=None, is_subset=False):
    # Create dataset
    if isinstance(video, PNGVideo):
        dataset = PNGVideoDataset(logger, video, 
                                  channel_scale=channel_scale, channel_shift=channel_shift,
                                  is_subset=is_subset)
    elif isinstance(video, YUVVideo):
        dataset = YUVVideoDataset(logger, video, 
                                  channel_scale=channel_scale, channel_shift=channel_shift,
                                  is_subset=is_subset)
    else:
        raise ValueError
    return dataset