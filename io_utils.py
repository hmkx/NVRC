"""
IO utilities for video read/write
"""
from utils import *
from PIL import Image


def yuv420_to_yuv444(yuv420_frames, mode='nearest'):
    """
    yuv_frames: list of y, u, v frames in shape [N, 1, H, W]
    """
    yuv444_tensor = compressai.transforms.functional.yuv_420_to_444(
                        yuv420_frames, mode=mode
                    )
    return yuv444_tensor


def yuv444_to_yuv420(yuv444_tensor, mode='avg_pool'):
    """
    yuv444_tensor: yuv444 tensor in shape [N, 3, H, W]
    """
    yuv420_frames = compressai.transforms.functional.yuv_444_to_420(
                        yuv444_tensor, mode=mode
                    )
    return yuv420_frames


def compute_scale_shift(channels, bit_depth, scale, shift):
    scale = torch.tensor(scale if scale is not None else [1.] * channels, dtype=torch.float32) / (2 ** bit_depth - 1)
    shift = torch.tensor(shift if shift is not None else [0.] * channels, dtype=torch.float32)
    return scale, shift


class VideoBase:
    """
    Base class for video read/write.
    The process of reading/writing a video patch is:
        - Crop the video over the time axis with start_frame and num_frames.
        - Pad the video with padding.
        - Extract the patch from the padded video with the idx (t, h, w),

    Inputs:
        idx: The patch index in [t, h, w], relative to the padded video and the start_frame/num_frames.
        video_size: The video size but excluding padding, with a format: [T, H, W].
        patch_size: The patch size with a format: [T_patch, H_patch, W_patch].
        padding: The size of padding, with a format:
                [[T_pad_0, T_pad_1], [H_pad_0, H_pad_1], [W_pad_0, W_pad_1]].    
    """
    idx_min = (0, 0, 0)
    idx_max = (1, 1, 1)
    video_size = (1, 1, 1)
    patch_size = (1, 1, 1)
    padding = ((0, 0), (0, 0), (0, 0))
    channels = 3
    bit_depth = 8
    start_frame = 0
    num_frames = 1
    write_mode = False
    cache = None

    def get_path(self):
        return self.path

    def get_video_size(self):
        return self.video_size

    def get_patch_size(self):
        return self.patch_size

    def set_patch_size(self, patch_size):
        assert all(isinstance(patch_size[d], int) and patch_size[d] > 0 for d in range(3)), \
            'Patch size should be positive integer.'
        self.patch_size = patch_size
        self.cache = None

    def get_padding(self):
        return self.padding
    
    def set_padding(self, padding):
        assert all(len(padding[d]) == 2 and isinstance(padding[d][0], int) and isinstance(padding[d][1], int) and \
                   padding[d][0] >= 0 and padding[d][1] >= 0 for d in range(3)), \
            'Padding should have two non-negative integers.'
        self.padding = tuple(tuple(padding[d]) for d in range(3))
        self.cache = None

    def get_start_frame(self):
        return self.start_frame

    def get_num_frames(self):
        return self.num_frames

    def set_frames(self, start_frame, num_frames):
        assert start_frame == -1 or start_frame >= 0
        self.start_frame = start_frame if start_frame != -1 else 0
        assert num_frames == -1 or num_frames <= self.video_size[0] - self.start_frame
        self.num_frames = num_frames if num_frames != -1 else self.video_size[0] - self.start_frame
        self.cache = None

    def get_num_channels(self):
        return self.channels

    def get_bit_depth(self):
        return self.bit_depth

    def get_fmt(self):
        return self.fmt

    def get_idx_max(self):
        return self.idx_max

    def _pad(self, patch, start_pad, end_pad, constant_values=2**7):
        assert patch.ndim >= 2, 'Patch should have at least two dimensions (H, W, ...).'
        assert len(start_pad) == len(end_pad) == 2, \
            'Paddings should have two dimensions (H_pad_start, W_pad_start) and (H_pad_end, W_pad_end).'
        return np.pad(patch,  [[start_pad[0], end_pad[0]], [start_pad[1], end_pad[1]]] + [[0, 0]] * (patch.ndim - 2),
                      constant_values=constant_values)

    def _unpad(self, patch, start_pad, end_pad):
        assert patch.ndim >= 2, \
            'Patch should have at least two dimensions (H, W, ...).'
        assert len(start_pad) == len(end_pad) == 2, \
            'Paddings should have two dimensions (H_pad_start, W_pad_start) and (H_pad_end, W_pad_end).'
        return patch[start_pad[0]:patch.shape[0] - end_pad[0], start_pad[1]:patch.shape[1] - end_pad[1]]

    def create_cache(self, enable=True):
        assert (self.num_frames + sum(self.padding[0])) % self.patch_size[0] == 0, \
            f'Number of frames ({self.num_frames}) + padding ({self.padding[0]}) should be divisible by ' \
            f'patch size ({self.patch_size[0]}).'
        assert (self.video_size[1] + sum(self.padding[1])) % self.patch_size[1] == 0, \
            f'Video height ({self.video_size[1]}) + padding ({self.padding[1]}) should be divisible by ' \
            f'patch size ({self.patch_size[1]}).'
        assert (self.video_size[2] + sum(self.padding[2])) % self.patch_size[2] == 0, \
            f'Video width ({self.video_size[2]}) + padding ({self.padding[2]}) should be divisible by ' \
            f'patch size ({self.patch_size[2]}).'
        self.idx_min = (0, 0, 0)
        self.idx_max = (
            (self.num_frames + sum(self.padding[0])) // self.patch_size[0],
            (self.video_size[1] + sum(self.padding[1])) // self.patch_size[1],
            (self.video_size[2] + sum(self.padding[2])) // self.patch_size[2]
        )
        if enable:
            self._create_cache()

    def read_patch(self, idx):
        assert self.cache is not None, 'Cache is not created.'
        assert not self.write_mode, 'Cannot read from a video with write mode.'
        assert len(idx) == 3 and all(idx[d] >= self.idx_min[d] and idx[d] < self.idx_max[d] for d in range(3)), \
            f'Index {idx} out of range [{self.idx_min}, {self.idx_max}).'
        return self._read_patch(idx)

    def write_patch(self, idx, input):
        assert self.cache is not None, 'Cache is not created.'
        assert self.write_mode, 'Cannot write to a video with read mode.'
        assert len(idx) == 3 and all(idx[d] >= self.idx_min[d] and idx[d] < self.idx_max[d] for d in range(3)), \
            f'Index {idx} out of range [{self.idx_min}, {self.idx_max}).'
        return self._write_patch(idx, input)

    def flush(self, path=None):
        assert self.cache is not None, 'Cache is not created.'
        assert self.write_mode, 'Cannot flush a video with read mode.'
        self._flush(os.path.join(path, self.path.split('/')[-1]) if path is not None else self.path)


class PNGVideo(VideoBase):
    """
    This class read/write video from a folder of PNG images.
    
    Inputs:
        path: png directory path
        video_size: The full video size, excluding padding, with a format: [T, H, W].
        padding: The size of padding, with a format: 
                    [[T_pad_0, T_pad_1], [H_pad_0, H_pad_1], [W_pad_0, W_pad_1]].
        channels: The number of channels, e.g. 3 for RGB videos.
        bit_depth: The bit depth, e.g. 8 for 8-bit videos.
        write_mode: True for read and write, False for read only
    """
    def __init__(self, path, video_size, channels, bit_depth, write_mode):
        self.path = os.path.expanduser(path)
        self.write_mode = write_mode

        # Compute settings
        # See https://pillow.readthedocs.io/en/stable/handbook/concepts.html
        if self.write_mode:
            assert (all(video_size[d] > 0 for d in range(3))), \
                'The given video size should be positive.'
            assert channels > 0, \
                'The given number of channels should be positive.'
            assert bit_depth in [8, 16], \
                'The given bit depth should be 8 or 16.'
            self.img_names = [f'{i:04d}.png' for i in range(video_size[0])]
            self.video_size = tuple(video_size)
            self.channels = channels 
            self.bit_depth = bit_depth
            self.dtype = {8: np.uint8, 16: np.uint16}[self.bit_depth]
        else:
            assert len([name for name in os.listdir(self.path) if name.endswith('.png')]) > 0, \
                'No PNG images found in the given directory.'
            self.img_names = sorted([name for name in os.listdir(self.path) if name.endswith('.png')])
            img = np.array(Image.open(os.path.join(self.path, self.img_names[0])))
            img_size = img.shape[:2]

            assert all(video_size[d] == -1 or video_size[d] == [len(self.img_names), img_size[0], img_size[1]][d] \
                       for d in range(3)), \
                'The given video size does not match the actual video size.'
            assert channels == -1 or channels == img.shape[-1], \
                'The given number of channels does not match the actual number of channels.'
            assert bit_depth -1 or bit_depth == {np.int32: 16, np.uint8: 8}[img.dtype.type], \
                'The given bit depth does not match the actual bit depth.'
            self.video_size = (len(self.img_names), img_size[0], img_size[1])
            self.channels = img.shape[-1]
            self.dtype = img.dtype.type
            self.bit_depth = {np.int32: 16, np.uint8: 8}[self.dtype]

        self.set_patch_size([1, self.video_size[1], self.video_size[2]])
        self.set_padding([[0, 0], [0, 0], [0, 0]])
        self.set_frames(0, self.video_size[0])

        # Cache
        self.cache = None

    def _create_cache(self):
        # Create np array
        if self.write_mode:
            self.cache = np.zeros(
                (self.num_frames,) + self.video_size[1:3] + (self.channels,),
                dtype=self.dtype
            )
        else:
            self.cache = np.stack(
                [np.array(Image.open(os.path.join(self.path, self.img_names[self.start_frame + i]))) \
                 for i in range(self.num_frames)],
                axis=0
            )

    def _get_patch_cfg(self, idx):
        idx = torch.tensor(idx)
        patch = []

        start = np.array(idx) * np.array(self.patch_size) - np.array(self.padding)[:, 0]
        end = start + np.array(self.patch_size)
        crop_start = np.maximum(start[1:], 0)
        pad_start = (start[1:] < 0) * (0 - start[1:])
        crop_end = np.minimum(end[1:], self.video_size[1:])
        pad_end = (end[1:] > self.video_size[1:]) * (end[1:] - self.video_size[1:])

        for i in range(self.patch_size[0]):
            t = start[0] + i
            if t < 0 or t >= self.num_frames:
                frame = None
            else:
                frame = self.cache[t]
            patch.append((frame, crop_start, crop_end, pad_start, pad_end))

        return patch

    def _read_patch(self, idx):
        patch_cfg = self._get_patch_cfg(idx)
        patch_out = []

        for i in range(self.patch_size[0]):
            frame, crop_start, crop_end, pad_start, pad_end = patch_cfg[i]
            if frame is None:
                patch_out.append(np.full((self.patch_size[1], self.patch_size[2], self.channels), \
                                         2 ** (self.bit_depth - 1), self.dtype))
            else:
                patch_out.append(self._pad(frame[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]], \
                                           pad_start, pad_end, 2 ** (self.bit_depth - 1)))

        return np.stack(patch_out, axis=0)

    def _write_patch(self, idx, input):
        patch_cfg = self._get_patch_cfg(idx)

        for i in range(self.patch_size[0]):
            frame, crop_start, crop_end, pad_start, pad_end = patch_cfg[i]
            if frame is not None:
                frame[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]] = self._unpad(input[i], pad_start, pad_end)

    def _flush(self, path=None):
        path = path if path is not None else self.path
        os.makedirs(path, exist_ok=True)
        for i in range(self.num_frames):
            Image.fromarray(self.cache[i].astype(self.dtype)) \
                 .save(os.path.join(path, self.img_names[self.start_frame + i]), format='png')


class YUVVideo(VideoBase):
    """
    This class utilize numpy.mempy for efficient randon patch access.
    Logically, it first pad the video, then extract patches from the padded video.
      
    Inputs:
        path: video file path
        video_size: [T, H, W]
        fmt: yuv format
        write_mode: True for read and write, False for read only
    """
    def __init__(self, path, video_size, fmt, write_mode):
        self.path = os.path.expanduser(path)
        self.fmt = fmt
        self.write_mode = write_mode
        self.channels = 3

        # Settings
        assert len(video_size) == 3
        cfg = {
            'yuv420p': {'bit_depth': 8, 'sample_per_pixel': 1.5, 'strides': [[1, 1, 1], [1, 2, 2], [1, 2, 2]],
                        'bytes_per_sample': 1, 'dtype': np.uint8},
            'yuv420p10le': {'bit_depth': 10, 'sample_per_pixel': 1.5, 'strides': [[1, 1, 1], [1, 2, 2], [1, 2, 2]],
                            'bytes_per_sample': 2, 'dtype': np.uint16},
            'yuv420p16le': {'bit_depth': 16, 'sample_per_pixel': 1.5, 'strides': [[1, 1, 1], [1, 2, 2], [1, 2, 2]],
                            'bytes_per_sample': 2, 'dtype': np.uint16},
        }[self.fmt]
        self.bit_depth = cfg['bit_depth']
        self.sample_per_pixel = cfg['sample_per_pixel']
        self.strides = tuple(cfg['strides'])
        self.bytes_per_sample = cfg['bytes_per_sample']
        self.dtype = cfg['dtype']

        if self.write_mode:
            assert all(video_size[d] > 0 for d in range(1, 3)), \
                'Cannot write to a video with unknown or zero size.'
            self.video_size = tuple(video_size)
        else:
            assert all(video_size[d] > 0 for d in range(1, 3)), \
                'Cannot read a video with unknown or zero spatial size.'
            assert (video_size[0] == -1 or \
                    video_size[0] == os.path.getsize(self.path) // \
                                     int(math.prod(video_size[1:3]) * self.sample_per_pixel * self.bytes_per_sample)), \
                'The given video size does not match the actual number of frames.'
            self.video_size = tuple(
                os.path.getsize(self.path) // \
                int(math.prod(video_size[1:3]) * self.sample_per_pixel * self.bytes_per_sample),
                video_size[1], video_size[2]
            )

        self.set_patch_size([1, self.video_size[1], self.video_size[2]])
        self.set_padding([[0, 0], [0, 0], [0, 0]])
        self.set_frames(0, self.video_size[0])

        # Cache
        self.cache = None

    def _create_cache(self):
        # Channel settings
        assert np.all(self.video_size[None, :] % self.strides[:, :] == 0), \
            f'Video size ({self.video_size}) does not match the chroma subsampling settings ({self.strides})'
        assert np.all(self.padding[None, :, :] % self.strides[:, :, None] == 0), \
            f'Padding size ({self.padding}) does not match the chroma subsampling settings ({self.strides})'
        self.channel_video_sizes = self.video_size[None] // self.strides
        self.channel_patch_sizes = self.patch_size[None]  // self.strides
        self.channel_paddings = self.padding[None] // np.array(self.strides)[:, :, None]

        self.channel_strides = self.channel_video_sizes[:, 1] * self.channel_video_sizes[:, 2]
        self.channel_offsets = np.stack([np.cumsum(self.channel_strides) - self.channel_strides,
                                         np.cumsum(self.channel_strides)], axis=1)

        # Create memmap
        os.makedirs('/'.join(self.path.split('/')[:-1]), exist_ok=True)
        self.cache = np.memmap(self.path, dtype=self.dtype, mode='w+' if self.write_mode else 'r',
                               shape=(self.video_size[0], int(self.video_size[1:].prod() * self.sample_per_pixel))) \
                              [self.start_frame:self.start_frame + self.num_frames]

    def _get_patch_cfg(self, idx):
        idx = torch.tensor(idx)
        patch = [[] for _ in range(len(self.strides))]

        for i in range(len(self.strides)):
            start = idx * self.channel_patch_sizes[i] - self.channel_paddings[i, :, 0]
            end = start + self.channel_patch_sizes[i]
            crop_start = np.maximum(start[1:], 0)
            pad_start = (start[1:] < 0) * (0 - start[1:])
            crop_end = np.minimum(end[1:], self.channel_video_sizes[i, 1:])
            pad_end = (end[1:] > self.channel_video_sizes[i, 1:]) * (end[1:] - self.channel_video_sizes[i, 1:])
            for j in range(self.patch_size[0]):
                t = start[0] + j
                if t < 0 or t >= self.num_frames:
                    frame = None
                else:
                    frame = self.cache[t][self.channel_offsets[i, 0]:self.channel_offsets[i, 1]] \
                                         .reshape(self.channel_video_sizes[i, 1:])
                patch[i].append((frame, crop_start, crop_end, pad_start, pad_end))

        return patch

    def _read_patch(self, idx):
        patch_cfg = self._get_patch_cfg(idx)
        patch_out = [[] for i in range(len(self.strides))]

        for i in range(len(self.strides)):
            for j in range(self.patch_size[0]):
                frame, crop_start, crop_end, pad_start, pad_end = patch_cfg[i][j]
                if frame is None:
                    patch_out[i].append(np.full((self.channel_patch_sizes[i, 1], self.channel_patch_sizes[i, 2]), \
                                                2 ** (self.bit_depth - 1), self.dtype))
                else:
                    patch_out[i].append(self._pad(frame[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]], \
                                                  pad_start, pad_end, 2 ** (self.bit_depth - 1)))

        return [np.stack(p, axis=0) for p in patch_out]

    def _write_patch(self, idx, input):
        patch_cfg = self._get_patch_cfg(idx)

        for i in range(len(self.strides)):
            for j in range(self.patch_size[0]):
                frame, crop_start, crop_end, pad_start, pad_end = patch_cfg[i][j]
                if frame is not None:
                    frame[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]] = self._unpad(input[i][j],
                                                                                              pad_start, pad_end)

    def _flush(self, path):
        if path is not None:
            os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
            self.cache.tofile(path)
        else:
            self.cache.flush()