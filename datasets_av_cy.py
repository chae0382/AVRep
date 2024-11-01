import os
from torchvision import transforms
from transforms import *
from masking_generator_cy import TubeMaskingGenerator, TubeWindowMaskingGenerator, RandomMaskingGenerator2D, MeshMaskingGenerator
from kinetics_av_cy import VideoClsDataset, VideoMAE, VideoClsDatasetFrame, VideoMAERetrieval, MeshMAE, FullMeshMAE, FullMeshConformer, VocasetDataset
from ssv2 import SSVideoClsDataset


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        # me: new added
        if not args.no_augmentation:
            self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        else:
            print(f"==> Note: do not use 'GroupMultiScaleCrop' augmentation during pre-training!!!")
            self.train_augmentation = IdentityTransform()
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'part_window':
            print(
                f"==> Note: use 'part_window' masking generator (window_size={args.part_win_size[1:]}, apply_symmetry={args.part_apply_symmetry})")
            self.masked_position_generator = TubeWindowMaskingGenerator(
                args.window_size, args.mask_ratio, win_size=args.part_win_size[1:],
                apply_symmetry=args.part_apply_symmetry
            )
        self.max_length = args.num_frames

    def __call__(self, images):
        process_data, _ = self.transform(images)
        #frames = int(process_data.shape[0]/3)
        return process_data, self.masked_position_generator(self.max_length)

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

class DataAugmentationForMeshMAE(object):
    def __init__(self, args):
        self.masked_position_generator = MeshMaskingGenerator(
            args.window_size, args.mask_ratio)

        self.max_length = args.num_frames

    def __call__(self, vertices):
        vertices = torch.FloatTensor(vertices)
        #frames = int(process_data.shape[0]/3)
        # process_data : [B, 5, 5023, 3]
        # self.masked_position_generator(self.max_length) : [B, patch num]
        return vertices, self.masked_position_generator(self.max_length)

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

class DataAugmentationForVideoMAERetrieval(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = ScaleCrop(args.input_size, [1, 1, 1, 1])
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])

    def __call__(self, images):
        process_data, _ = self.transform(images)
        #frames = int(process_data.shape[0]/3)
        return process_data

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAERetrieval,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr

class MaskGeneratorForAudio(object):
    def __init__(self, mask_type, input_size, mask_ratio):
        if mask_type == 'random':
            self.masked_position_generator = RandomMaskingGenerator2D(
                input_size, mask_ratio
            )
        else:
            raise NotImplementedError

    def __call__(self, vis):
        return self.masked_position_generator(vis)

    def __repr__(self):
        repr = "(MaskGeneratorForAudio,\n"
        # repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    mask_generator_audio = MaskGeneratorForAudio(
        mask_type=args.mask_type_audio,
        input_size=args.window_size_audio,
        mask_ratio=args.mask_ratio_audio,
    )
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        root_data_dir=args.data_root,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate, # change to 1
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        # me: new added for VoxCeleb2
        # model=args.model,
        image_size=args.input_size,
        num_segments=args.num_samples,
        # me: for audio
        audio_conf=args.audio_conf,
        roll_mag_aug=args.roll_mag_aug,
        audio_sample_rate=args.audio_sample_rate,
        mask_generator_audio=mask_generator_audio,
        max_length=args.num_frames
    )
    print("Data Aug = %s" % str(transform))
    return dataset

def build_val_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    mask_generator_audio = MaskGeneratorForAudio(
        mask_type=args.mask_type_audio,
        input_size=args.window_size_audio,
        mask_ratio=args.mask_ratio_audio,
    )
    dataset = VideoMAE(
        root=None,
        setting=args.eval_data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate, # change to 1
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        # me: new added for VoxCeleb2
        # model=args.model,
        image_size=args.input_size,
        num_segments=args.num_samples,
        # me: for audio
        audio_conf=args.audio_conf,
        roll_mag_aug=args.roll_mag_aug,
        audio_sample_rate=args.audio_sample_rate,
        mask_generator_audio=mask_generator_audio,
        max_length=args.num_frames
    )
    print("Data Aug = %s" % str(transform))
    return dataset

def build_pretraining_mesh_dataset(args):
    transform = DataAugmentationForMeshMAE(args)
    mask_generator_audio = MaskGeneratorForAudio(
        mask_type=args.mask_type_audio,
        input_size=args.window_size_audio,
        mask_ratio=args.mask_ratio_audio,
    )
    dataset = MeshMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate, # change to 1
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        # me: new added for VoxCeleb2
        # model=args.model,
        image_size=args.input_size,
        num_segments=args.num_samples,
        # me: for audio
        audio_conf=args.audio_conf,
        roll_mag_aug=args.roll_mag_aug,
        audio_sample_rate=args.audio_sample_rate,
        mask_generator_audio=mask_generator_audio,
        max_length=args.num_frames
    )
    print("Data Aug = %s" % str(transform))
    return dataset

def build_pretraining_full_mesh_dataset(args):
    transform = DataAugmentationForMeshMAE(args)
    mask_generator_audio = MaskGeneratorForAudio(
        mask_type=args.mask_type_audio,
        input_size=args.window_size_audio,
        mask_ratio=args.mask_ratio_audio,
    )
    dataset = FullMeshMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate, # change to 1
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        # me: new added for VoxCeleb2
        # model=args.model,
        image_size=args.input_size,
        num_segments=args.num_samples,
        # me: for audio
        audio_conf=args.audio_conf,
        roll_mag_aug=args.roll_mag_aug,
        audio_sample_rate=args.audio_sample_rate,
        mask_generator_audio=mask_generator_audio,
        max_length=args.num_frames
    )
    print("Data Aug = %s" % str(transform))
    return dataset

def build_finetuning_full_mesh_vocaset_dataset(args, data, subjects_dict, mode):

    transform = DataAugmentationForMeshMAE(args)
    mask_generator_audio = MaskGeneratorForAudio(
        mask_type=args.mask_type_audio,
        input_size=args.window_size_audio,
        mask_ratio=args.mask_ratio_audio,
    )

    dataset = VocasetDataset(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate, # change to 1
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        # me: new added for VoxCeleb2
        # model=args.model,
        image_size=args.input_size,
        num_segments=args.num_samples,
        # me: for audio
        audio_conf=args.audio_conf,
        roll_mag_aug=args.roll_mag_aug,
        audio_sample_rate=args.audio_sample_rate,
        mask_generator_audio=mask_generator_audio,
        max_length=args.num_frames,
        data=data,
        batch_size=args.batch_size
    )
    print("Data Aug = %s" % str(transform))
    return dataset

def build_finetuning_mesh_vocaset_dataset(args, data, subjects_dict, mode):

    transform = DataAugmentationForMeshMAE(args)
    mask_generator_audio = MaskGeneratorForAudio(
        mask_type=args.mask_type_audio,
        input_size=args.window_size_audio,
        mask_ratio=args.mask_ratio_audio,
    )

    dataset = VocasetDataset(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate, # change to 1
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        # me: new added for VoxCeleb2
        # model=args.model,
        image_size=args.input_size,
        num_segments=args.num_samples,
        # me: for audio
        audio_conf=args.audio_conf,
        roll_mag_aug=args.roll_mag_aug,
        audio_sample_rate=args.audio_sample_rate,
        mask_generator_audio=mask_generator_audio,
        max_length=args.num_frames,
        data=data,
        batch_size=args.batch_size,
        is_mouth=True,
    )
    print("Data Aug = %s" % str(transform))
    return dataset

def build_pretraining_conformer_full_mesh_dataset(args):
    transform = DataAugmentationForMeshMAE(args)
    dataset = FullMeshConformer(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate, # change to 1
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        # me: new added for VoxCeleb2
        # model=args.model,
        image_size=args.input_size,
        num_segments=args.num_samples,
        # me: for audio
        num_mel_bins=args.num_mel_bins,
        roll_mag_aug=args.roll_mag_aug,
        audio_sample_rate=args.audio_sample_rate,
        mask_generator_audio=None,
        max_length=args.num_frames
    )
    print("Data Aug = %s" % str(transform))
    return dataset

def build_retrieval_dataset(args):
    transform = DataAugmentationForVideoMAERetrieval(args)
    dataset = VideoMAERetrieval(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate, # change to 1
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        # me: new added for VoxCeleb2
        # model=args.model,
        image_size=args.input_size,
        num_segments=1,
        # me: for audio
        audio_conf=args.audio_conf,
        roll_mag_aug=args.roll_mag_aug,
        audio_sample_rate=16000,
        max_length=args.num_frames
    )
    print("Data Aug = %s" % str(transform))
    return dataset

def build_classification_dataset(args, is_train, test_mode):
    anno_path = None
    if is_train is True:
        anno_path = '/home/chaeyeon/krafton/HiCMAE/preprocess/lrs3_viseme_trainval_classification_train.csv'
    elif test_mode is True:
        anno_path = '/home/chaeyeon/krafton/HiCMAE/preprocess/lrs3_viseme_trainval_classification_test.csv'
    else:
        anno_path = '/home/chaeyeon/krafton/HiCMAE/preprocess/lrs3_viseme_trainval_classification_val.csv'

    transform = DataAugmentationForVideoMAERetrieval(args)
    dataset = VideoMAERetrieval(
        root=None,
        setting=anno_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate, # change to 1
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        # me: new added for VoxCeleb2
        # model=args.model,
        image_size=args.input_size,
        num_segments=1,
        # me: for audio
        audio_conf=args.audio_conf,
        roll_mag_aug=args.roll_mag_aug,
        audio_sample_rate=16000,
        max_length=args.num_frames
    )
    print("Data Aug = %s" % str(transform))
    return dataset

def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400

    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101

    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51

    elif args.data_set == 'DFEW':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,  # me: actually no use
            new_width=320,  # me: actually no use
            args=args)
        nb_classes = 7


    elif args.data_set == 'MAFW':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,  # me: actually no use
            new_width=320,  # me: actually no use
            args=args,
            file_ext='png'
        )
        nb_classes = 11
        # for using 43 compound expressions
        if args.nb_classes == 43:
            nb_classes = args.nb_classes
            print(f"==> NOTE: using 43 compound expressions for MAFW!")


    elif args.data_set == 'RAVDESS':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,  # me: actually no use
            new_width=320,  # me: actually no use
            args=args)
        nb_classes = 8


    elif args.data_set == 'CREMA-D':
        mode = None
        anno_path = args.data_path
        if is_train is True:
            mode = 'train'
        #     anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
        #     anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
        #     anno_path = os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,  # me: actually no use
            new_width=320,  # me: actually no use
            args=args,
        )
        nb_classes = 6
        # for 4 basic emotions
        if args.nb_classes == 4:
            nb_classes = args.nb_classes
            print(f"==> NOTE: only using 4 emotions ('ANG', 'HAP', 'NEU', 'SAD')!")


    elif args.data_set == 'WEREWOLF-XL':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,  # me: actually no use
            new_width=320,  # me: actually no use
            args=args,
            task='regression',
        )
        nb_classes = 3


    elif args.data_set == 'AVCAFFE':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,  # me: actually no use
            new_width=320,  # me: actually no use
            args=args,
        )
        nb_classes = 5  # arousal or valence

    elif args.data_set == 'MSP-IMPROV':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,  # me: actually no use
            new_width=320,  # me: actually no use
            args=args,
        )
        nb_classes = 4
        print(f"==> NOTE: using 4 categorical emotions ('A', 'H', 'N', 'S')!")

    elif args.data_set == 'IEMOCAP':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,  # me: actually no use
            new_width=320,  # me: actually no use
            args=args,
        )
        nb_classes = 4


    elif args.data_set == 'MER2023':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,  # me: actually no use
            new_width=320,  # me: actually no use
            args=args,
        )
        nb_classes = 6

    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
