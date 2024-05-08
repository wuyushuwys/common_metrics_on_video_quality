import warnings
warnings.filterwarnings('ignore')
import random
import os
from tqdm import tqdm
from glob import glob

import numpy as np

import torch
from torchvision.io import read_video




class Evaluator:

    def __init__(self, fake_folder, real_folder, ext='mp4', random_sample=False):
        
        assert os.path.exists(fake_folder), f"{fake_folder} not exist"
        assert os.path.exists(real_folder), f"{real_folder} not exist"

        # retrive video files
        self.fake_video_list = glob(f"{fake_folder}/**/*.{ext}", recursive=True)
        self.reak_video_list = glob(f"{real_folder}/**/*.{ext}", recursive=True)

        assert len(self.fake_video_list) > 0
        assert len(self.reak_video_list) > 0

        if random_sample or len(self.reak_video_list) < len(self.fake_video_list):
            self.reak_video_list = random.sample(self.reak_video_list, len(self.fake_video_list))
        else:
            self.reak_video_list = self.reak_video_list[:len(self.fake_video_list)]
        
        self.shuffle()

        self._minimum_frames = 10

    def shuffle(self):
        random.shuffle(self.fake_video_list)
        random.shuffle(self.reak_video_list)
        
    def _load_videos(self, i, batch_size):
        
        # load fake video, their length should be the same
        fake_video_files = self.fake_video_list[i : i + batch_size]
        fake_video_batch = torch.stack([read_video(vname)[0].permute(3, 0, 1, 2)/255 for vname in fake_video_files])
        bsz, c, f, h, w = fake_video_batch.shape

        assert f > self._minimum_frames, f"Expect at least {self._minimum_frames} frames but got {f}"

        # load real video, trim the length to be the same
        real_video_files = self.reak_video_list[i : i + batch_size]
        real_video_batch = torch.stack([read_video(vname)[0].permute(3, 0, 1, 2)[:f]/255 for vname in real_video_files])

        assert real_video_batch.shape == fake_video_batch.shape, f"Except {real_video_batch.shape} == {fake_video_batch.shape}"

        # output shape is [batch, channels, frames, height, width]
        return fake_video_batch, real_video_batch    
    
    def _load_fvd_model(self, method='styleganv', device=torch.device("cuda")):

        if method == 'styleganv':
            from fvd.styleganv.fvd import load_i3d_pretrained
        elif method == 'videogpt':
            from fvd.videogpt.fvd import load_i3d_pretrained

        self.i3d = load_i3d_pretrained(device=device)

    def __call__(self, batch_size=8, method='styleganv', device=torch.device("cuda")):
        
        print(f"Using {method} FVD...")

        if method == 'styleganv':
            from fvd.styleganv.fvd import get_fvd_feats, frechet_distance
        elif method == 'videogpt':
            from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
            from fvd.videogpt.fvd import frechet_distance

        # Load i3d model
        self._load_fvd_model(method, device=device)

        print(f"Calculate FVD...")

        fvd_results = {}

        for i in tqdm(range(0, len(self.fake_video_list), batch_size), dynamic_ncols=True, position=0):
            fake_video_batch, real_video_batch = self._load_videos(i, batch_size=batch_size)
            for clip_timestamp in tqdm(range(10, fake_video_batch.shape[-3]+1), dynamic_ncols=True, position=1, leave=False):
       
                # videos_clip [batch_size, channel, timestamps[:clip], h, w]
                videos_clip1 = fake_video_batch[:, :, : clip_timestamp]
                videos_clip2 = real_video_batch[:, :, : clip_timestamp]

                # get FVD features
                feats1 = get_fvd_feats(videos_clip1, i3d=self.i3d, device=device)
                feats2 = get_fvd_feats(videos_clip2, i3d=self.i3d, device=device)
            
                # calculate FVD when timestamps[:clip]
                fvd_results[clip_timestamp] = frechet_distance(feats1, feats2)
        
        fvd_values = list(fvd_results.values())
        fvd_mean = np.mean(fvd_values)
        fvd_std = np.std(fvd_values)
        print(f"FVD: {fvd_mean:.04f} ± {fvd_std:.04f}")
        return fvd_results
        

# test code / using example
if __name__ == "__main__":
    runner = Evaluator('/nfs/ywu6/animate-anything/eval', '/nfs/ywu6/animate-anything/eval')

    runner()
