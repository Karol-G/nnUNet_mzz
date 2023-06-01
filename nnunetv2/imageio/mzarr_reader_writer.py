#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from mzarr import Mzarr


class MzarrIO(BaseReaderWriter):
    """
    Reader and writer for Multi-resolution Zarr (Mzarr) images.
    """

    supported_file_endings = [
        '.mzarr'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        print("Read")
        images = []
        for f in image_fnames:
            # spacing = (999, 1, 1)
            spacing = (1, 1, 1)
            img_mzarr = Mzarr(f)
            attributes = img_mzarr.attrs()
            img = img_mzarr.numpy()
            if attributes["channel_axis"] is not None:
                img = np.moveaxis(img, attributes["channel_axis"], 0)
            if attributes["channel_axis"] is None:
                img = img[np.newaxis, ...]
            if attributes["num_spatial"] == 2:
                img = img[:, np.newaxis, ...]
            images.append(img)

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        return np.vstack(images).astype(np.float32), {'spacing': spacing}

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        seg_mzarr = Mzarr(seg_fname)
        attributes = seg_mzarr.attrs()
        seg = seg_mzarr.numpy()
        seg = seg.astype(np.int32)
        if attributes["num_spatial"] == 2:
            seg = seg[np.newaxis, ...]
        seg = seg[np.newaxis, ...]  # Add 1 channel dim
        # properties = {'spacing': (999, 1, 1)}
        properties = {'spacing': (1, 1, 1)}
        return seg, properties

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        print("Write")
        seg = seg.squeeze()
        print("Writing Mzarr image {}...".format(output_fname))
        mzz.Mzz(seg).save(output_fname, properties=properties, is_seg=True)
        print("Finished writing image {}".format(output_fname))


if __name__ == '__main__':
    # images = ("/home/k539i/Documents/datasets/preprocessed/nnUNet/nnUNet_raw_data/nnUNet_raw_data/Dataset301_vesuvius/imagesTr/001_0000.mzz", )
    # segmentation = "/home/k539i/Documents/datasets/preprocessed/nnUNet/nnUNet_raw_data/nnUNet_raw_data/Dataset301_vesuvius/labelsTr/001.mzz"
    # imgio = MzzIO()
    # img, props = imgio.read_images(images)
    # seg, segprops = imgio.read_seg(segmentation)
    # imgio.write_seg(seg, "/home/k539i/Documents/datasets/original/HMGU_2022_DIADEM/dataset_patches/Dataset107_DIADEMv3/tmp.mzz", {"spacing": (999, 1, 1)})
