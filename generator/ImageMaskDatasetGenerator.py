# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ImageMaskDatasetGenerator.py
# 2026/01/12

import os
import glob
import cv2
import shutil
import numpy as np

import traceback

class ImageMaskDatasetGenerator:
  def __init__(self, size=224):
    self.file_format = ".png"
    self.RESIZE      = (size, size)
    self.index       = 10000

  def generate(self, images_dir, masks_dir, output_images_dir, output_masks_dir):
    image_files = sorted(glob.glob(images_dir + "/*.png") )
    mask_files  = sorted(glob.glob(masks_dir  + "/*.png") )
    num_images  = len(image_files)
    num_masks   = len(mask_files)
    print(num_images, num_masks)
    if num_images != num_masks:
      raise Exception("Unmatched number of images and mask")
    input("HIT any key")
    for i in range(num_masks):
      self.index += 1
      mask_file  = mask_files[i]
      image_file = image_files[i]

      self.generate_mask_file(mask_file , self.index, output_masks_dir )      
      self.generate_image_file(image_file, self.index, output_images_dir ) 

  def colorize_mask(self, mask):
    h, w = mask.shape[:2]
    #6 classes
    #            1. Patches, 2. Inclusion, 3. Scratches
    rgb_colors = [ (0,0,255), (0,255,0), (255,0,0),]
    #BGR color
    colorized = np.zeros((h, w, 3), dtype=np.uint8)
    index = 1
    for rgb_color in rgb_colors:
      [r, g, b] = rgb_color
      colorized[np.equal(mask, index)] = (b, g, r)
      index += 1
    return colorized


  def generate_mask_file(self, image_file, index, output_dir):
    image = cv2.imread(image_file)
    image = cv2.resize(image, self.RESIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = self.colorize_mask(image)
    filename = str(index) + self.file_format
    output_filepath = os.path.join(output_dir, filename)
    cv2.imwrite(output_filepath, image)
    print("Saved {}".format(output_filepath))

  def generate_image_file(self, image_file, index, output_dir):
    image = cv2.imread(image_file)
    image = cv2.resize(image, self.RESIZE)
 
    filename = str(index) + self.file_format
    output_filepath = os.path.join(output_dir, filename)
      
    cv2.imwrite(output_filepath, image)
    print("Saved {}".format(output_filepath))
        
  
if __name__ == "__main__":
  try:
     images_dir = "./Synthetic_NEU-Seg_Images/images/"
     masks_dir  = "./Synthetic_NEU-Seg_Images/annotations/"
     master_dir = "./Synthetic-NEU-Seg-master/"
     if os.path.exists(master_dir):
       shutil.rmtree(master_dir)
     os.makedirs(master_dir)

     output_images_dir = os.path.join(master_dir, "images")
     output_masks_dir  = os.path.join(master_dir, "masks")

     os.makedirs(output_images_dir)
     os.makedirs(output_masks_dir)

     generator = ImageMaskDatasetGenerator()
     generator.generate(images_dir, masks_dir, output_images_dir, output_masks_dir,)

  except:
    traceback.print_exc()
