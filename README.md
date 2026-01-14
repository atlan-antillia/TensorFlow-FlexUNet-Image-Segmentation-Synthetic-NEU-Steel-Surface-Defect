<h2>TensorFlow-FlexUNet-Image-Segmentation-Synthetic-NEU-Steel-Surface-Defect  (2026/01/14)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Synthetic-NEU-Steel-Surface-Defect</b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and <b>Synthetic-NEU-Seg</b> dataset with colorized masks, which was derived by us from <br><br>
<a href="https://opendatasets.vicomtech.org/di11-synthetic-neu-seg-images-via-stable-diffusion/33e9a1ec">
Synthetic NEU-Seg Images via Stable Diffusion</a>
<br><br>
<hr>
<b>Actual Image Segmentation for Synthetic-NEU-Steel-Surface-Defect Images of 224x224 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>
<b>rgb_map={Patches:blue, Inclusion:green,  Scratches:red}</b><br><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/prediction_top.png" width="1024" height="auto"><br>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<b>Synthetic_NEU-Seg_Images</b> dataset, which can be provided by your request  through the following web site<br><br>
<a href="https://opendatasets.vicomtech.org/di11-synthetic-neu-seg-images-via-stable-diffusion/33e9a1ec">
Synthetic NEU-Seg Images via Stable Diffusion</a>
<br><br>
Please refer to: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11436218/">
<b>Latent Diffusion Models to Enhance the Performance of Visual Defect Segmentation Networks in Steel Surface Inspection</b></a>
<br><br>
The following explanation was taken from <a href="https://github.com/Vicomtech/synthetic-neu-seg-images-via-stable-diffusion">
Synthetic NEU-Seg Images via Stable Diffusion</a><br><br>
<b>Usage</b><br>
The dataset is organized in the same way as the original NEU-seg dataset: the images folder contains the synthetic images of the defects, 
and the annotations folder contains the pixel-level annotations of each. <br>
The class values of the pixels in the masks are assigned as follows: <br>
<b>0. Background</b>, <br>
<b>1. Patches</b>, <br>
<b>2. Inclusion</b>, <br>
<b>3. Scratches</b>. <br>
Each image/annotation filename is self-descriptive, with regards to the class it belongs to.
<br><br>
<b>Authors</b><br>
The following researchers have collaborated in the dataset creation and curation process:<br>
<ul>
<li>Jon Leiñena Otamendi</li>
<li>Fátima Saiz Álvaro</li>
</ul>
<br>
<b>Citation</b>
<pre>
@Article{s24186016,
  AUTHOR = {Leiñena, Jon and Saiz, Fátima A. and Barandiaran, Iñigo},
  TITLE = {Latent Diffusion Models to Enhance the Performance of Visual Defect Segmentation Networks in Steel Surface Inspection},
  JOURNAL = {Sensors},
  VOLUME = {24},
  YEAR = {2024},
  NUMBER = {18},
  ARTICLE-NUMBER = {6016},
  URL = {https://www.mdpi.com/1424-8220/24/18/6016},
  ISSN = {1424-8220},
  ABSTRACT = {This paper explores the use of state-of-the-art latent diffusion models, specifically stable diffusion, 
  to generate synthetic images for improving the robustness of visual defect segmentation in manufacturing components. 
  Given the scarcity and imbalance of real-world defect data, synthetic data generation offers a promising solution for 
  training deep learning models. We fine-tuned stable diffusion using the LoRA technique on the NEU-seg dataset and 
  evaluated the impact of different ratios of synthetic to real images on the training set of DeepLabV3+ and FPN segmentation models. 
  Our results demonstrated a significant improvement in mean Intersection over Union (mIoU) when the training dataset was 
  augmented with synthetic images. This study highlights the potential of diffusion models for enhancing the quality and diversity 
  of training data in industrial defect detection, leading to more accurate and reliable segmentation results. 
  The proposed approach achieved improvements of 5.95% and 6.85% in mIoU of defect segmentation on each model over the original dataset.},
  DOI = {10.3390/s24186016}
}
</pre>
<br>
<b>License</b><br>
All datasets on this page are copyrigh by Vicomtech and published under the 
<b>Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 License</b>. <br> 
This means that you must attribute the work in the manner specified by the authors, 
you may not use this work for commercial purposes and if you remix, transform, or <br>
build upon the material, you may not distribute the modified material.<br>
<br>
<h3>
2 Synthetic-NEU-Seg ImageMask Dataset
</h3>
 If you would like to train this Synthetic-NEU-Seg Segmentation model by yourself,
please download the master  dataset from
<a href="https://opendatasets.vicomtech.org/di11-synthetic-neu-seg-images-via-stable-diffusion/33e9a1ec">
Synthetic NEU-Seg Images via Stable Diffusion</a>
<br>
We used the following 2 Python scripts to generate the Synthetic-NEU-Seg with colorized masks  from the original dataset.<br>
<ul>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
</ul>
 Please put the generated dataset under <b>./dataset</b> folder to be:
<pre>
./dataset
└─Synthetic-NEU-Seg
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>You may not redistribute this dataset with colorized mask, and the commerical use of the dataset is prohibited.</b><br><br>
As shown below, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<b>Synthetic-NEU-Seg Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/Synthetic-NEU-Seg_Statistics.png" width="512" height="auto"><br>
<br>
<br><br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Synthetic-NEU-Seg TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 256
image_height   = 256
image_channels = 3
input_normalize = True
normalization  = False

num_classes    = 4

base_filters   = 16
base_kernels  = (9,9)
num_layers    = 8

dropout_rate   = 0.03
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00006
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Synthetic-NEU-Seg 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Synthetic-NEU-Seg 1+3
;                  Patches:blue, Inclusion:green,  Scratches:red
rgb_map = {(0,0,0):0, (0,0,255):1, (0,255,0):2, (255,0,0):3,]
    
 
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (16,17,18)</b><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (33,34,35)</b><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 35 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/train_console_output_at_epoch35.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for Synthetic-NEU-Seg.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/evaluate_console_output_at_epoch35.png" width="880" height="auto">
<br><br>Image-Segmentation-Synthetic-NEU-Seg

<a href="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Synthetic-NEU-Seg/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.0459
dice_coef_multiclass,0.9775
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Synthetic-NEU-Seg.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  Steel-Rail-Surface-Defect Images of 224x224 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>
<b>rgb_map={Patches:blue, Inclusion:green,  Scratches:red}</b><br>

<br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/prediction_bottom_1.png" width="1023" height="auto"><br>
<img src="./projects/TensorFlowFlexUNet/Synthetic-NEU-Seg/asset/prediction_bottom_3.png" width="1025" height="auto"><br>

<hr>
<br>
<h3>
References
</h3>
<b>1. Latent Diffusion Models to Enhance the Performance of Visual Defect Segmentation Networks in Steel Surface Inspection</b><br>
Jon Leiñena ,  Fátima A Saiz , Iñigo Barandiaran <br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11436218/">
https://pmc.ncbi.nlm.nih.gov/articles/PMC11436218/</a>
<br>
<br>
<b>2. Steel surface defect segmentation with SME-DeeplabV3+</b><br>
Haiyan Zhang, Zining Zhao, Yilin Liu, Jiange Liu, Tingmei Ma, Kexiao Wu, Zhiwen Zhuang, Jiajun Wang<br>
<a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0329628">
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0329628</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
