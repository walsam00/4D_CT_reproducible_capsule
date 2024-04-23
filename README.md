# 4D_CT_reproducible_capsule
This is a reproducible capsule for the code in the following repository: https://github.com/walsam00/4D_CT_code

How to run the code:
1. Install python 3 (for example from the windows app store)
2.  Install necessary packages (for example from command line using pip):
    a) python3 -m pip install tensorflow
    b) python3 -m pip install h5py
    c) python3 -m pip install numpy
    Please note:
    The requirements.txt file contains exact versions of these packages that were used to create this reproducible capsule.
    However, simply installing the latest versions of the packages like the suggested pip commands (2a, 2b, 2c) do will most likely work just fine.
    Tensorflow runs a lot faster when it is able to run on a GPU. For tasks as small as the one in this reproducible capsule, the code should run in reasonable time on the CPU.
    If desired, the tensorflow package can be installed to run on the GPU as detailed on the tensorflow website (this replaces step 2a) https://www.tensorflow.org/install/pip
4. Unzip input image (N24_w_01_reconstructed_binned_5.zip) and place the unzipped image file (N24_w_01_reconstructed_binned_5.h5) in the same directory as the python script (UNet_inference.py) and the trained model save state (trained_model_4_layers_N1_to_N32).
5. Run python code (file paths have to be specified differently based on operating system) to apply the trained CNN to segment the provided reconstructed CT image file:
   a) Windows command line:
      python3 inference_overlap.py -i N24_w_01_reconstructed_binned_5.h5 -m \trained_model_4_layers_N1_to_N32\saved_model       
   b) Linux terminal:
      python3 inference_overlap.py -i N24_w_01_reconstructed_binned_5.h5 -m /trained_model_4_layers_N1_to_N32/saved_model
6. Inspect the resulting segmented recontsructed CT image file (N24_w_01_reconstructed_binned_5_segmented.h5), and compare it against the provided segmented file (needs unzipping). The provided file and the one the code produces should be identical. FIJI can be used to look at the HDF5 image files (https://imagej.net/software/fiji/downloads). Import *.h5 files into FIJI using the Ilastik plugin (https://www.ilastik.org/documentation/fiji_export/plugin)

Users can apply the trained CNN to their own data by changing the import (-i) argument of the python script. Data needs to be in the HDF5 file format (*.h5) and needs to conform to the data shape that the CNN import expects (x: 240 px, y: 240 px, z: any). The data needs to be stored in the HDF5 file in a group named 'data'. The axis order of the data needs to be z,y,x,c (c is a color channel of size 1 - images need to be greyscale). 
The U-Net that is applied to the data in this example segments images into 4 segmentation categories (mask, background, organic tablet material, inorganic tablet material). A version of the U-Net trained to segment images into three categories for images that lack inorganic material is also available (https://github.com/walsam00/4D_CT_code). The U-Net to be used for segmentation can be specified when running the python script using the 'model_path' (-m) argument.

DOI link to this repository:
https://zenodo.org/doi/10.5281/zenodo.11046923
