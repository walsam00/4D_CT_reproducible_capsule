# 4D_CT_reproducible_capsule
This is a reproducible capsule for the code in the following repository: https://github.com/walsam00/4D_CT_code

How to run the code:
1. Install python 3 (for example from the windows app store)
2.  Install necessary packages (for example from command line using pip):
    a) python3 -m pip install tensorflow[and-cuda]
    b) python3 -m pip install h5py
    c) python3 -m pip install numpy
    The requirements.txt file contains exact versions of these packages that were used to create this reproducible capsule.
    However, simply installing the latest versions of the packages like the suggested pip commands do will most likely also work fine.
4. Unzip input image (N24_w_01_reconstructed_binned_5.zip) and place the unzipped image file (N24_w_01_reconstructed_binned_5.h5) in the same directory as the python script (UNet_inference.py) and the trained model save state (trained_model_4_layers_N1_to_N32).
5. Run python code (file paths have to be specified differently based on operating system) to apply the trained CNN to segment the provided reconstructed CT image file:
   a) Windows command line:
      python3 inference_overlap.py -i N24_w_01_reconstructed_binned_5.h5 -m \trained_model_4_layers_N1_to_N32\saved_model       
   b) Linux terminal:
      python3 inference_overlap.py -i N24_w_01_reconstructed_binned_5.h5 -m /trained_model_4_layers_N1_to_N32/saved_model
6. Inspect the resulting segmented recontsructed CT image file (N24_w_01_reconstructed_binned_5_segmented.h5), and compare it against the provided segmented file (needs unzipping). The provided file and the one the code produces should be identical. FIJI can be used to look at the HDF5 image files (https://imagej.net/software/fiji/downloads). Import *.h5 files into FIJI using the Ilastik plugin (https://www.ilastik.org/documentation/fiji_export/plugin)
