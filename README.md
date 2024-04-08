# 4D_CT_reproducible_capsule

1) Install python 3 (for example from the windows app store)
2) Install necessary packages (for example from command line using pip):
    python3 -m pip install tensorflow[and-cuda]
    python3 -m pip install h5py
    python3 -m pip install numpy
3) Unzip input image (N24_w_01_reconstructed_binned_5.zip), remove the resulting file (N24_w_01_reconstructed_binned_5.h5) from its folder, and place it in the same directory as the python script.
4) Run python code (file paths have to be specified differently based on operating system)
       Windows command line:
            python3 inference_overlap.py -i N24_w_01_reconstructed_binned_5.h5 -m \trained_model_4_layers_N1_to_N32\saved_model       
       Linux terminal:
            python3 inference_overlap.py -i N24_w_01_reconstructed_binned_5.h5 -m /trained_model_4_layers_N1_to_N32/saved_model
5) Inspect the resulting segmented file, and compare it against the provided segmented file (needs unzipping). The provided file and yours should be identical.
