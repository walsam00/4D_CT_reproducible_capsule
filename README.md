# 4D_CT_reproducible_capsule

1) Install python 3
2) Install packages:
    python3 -m pip install tensorflow[and-cuda]
    python3 -m pip install h5py
    python3 -m pip install numpy
3) Unzip input image (N16_w_01_reconstructed_binned_6.zip)
4) Run python code:
   python3 inference_overlap.py -i N16_w_01_reconstructed_binned_6.h5 -m /trained_model_4_layers_N1_to_N32/saved_model
