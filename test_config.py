# root to the testsets
Testdataroot = 'PATH_TO_TEST_SET'


# WildRF test set
vals = ["reddit", "twitter", "facebook"]
multiclass = [0, 0, 0]

# ForenSynth test set
# vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
# multiclass = [1, 1, 1, 0, 1, 0, 0, 0]

# # UFD test set
# vals = ["dalle", "glide_100_10", "glide_100_27", "glide_50_27", "ldm_100", "ldm_200", "ldm_200_cfg", "guided"]
# multiclass = [0, 0, 0, 0, 0, 0, 0, 0]



# model
model_path = 'weights/LaDeDa/WildRF_LaDeDa.pth'