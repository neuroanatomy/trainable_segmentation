print("get image")
img = np.reshape(img.to_py(),(img_height, img_width, 4))
print("Img shape:", img.shape)

print("get features")
t1 = time()
features = multiscale_basic_features(
    img,
    **segmentation_features_dict,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
)
t2 = time()
print("time", t2 - t1)
print("features", features.shape)

print("apply pre-trained classifier")
t1 = time()
segimg = predict_segmenter(features, clf)
t2 = time()
print("time:", t2 - t1)

print("done")
img_str = image_string(segimg)
