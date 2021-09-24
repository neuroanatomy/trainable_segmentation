def image_string(img):
  dpi = 96
  fig = plt.figure(frameon=False)
  fig.set_size_inches(img_width/dpi, img_height/dpi)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(img, aspect='equal')

  buf = io.BytesIO()
  fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
  buf.seek(0)
  return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('UTF-8')

# get image
print("img_height, img_width:", img_height, img_width)
img = np.reshape(img.to_py(),(img_height, img_width, 4))
print("Img shape:", img.shape)

# get mask
print("mask_height, mask_width:", mask_height, mask_width)
mask_img = np.reshape(mask_img.to_py(),(mask_height, mask_width, 4))
mask_img = mask_img[:,:,0]
print("Mask shape:", mask_img.shape)

labels = [16, 32, 48, 64, 80, 96]
mask = np.zeros(mask_img.shape, dtype=np.uint8)
for i,l in enumerate(labels):
    ind = mask_img==l
    mask[ind] = i+1
print("Mask shape:", mask.shape)

print("get features from the training image")
segmentation_features_dict = {
    "intensity": True,
    "edges": True,
    "texture": True,
}
sigma_min=0.01
sigma_max=20
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

print("train the random forest classifier")
def compute_segmentations(mask=None, features=None):
    t1 = time()
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, max_samples=0.05)
    seg, clf = fit_segmenter(mask, features, clf)
    t2 = time()
    print("time:", t2 - t1)
    return (seg, clf)

segimg, clf = compute_segmentations(mask=mask,features=features)

print("done")
img_str = image_string(segimg)
