testset_root = './datasets/example'
test_size = (854, 480)
test_crop_size = (854, 480)

mean = [0.5, 0.5, 0.5]
std  = [1, 1, 1]

inter_frames = 3


model = 'QVI'
pwc_path = './utils/network-default.pytorch'


store_path = 'outputs/example/'
checkpoint = 'checkpoints/quadratic/model.ckpt'


