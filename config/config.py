class Config(object):
    lfw_root = '/data/Datasets/lfw/lfw-align-128'
    lfw_test_list = '/data/Datasets/lfw/lfw_test_pair.txt'
    model = 'checkpoint/face_step3_best.pth'
    test_model_path='checkpoint/face_step3_best.pth'
    test_batch_size = 60
    input_shape = (1, 112, 112)
    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    net_depth=50
    drop_ratio=0.4
    net_mode='ir'
 