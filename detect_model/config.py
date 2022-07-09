#data
dataroot='./data'  #./data/train/img      ./data/train/gt
test_img_path='./dataset/test/img2'
result = './result'

lr = 0.0001
gpu_ids = [0]
gpu = 1
init_type = 'xavier'

resume = True
checkpoint = '' # should be file
train_batch_size_per_gpu  = 16
num_workers = 8

print_freq = 1
eval_iteration = 1
save_iteration = 1
max_epochs = 65







