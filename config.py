# data parameters
pretrain_model_path = '/home/qjq/pretrain_model/imagenet-vgg-f.mat'
training_size = 10000
query_size = 2000
database_size = 18015
batch_size = 128
y_dim = 1386
y_dim_nus = 1000
num_train = 10000

# hyper-parameters
max_epoch = 500
gamma = 1
eta = 1
bit = 64  # final binary code length
# lr = 10 ** (-1.5)  # initial learning rate
lr = 0.01
valid = True
