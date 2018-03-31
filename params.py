"""Params for ADDA."""

# params for dataset and data loader
data_root = "/media/Data/dataset_xian/VisDA/"
dataset_mean = [0.485, 0.456, 0.406]
dataset_std = [0.229, 0.224, 0.225]
batch_size = 256
image_size = 224
num_workers = 2
gpu_ids = [0,1]

# params for source dataset
# src_dataset = "MNIST"
# src_encoder_restore = '/home/zxf/.torch/models/resnet34-333f7ec4.pth'
src_encoder_restore = 'snapshots/target/ADDA-source-encoder-10.pt'
src_classifier_restore = 'snapshots/target/ADDA-source-classifier-10.pt'
src_model_trained = True
# tgt_dataset = "USPS"
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

# params for setting up models
model_root = "snapshots/target"
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = "snapshots/ADDA-critic-final.pt"

# params for training network
num_epochs_pre = 50
log_step_pre = 20
eval_epoch_pre = 5
save_epoch_pre = 5

num_epochs = 2000
log_step = 100
save_step = 100
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-3
# beta1 = 0.5
# beta2 = 0.9
