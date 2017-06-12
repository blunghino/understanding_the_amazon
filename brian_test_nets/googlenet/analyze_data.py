
# from training_utils import validate_epoch
from optimize_cutoffs import *

root_name = 'googlenet_state_dict_'
dtype = torch.cuda.FloatTensor

# train_loader, val_loader = generate_train_val_dataloader(
#         dataset,
#         batch_size=32,
#         num_workers=0,
#         split = 1
#     )

for i in range(1):
    save_model_path = root_name + str(i) + '.pkl'
    # model = GoogLeNet()
    # model.type(dtype)
    # state_dict = torch.load(save_model_path, map_location=lambda storage, loc: storage)
    #    model.load_state_dict(state_dict)
    model_function = get_googlenet_model
    print(get_optimal_cutoffs(save_model_path, model_function,
                              precision = 0.001,
                              csv_path = 'data/train_v2.csv',
                              img_path = 'data/train-jpg', img_ext = '.jpg',
                              dtype = dtype, batch_size = 32,
                              num_workers = 0, verbose = False))
