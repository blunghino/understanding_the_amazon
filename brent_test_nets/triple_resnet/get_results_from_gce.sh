#!/usr/bin/env bash

amazon_dir='brent_test_nets/triple_resnet'
root='triple_resnet18'

if [ ! -d "$root" ]; then
    mkdir $root
fi

files="triple_resnet18_rgb_data_fc.mat
triple_resnet18_inf_data_fc.mat
triple_resnet18_grad_data_fc.mat
triple_resnet18_rgb_data_tune.mat
triple_resnet18_inf_data_tune.mat
triple_resnet18_grad_data_tune.mat
triple_resnet18_rgb_state_dict.pkl
triple_resnet18_inf_state_dict.pkl
triple_resnet18_grad_state_dict.pkl"

for f in $files
do
    gcloud compute copy-files $GC_INSTANCE:$AMAZON_GCLOUD_PATH/$amazon_dir/$f $root/
done
