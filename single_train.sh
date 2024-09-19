scene='mill/city'
exp_name='Full'
voxel_size=0
n_offsets=10
update_init_factor=16
appearance_dim=0
ratio=1
gpu=-1

# example:
./train.sh -d ${scene} -l ${exp_name} --gpu ${gpu} --voxel_size ${voxel_size} --n_offsets ${n_offsets} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio}