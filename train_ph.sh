CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 \
--master_port 6665 \
main_task_retrieval.py \
--do_train \
--data_path data_ph \
--alpha 0.9 \
--datatype ph \
--features_path '/comp_robot/caomeng/code/SLRT/CiCo/CLCL/sign_feature/ph_domain_agnostic' \
--features_path_retrain '/comp_robot/caomeng/code/SLRT/CiCo/CLCL/sign_feature/ph_domain_aware' \