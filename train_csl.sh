CUDA_VISIBLE_DEVICES=0,1,2,3, \
python -m torch.distributed.launch --nproc_per_node=4 \
--master_port 6665 \
main_task_retrieval.py \
--do_train \
--data_path data_csl \
--alpha 0.8 \
--datatype csl \
--features_path '/comp_robot/caomeng/code/SLRT/CiCo/CLCL/sign_feature/csl_domain_agnostic' \
--features_path_retrain '/comp_robot/caomeng/code/SLRT/CiCo/CLCL/sign_feature/csl_domain_aware' \