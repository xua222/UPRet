CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port 6665 \
main_task_retrieval.py \
--do_eval \
--data_path data_csl \
--alpha 0.8 \
--datatype csl \
--features_path '/comp_robot/caomeng/code/SLRT/CiCo/CLCL/sign_feature/csl_domain_agnostic' \
--features_path_retrain '/comp_robot/caomeng/code/SLRT/CiCo/CLCL/sign_feature/csl_domain_aware' \
--init_model '/comp_robot/caomeng/code/lhx/SLRT/CiCo/CLCL_ot/ot_csl/pytorch_model.bin.110' \