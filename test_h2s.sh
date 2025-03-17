CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port 6665 \
main_task_retrieval.py \
--do_eval \
--init_model '/comp_robot/caomeng/code/lhx/SLRT/CiCo/CLCL_ot/best_checkpoint/best_model.bin.114' \