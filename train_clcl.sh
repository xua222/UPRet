CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
--master_port 6665 \
main_task_retrieval.py \
--do_train \