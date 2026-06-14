# RobuSTereo evaluation script
CUDA_VISIBLE_DEVICES=0 python tools/eval.py --cfg_file cfgs/dinobase/Dinobase-eval.yaml --eval_data_cfg_file cfgs/driving_rainy_eval.yaml --pretrained_model pretrain/RobuSTereo.pth

CUDA_VISIBLE_DEVICES=0 python tools/eval.py --cfg_file cfgs/dinobase/Dinobase-eval.yaml --eval_data_cfg_file cfgs/driving_foggy_eval.yaml --pretrained_model pretrain/RobuSTereo.pth

CUDA_VISIBLE_DEVICES=0 python tools/eval.py --cfg_file cfgs/dinobase/Dinobase-eval.yaml --eval_data_cfg_file cfgs/driving_sunny_eval.yaml --pretrained_model pretrain/RobuSTereo.pth

CUDA_VISIBLE_DEVICES=0 python tools/eval.py --cfg_file cfgs/dinobase/Dinobase-eval.yaml --eval_data_cfg_file cfgs/driving_cloudy_eval.yaml --pretrained_model pretrain/RobuSTereo.pth

CUDA_VISIBLE_DEVICES=0 python tools/eval.py --cfg_file cfgs/dinobase/Dinobase-eval.yaml --eval_data_cfg_file cfgs/STF_snow_eval.yaml --pretrained_model pretrain/RobuSTereo.pth

CUDA_VISIBLE_DEVICES=0 python tools/eval.py --cfg_file cfgs/dinobase/Dinobase-eval.yaml --eval_data_cfg_file cfgs/STF_rainy_eval.yaml --pretrained_model pretrain/RobuSTereo.pth

CUDA_VISIBLE_DEVICES=0 python tools/eval.py --cfg_file cfgs/dinobase/Dinobase-eval.yaml --eval_data_cfg_file cfgs/STF_dense_fog_eval.yaml --pretrained_model pretrain/RobuSTereo.pth

CUDA_VISIBLE_DEVICES=0 python tools/eval.py --cfg_file cfgs/dinobase/Dinobase-eval.yaml --eval_data_cfg_file cfgs/STF_light_fog_eval.yaml --pretrained_model pretrain/RobuSTereo.pth