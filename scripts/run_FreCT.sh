export CUDA_VISIBLE_DEVICES=1

python main_FreCT.py --anormly_ratio 1.8 --num_epochs 3    --batch_size 256  --mode train --dataset PSM  --data_path PSM --input_c 25    --output_c 25  --loss_fuc MSE  --win_size 60  --patch_size 135 --groups 6
python main_FreCT.py --anormly_ratio 1.8 --num_epochs 10       --batch_size 256     --mode test    --dataset PSM   --data_path PSM  --input_c 25    --output_c 25  --loss_fuc MSE  --win_size 60  --patch_size 135 --groups 6

python main_FreCT.py --anormly_ratio 1 --num_epochs 4  --batch_size 256  --mode train --dataset SMD  --data_path SMD   --input_c 38   --output_c 38  --loss_fuc MSE  --win_size 105  --patch_size 57 --groups 7
python main_FreCT.py --anormly_ratio 1 --num_epochs 10   --batch_size 256  --mode test    --dataset SMD   --data_path SMD     --input_c 38      --output_c 38   --loss_fuc MSE   --win_size 105  --patch_size 57 --groups 7

python main_FreCT.py --anormly_ratio 1 --num_epochs 3   --batch_size 128  --mode train --dataset SWAT  --data_path SWAT  --input_c 51    --output_c 51  --loss_fuc MSE --patch_size 135 --win_size 105 --groups 7
python main_FreCT.py --anormly_ratio 1  --num_epochs 10   --batch_size 128     --mode test    --dataset SWAT   --data_path SWAT  --input_c 51    --output_c 51   --loss_fuc MSE --patch_size 135 --win_size 105 --groups 7

python main_FreCT.py --anormly_ratio 1 --num_epochs 3  --batch_size 64  --mode train --dataset MSL  --data_path MSL  --input_c 55 --output_c 55  --win_size 90  --patch_size 359 --groups 9
python main_FreCT.py --anormly_ratio 1  --num_epochs 10     --batch_size 64    --mode test    --dataset MSL   --data_path MSL --input_c 55    --output_c 55   --win_size 90  --patch_size 359 --groups 9


python main_FreCT.py --anormly_ratio 0.85 --num_epochs 3   --batch_size 256  --mode train --dataset SMAP  --data_path SMAP --input_c 25    --output_c 25  --loss_fuc MSE --patch_size 15 --win_size 105 --groups 7
python main_FreCT.py --anormly_ratio 0.85  --num_epochs 10   --batch_size 256     --mode test    --dataset SMAP   --data_path SMAP  --input_c 25    --output_c 25   --loss_fuc MSE --patch_size 15 --win_size 105 --groups 7
