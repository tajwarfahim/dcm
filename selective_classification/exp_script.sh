python run_expt.py --task cifar10 --method msp --lr 0.001 --base_model_path ./checkpoints/cifar10_erm.pt --data_dir ./data

python run_expt.py --task cifar10 --method max_logit --lr 0.001 --base_model_path ./checkpoints/cifar10_erm.pt --data_dir ./data

python run_expt.py --task cifar10 --method bc --lr 0.001 --base_model_path ./checkpoints/cifar10_erm.pt --data_dir ./data

python run_expt.py --task cifar10 --method dg --lr 0.001 --reward 3.2 --base_model_path ./checkpoints/cifar10_dg_sat_erm.pt --data_dir ./data

python run_expt.py --task cifar10 --method sat --lr 0.001 --base_model_path ./checkpoints/cifar10_dg_sat_erm.pt --data_dir ./data

python run_expt.py --task cifar10 --method ft --lr 0.001 --base_model_path ./checkpoints/cifar10_erm.pt --data_dir ./data

python run_expt.py --task cifar10 --method dcm --lr 0.0001 --confidence_weight 0.5 --base_model_path ./checkpoints/cifar10_erm.pt --data_dir ./data