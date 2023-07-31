for i in {5..5}
do
    python train_classifier.py --seed $i --lr 0.1 --num_classes 100 --id_dataset_name CIFAR100 --model_type WideResNet-40-2-0.3 --num_epochs_train 100 --train_batch_size 128 --model_save_path ./Saved_Models/ --log_directory ./training_logs/classifier --id_train_end_index 400 --id_val_start_index 400 --id_test_start_index 0 --id_dataset_end_label 100
    
    python3 train_classifier.py --seed $i --lr 0.1 --num_classes 10 --id_dataset_name CIFAR10 --model_type WideResNet-40-2-0.3 --num_epochs_train 100 --train_batch_size 128 --model_save_path ./Saved_Models/ --log_directory ./training_logs/classifier --id_dataset_end_label 10
done
