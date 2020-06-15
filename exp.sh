# README: Since our script generating the random images for training/validation on the fly using memory,
#         sometime deadlocks may happen in PyTorch dataloader when num_workers > 0.
#         The solution is to use "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1" as prefix to each of those scripts.

# Demosaicing Algorithm: Malvar2004
# 512+64=576 image size, while random cropping 512 within 16px boundary
    # Achieve 99% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type both  --crop random_crop_inside_boundary
    
    # Stuck at ~50% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type jpeg  --crop random_crop_inside_boundary
        # lr tuning, and none achieve a better accuracy
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type jpeg  --crop random_crop_inside_boundary  --learning_rate 0.1
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type jpeg  --crop random_crop_inside_boundary  --learning_rate 0.01
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type jpeg  --crop random_crop_inside_boundary  --learning_rate 0.0001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type jpeg  --crop random_crop_inside_boundary  --learning_rate 0.00001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type jpeg  --crop random_crop_inside_boundary  --learning_rate 0.000001

    # Stuck at ~50% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type demosaic  --crop random_crop_inside_boundary
        # lr tuning, and none achieve a better accuracy
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type demosaic  --crop random_crop_inside_boundary  --learning_rate 0.1
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type demosaic  --crop random_crop_inside_boundary  --learning_rate 0.01
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type demosaic  --crop random_crop_inside_boundary  --learning_rate 0.0001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type demosaic  --crop random_crop_inside_boundary  --learning_rate 0.00001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 576 --crop_size 512 --image_type demosaic  --crop random_crop_inside_boundary  --learning_rate 0.000001

# 99 No crop
    # Achieve 99% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 99 --image_type both  --crop none
    
    # Achieve 99% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 99 --image_type jpeg  --crop none
    
    # Stuck at ~50% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 99 --image_type demosaic --crop none
        # lr tuning, and none achieve a better accuracy
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 99 --image_type demosaic --crop none --learning_rate 0.1
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 99 --image_type demosaic --crop none --learning_rate 0.01
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 99 --image_type demosaic --crop none --learning_rate 0.0001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 99 --image_type demosaic --crop none --learning_rate 0.00001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 99 --image_type demosaic --crop none --learning_rate 0.000001

# 100 No crop
    # All below 3 achieve 99% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 100 --image_type both  --crop none
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 100 --image_type jpeg  --crop none
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 100 --image_type demosaic  --crop none

# 112 No crop
    # Achieve 99% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 112 --image_type both  --crop none
    
    # Stuck at ~50% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 112 --image_type jpeg  --crop none
        # lr tuning, and none achieve a better accuracy
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 112 --image_type jpeg  --crop none --learning_rate 0.1
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 112 --image_type jpeg  --crop none --learning_rate 0.01
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 112 --image_type jpeg  --crop none --learning_rate 0.0001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 112 --image_type jpeg  --crop none --learning_rate 0.00001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 112 --image_type jpeg  --crop none --learning_rate 0.000001

    # Achieve 99% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004 --image_size 112 --image_type demosaic --crop none


# Demosaicing Algorithm: bilinear
# 512+64=576 image size, while random cropping 512 within 16px boundary
    # Achieve 99% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type both  --crop random_crop_inside_boundary
    
    # Stuck at ~50% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type jpeg  --crop random_crop_inside_boundary
        # lr tuning, and none achieve a better accuracy
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type jpeg  --crop random_crop_inside_boundary  --learning_rate 0.1
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type jpeg  --crop random_crop_inside_boundary  --learning_rate 0.01
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type jpeg  --crop random_crop_inside_boundary  --learning_rate 0.0001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type jpeg  --crop random_crop_inside_boundary  --learning_rate 0.00001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type jpeg  --crop random_crop_inside_boundary  --learning_rate 0.000001

    # Stuck at ~50% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type demosaic  --crop random_crop_inside_boundary
        # lr tuning, and none achieve a better accuracy
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type demosaic  --crop random_crop_inside_boundary  --learning_rate 0.1
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type demosaic  --crop random_crop_inside_boundary  --learning_rate 0.01
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type demosaic  --crop random_crop_inside_boundary  --learning_rate 0.0001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type demosaic  --crop random_crop_inside_boundary  --learning_rate 0.00001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 576 --crop_size 512 --image_type demosaic  --crop random_crop_inside_boundary  --learning_rate 0.000001

# 99 No crop
    # Achieve 99% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 99 --image_type both  --crop none
    
    # Achieve 99% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 99 --image_type jpeg  --crop none
    
    # Stuck at ~50% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 99 --image_type demosaic --crop none
        # lr tuning, and none achieve a better accuracy
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 99 --image_type demosaic --crop none --learning_rate 0.1
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 99 --image_type demosaic --crop none --learning_rate 0.01
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 99 --image_type demosaic --crop none --learning_rate 0.0001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 99 --image_type demosaic --crop none --learning_rate 0.00001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 99 --image_type demosaic --crop none --learning_rate 0.000001

# 100 No crop
    # All below 3 achieve 99% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 100 --image_type both  --crop none
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 100 --image_type jpeg  --crop none
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 100 --image_type demosaic  --crop none

# 112 No crop
    # Achieve 99% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 112 --image_type both  --crop none
    
    # Stuck at ~50% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 112 --image_type jpeg  --crop none
        # lr tuning, and none achieve a better accuracy
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 112 --image_type jpeg  --crop none --learning_rate 0.1
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 112 --image_type jpeg  --crop none --learning_rate 0.01
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 112 --image_type jpeg  --crop none --learning_rate 0.0001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 112 --image_type jpeg  --crop none --learning_rate 0.00001
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 112 --image_type jpeg  --crop none --learning_rate 0.000001

    # Achieve 99% accuracy
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --train_size 100000 --val_size 5000 --image_pattern gaussian_rgb --demosaic_algo bilinear --image_size 112 --image_type demosaic --crop none
