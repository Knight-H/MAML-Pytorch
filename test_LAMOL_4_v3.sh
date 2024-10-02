#!/bin/bash

# (i) yelp agnews dbpedia amazon yahoo - DONE
# (ii) dbpedia yahoo agnews amazon yelp - WRONG
# (iii) yelp yahoo amazon dbpedia agnews - DONE
# (iv) agnews yelp amazon yahoo dbpedia - DONE
# Test template 
# python maml_pytorch_test_v2-MAML.py --is_lamol --tasks ag yelp amazon yahoo dbpedia --model_dir_name 20211023T083631_agyelamayahdbp_LAMOL_MAML

# (i)
python maml_pytorch_test_v3-MAML.py --is_lamol --tasks yelp10k ag10k dbpedia10k amazon10k yahoo10k --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --gen_lm_sample_percentage 0.2 --update_lr 3e-3 --meta_lr 3e-5 --model_dir_name 20220202T115621_yelag1dbpamayah_LAMOL_MAML
# (ii)
python maml_pytorch_test_v3-MAML.py --is_lamol --tasks dbpedia10k yahoo10k ag10k amazon10k yelp10k --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --gen_lm_sample_percentage 0.2 --update_lr 3e-3 --meta_lr 3e-5 --model_dir_name 20220202T164844_dbpyahag1amayel_LAMOL_MAML
# (iii)
python maml_pytorch_test_v3-MAML.py --is_lamol --tasks yelp10k yahoo10k amazon10k dbpedia10k ag10k --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --gen_lm_sample_percentage 0.2 --update_lr 3e-3 --meta_lr 3e-5 --model_dir_name 20220202T220649_yelyahamadbpag1_LAMOL_MAML
# (iv)
python maml_pytorch_test_v3-MAML.py --is_lamol --tasks ag10k yelp10k amazon10k yahoo10k dbpedia10k --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --gen_lm_sample_percentage 0.2 --update_lr 3e-3 --meta_lr 3e-5 --model_dir_name 20220203T034532_ag1yelamayahdbp_LAMOL_MAML