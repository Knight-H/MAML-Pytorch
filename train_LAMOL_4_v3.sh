# (i)
python maml_pytorch_clean_v3-LAMOL_MAML.py --tasks yelp10k ag10k dbpedia10k amazon10k yahoo10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --update_lr 3e-3 --meta_lr 3e-5
# (ii)
python maml_pytorch_clean_v3-LAMOL_MAML.py --tasks dbpedia10k yahoo10k ag10k amazon10k yelp10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --update_lr 3e-3 --meta_lr 3e-5
# (iii)
python maml_pytorch_clean_v3-LAMOL_MAML.py --tasks yelp10k yahoo10k amazon10k dbpedia10k ag10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --update_lr 3e-3 --meta_lr 3e-5
# (iv)
python maml_pytorch_clean_v3-LAMOL_MAML.py --tasks ag10k yelp10k amazon10k yahoo10k dbpedia10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --update_lr 3e-3 --meta_lr 3e-5