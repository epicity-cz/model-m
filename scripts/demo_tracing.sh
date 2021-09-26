python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 100 ../config/demo_0.ini demo0
python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 100 ../config/demo_1.ini demo1
python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 100 ../config/demo_2.ini demo2
python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 100 ../config/demo_3.ini demo3
python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 100 ../config/demo_4.ini demo4

FILES="../data/output/model/history_demo0.zip ../data/output/model/history_demo1.zip ../data/output/model/history_demo2.zip ../data/output/model/history_demo3.zip ../data/output/model/history_demo4.zip"
python plot_experiments.py $FILES --label_names 0,1,2,3,4 --out_file demo_tracing.png
