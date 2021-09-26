python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 10 ../config/demo.ini demo
python plot_experiments.py ../data/output/model/history_demo.zip --label_names demo --out_file demo.png
