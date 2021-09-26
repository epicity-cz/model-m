if [ -z "$1" ]
then
    NJOBS=`nproc --all`
    NJOBS=`expr $NJOBS / 2`
else
    NJOBS=$1
fi
echo "Running with $NJOBS processors"


echo "GENERATE A GRAPH."
if test -e "../data/m-input/hodoninsko"
then
    echo "Hodoninsko found. Skipping graph generation."
else
echo "(This step may take more than one hour.)"
python generate.py ../config/hodoninsko.ini || exit
fi 
echo "GENERATE A GRAPH -- FINISHED."
echo
echo 

echo "RUN THE SCENARIOS"
for I in 0 1
do
    echo "Running scenario $I"
    CFG="../config/hodoninsko_exp2_$I.ini"
    python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs $NJOBS --n_repeat 1000 $CFG exp2_$I  
done 
echo "RUN THE SCENARIOS -- FINISHED."


echo "PLOT THE RESULTS"
FILES="../data/output/model/history_exp2_0.zip"
FILES="$FILES ../data/output/model/history_exp2_1.zip"
python plot_experiments.py $FILES --label_names "baseline,no contact restrictions"  --ymax 6500 --out_file exp2_all_infected.png 
python plot_experiments.py --column I_d $FILES --label_names "baseline,no contact restrictions" --ymax 6500  --out_file exp2_id.png 
python plot_experiments.py --column D $FILES --label_names "baseline,no contact restrictions" --ymax 180  --out_file exp2_d.png 
echo "PLOT THE RESULTS -- FINISHED."
