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
echo "We are going to run 1000 simulations for each scenario."
echo "This may require time, but it depends on the number of processors you have."
for I in `seq 0 4`
do
    echo "Running scenario $I"
    CFG="../config/hodoninsko_exp1_$I.ini"
    python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs $NJOBS --n_repeat 1000 $CFG exp1_$I  
done 
echo "RUN THE SCENARIOS -- FINISHED."


echo "PLOT THE RESULTS"
FILES="../data/output/model/history_exp1_0.zip"
FILES="$FILES ../data/output/model/history_exp1_1.zip"
FILES="$FILES ../data/output/model/history_exp1_2.zip"
FILES="$FILES ../data/output/model/history_exp1_3.zip"
FILES="$FILES ../data/output/model/history_exp1_4.zip"
python plot_experiments.py $FILES --label_names "no tracing,family,family+school&work,family+school&work+leisure,all" --ymax 6000 --out_file exp1_all_infected.png 
python plot_experiments.py --column I_d $FILES --label_names "no tracing,family,family+school&work,family+school&work+leisure,all" --ymax 6000 --out_file exp1_id.png 
echo "PLOT THE RESULTS -- FINISHED."
