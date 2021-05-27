ID=$NAME

#python -OO run_all.py -R random_seeds.txt --n_repeat 1000 --n_jobs $NJ ${NAME}.ini ${ID} > ${ID}.log  2> ${ID}.err 

python -OO run_all.py -R random_seeds.txt --n_repeat 1000 --n_jobs $NJ ${NAME}.ini ${ID} > ${ID}.log  2> ${ID}.err 


eval B=`cat ${NAME}.ini | grep beta=`
#eval E=`cat ${NAME}.ini | grep init_E=`
eval A=`cat ${NAME}.ini | grep beta_A=`
##eval P=`cat ${NAME}.ini | grep p=`
eval TT=`cat ${NAME}.ini | grep theta_Is=`
#T=`echo "$B $A $P $E $TT"`
T=`echo "$B $A $TT"`
D=`date +"%b%d_%H:%M:%S"`
T="$T $D"


python plot_all.py ${ID} ${ID} "$T"
zip history_${ID}.zip history_${ID}_*.csv durations_${ID}_*.csv
rm history_${ID}_*.csv durations_${ID}_*.csv
rm ${ID}_*.log 
rm ${ID}_*.err 

#scp ${ID}.png petra@pc402c.adui.cs.cas.cz:work/TOP/
#scp ${ID}.png petra@pc402c.adui.cs.cas.cz:work/TOP/
#scp ${NAME}_mean_waiting.png petra@pc402c.adui.cs.cas.cz:work/TOP/
#scp ${NAME}_mean_waiting.png petra@pc402c.adui.cs.cas.cz:work/TOP/

