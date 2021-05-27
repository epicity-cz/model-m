#for BETA in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0"
#for BETA in "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0"
#for BETA in "0.35" "0.36" "0.37" "0.38" "0.39" "0.4" "0.41" "0.42" "0.43" "0.44" "0.45"
for EPI in 1.0 0.5 0.25 0.1
do
for BETA in "0.32" 
do
#    for SCENAR in baseline prvni_stupen druhy_stupen alternace closed krouzky prvni_a_druha prvni_druha_devata alternace_prvni_sam alternace_prvni_a_druhy pulka pulka2
#    for SCENAR  in baseline alternace prvni_stupen #krouzky #baseline #prvni_a_druha_plus_alternace
    #for SCENAR in alternace_plus_pcr alternace_prvni_a_druhy
    for SCENAR in baseline closed alternace_plus_pcr alternace_prvni_a_druhy alternace_volne_po
   do
	NAME="epi_${EPI}_zs_${SCENAR}" ID="" NJ=100 ./multirun.sh
    done
done
done
