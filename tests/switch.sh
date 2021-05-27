# sed -i 's/zs_new.pickle/zs_raw.pickle/g' zs_baseline.ini
# sed -i 's/zs_new.pickle/zs_raw.pickle/g' zs_prvni_stupen.ini
# sed -i 's/zs_new.pickle/zs_raw.pickle/g' zs_druhy_stupen.ini
# sed -i 's/zs_new.pickle/zs_raw.pickle/g' zs_alternace.ini
#sed -i 's/zs_new.pickle/zs_raw.pickle/g' zs_krouzky.ini

#for BETA in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0"
#for BETA in "0.3" "0.31" "0.32" "0.33" "0.34" "0.35" "0.36" "0.37" "0.38" "0.39" "0.4" "0.41" "0.42" "0.43" "0.44" "0.45"
for  I in 0.1 0.25 0.5 1.0
do
for BETA in "0.32" 
do
    BETA_A=`python3 -c "print(${BETA}/2)"`

    echo "$BETA $BETA_A $I"
    #for SCENAR in baseline prvni_stupen alternace
#    for SCENAR in alternace_plus_pcr alternace_prvni_a_druhy
#    for SCENAR in alternace_volne_po
#   for SCENAR in rotace_oba_stupne rotace_plus_volne_pondeli baseline closed
    #    for SCENAR in test_v_nedeli_0.8 test_v_nedeli_0.4 baseline closed
    #    for SCENAR in rotace_polovicni_tridy_pcr
    #    for SCENAR in rotace_oba_stupne_bez_obeda #baseline_bez_obeda
    #    for SCENAR in testy_pondeli_a_streda_0.4  testy_pondeli_a_streda_0.2 testy_pondeli_a_streda_0.1
#    for SCENAR in rotace_testy_pondeli_a_streda_0.4  rotace_testy_pondeli_a_streda_0.2 rotace_testy_pondeli_a_streda_0.1 
    for SCENAR in baseline closed prvni_stupen rotace_oba_stupne rotace_druhy rotace_prvni
    do
	cp zs_${SCENAR}.ini epi_${I}_zs_${SCENAR}.ini 
	
	eval sed -i 's/daily_import=II/daily_import=${I}/g' epi_${I}_zs_${SCENAR}.ini
	
	eval sed  -i 's/beta_A=AA/beta_A=${BETA_A}/g' epi_${I}_zs_${SCENAR}.ini
	eval sed -i 's/beta_A_in_family=AA/beta_A_in_family=${BETA_A}/g' epi_${I}_zs_${SCENAR}.ini
 
	eval sed -i 's/beta=BB/beta=${BETA}/g' epi_${I}_zs_${SCENAR}.ini
	eval sed -i 's/beta_in_family=BB/beta_in_family=${BETA}/g' epi_${I}_zs_${SCENAR}.ini
    done
done
done
