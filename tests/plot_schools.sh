BETA="0.4"

# krouzky 
python plot_run.py krouzky ${BETA}_zs_baseline ${BETA}_zs_krouzky
python plot_p.py krouzky ${BETA}_zs_baseline ${BETA}_zs_krouzky
python plot_p_cat.py krouzky zs_baseline zs_krouzky

# prvni a druhy stupen 
python plot_run.py prvni_a_druhy ${BETA}_zs_baseline ${BETA}_zs_prvni_stupen ${BETA}_zs_druhy_stupen
python plot_p.py prvni_a_druhy ${BETA}_zs_baseline ${BETA}_zs_prvni_stupen ${BETA}_zs_druhy_stupen
python plot_p_cat.py prvni_a_druhy zs_baseline zs_prvni_stupen zs_druhy_stupen 


# alternace
python plot_run.py alternace ${BETA}_zs_baseline ${BETA}_zs_prvni_stupen ${BETA}_zs_alternace
python plot_p.py alternace ${BETA}_zs_baseline ${BETA}_zs_prvni_stupen ${BETA}_zs_alternace
python plot_p_cat.py alternace zs_baseline zs_prvni_stupen zs_alternace

# prvni a druha trida
python plot_run.py prvni_a_druha ${BETA}_zs_baseline ${BETA}_zs_prvni_stupen ${BETA}_zs_prvni_a_druha 
python plot_p.py prvni_a_druha ${BETA}_zs_baseline ${BETA}_zs_prvni_stupen ${BETA}_zs_prvni_a_druha 
python plot_p_cat.py prvni_a_druha zs_baseline zs_prvni_stupen zs_prvni_a_druha

# prvni, druha a devata
python plot_run.py okraje ${BETA}_zs_baseline ${BETA}_zs_prvni_stupen ${BETA}_zs_prvni_druha_devata
python plot_p.py okraje ${BETA}_zs_baseline ${BETA}_zs_prvni_stupen ${BETA}_zs_prvni_druha_devata 
python plot_p_cat.py okraje zs_baseline zs_prvni_stupen zs_prvni_druha_devata

# prvni, druha a devata + 8C
python plot_run.py okraje2 ${BETA}_zs_baseline ${BETA}_zs_prvni_stupen ${BETA}_zs_prvni_druha_devata2
python plot_p.py okraje2 ${BETA}_zs_baseline ${BETA}_zs_prvni_stupen ${BETA}_zs_prvni_druha_devata2
python plot_p_cat.py okraje2 zs_baseline zs_prvni_stupen zs_prvni_druha_devata2


