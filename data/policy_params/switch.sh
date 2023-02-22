for X in 21 28 35 42 49 56 63 70 77 84
do
cp LLLX.ini LLL${X}.ini 
eval sed -i 's/X/$X/g' LLL${X}.ini 
done
