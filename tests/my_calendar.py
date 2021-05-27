import datetime
import sys 

i = 0
d = datetime.datetime(2020, 2, 25)


if len(sys.argv) > 1:
    num = int(sys.argv[1])
else:
    num = 100 

while i < num:
    i += 1 
    d += datetime.timedelta(days=1)    
    print(f"{i},{d}")

