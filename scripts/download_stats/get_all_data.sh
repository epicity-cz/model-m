
if [ -z "$1" ]; then
  DATA_DIR='../../data/fit_data/'
else
  DATA_DIR="$1"
fi

[ -d "$DATA_DIR" ] || mkdir "$DATA_DIR"

python download_from_mzcr.py "$DATA_DIR"
python aggregate_stats.py "$DATA_DIR/okresy_aggregated.csv" --in_dir "$DATA_DIR/okresy/"
python create_fit_me.py "$DATA_DIR/fit_me.csv" "$DATA_DIR/okresy_aggregated.csv"
