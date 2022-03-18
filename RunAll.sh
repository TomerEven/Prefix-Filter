
mkdir -p ../scripts/Inputs
t="measure_perf";
s="measure_built";
time cmake --build ./ --target $t -j 50
if ! [ $? -eq 0 ]; then
    exit 1;
fi
time cmake --build ./ --target $s -j 50
if ! [ $? -eq 0 ]; then
    exit 2;
fi

if ! [ -z "$(ls -A ../scripts/Inputs)" ]; then
   echo "Inputs is not empty!"
   exit 3;
fi

time taskset -c 2 ./$t
if ! [ $? -eq 0 ]; then
    echo "The benchmark was not completed!"
    exit;
fi
#sleep 120
time taskset -c 2 ./$s
if ! [ $? -eq 0 ]; then
    echo "The Built-bench was not completed!"
    exit;
fi
files_path="../scripts/Inputs/"
../scripts/arg-plotter.py $files_path 
../scripts/build-csv-parser.py 
time cmake --build ./ --target measure_fpp -j 50 && time taskset -c 2 ./measure_fpp