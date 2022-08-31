# this file looks at the simulation queue and runs all the required simulations

SIMULATION_QUEUE_PATH=$PWD"/data/nyc/p1_o/sample_queue.txt"
SIMULATION_FILES_PATH=$PWD"/data/simulation_files/idf/CO_San_Juan/"
SIMULATION_OUTPU_PATH=$PWD"/data/simulation_files/output/"
WEATHER_FILE=$PWD"/data/simulation_files/centralpark.epw"

mkdir -p $SIMULATION_OUTPU_PATH

while read line; do
    simulation_input=$SIMULATION_FILES_PATH$line
    filename="${line%.*}"
    simulation_output=$SIMULATION_OUTPU_PATH$filename
    mkdir -p $simulation_output
    # energyplus -w $WEATHER_FILE -d $simulation_output $simulation_input
    # energyplus --help
done < $SIMULATION_QUEUE_PATH