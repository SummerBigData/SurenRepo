#PBS -N runIceCnn
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=16GB
#PBS -j oe
# uncomment if using qsub
if [ -z "$PBS_O_WORKDIR" ] 
then
        echo "PBS_O_WORKDIR not defined"
else
        cd $PBS_O_WORKDIR
        echo $PBS_O_WORKDIR
fi
#
# Setup GPU code
module load python/2.7.latest
source activate local
#
# This is the command the runs the python script
python -u icecnn.py 0 0 >& outIceCnn7-2_3.log
