#PBS -N runCnn60k
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=128GB
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
module load python/2.7.8
#
# This is the command the runs the python script
python -u cnndigit.py 60000 35 1e-4 >& outcnnHL35lamb1e-4.log
