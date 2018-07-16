#PBS -N runiCnn7-16_dn18tr0
#PBS -l walltime=02:30:00
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
#module load python/2.7.latest
source activate local

#
# This is the command the runs the python script

#python -u icecnn.py $PBS_ARRAYID >& outIceCnn7-2_$PBS_ARRAYID.log

python -u icecnn.py 18 0  >& outiCnn7-16_dn18tr0.log
