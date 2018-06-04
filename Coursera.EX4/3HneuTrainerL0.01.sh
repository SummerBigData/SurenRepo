#PBS -N CalcThetasMNIST30k0.01
#PBS -l walltime=240:00
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
module load python/2.7.8
#
# This is the command the runs the python script
python -u neuTrainerMNIST.py 30000 784 36 0.01 0.12 -4 false >& output.log
