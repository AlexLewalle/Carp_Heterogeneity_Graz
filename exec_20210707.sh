#!/bin/bash

# Modified from exec_20210521a.sh
# ALC_20210609.py allows Tref to be specified for the healthy tissue



OutputDir="out/20210707"

duration=200



# Vary healthy Tref, keeping necrotic Tref constant ###########################################################################

for nec in 0.1    
do
for Trefnec in 0 15 20  # $(seq 0 5 30) 
do
for Trefref in 15 # $(seq 0 5 30) 
do
	simoutput="${OutputDir}/fracnec${nec}_Trefref${Trefref}_Trefnec${Trefnec}"

	# bench --load-module ~/Carp/MyModels/TT2_Cai.so --plug-in LandHumanStress --numstim 1 --imp-par='CaiClamp=0,CaiSET=${pCa1}' --dt 0.01 --stim-curr 100 --bcl 1000

	# ./ALC_20210707.py --duration $duration --ID ${simoutput}  --EP TT2_Cai  --pCa 5.0 --Tref $Trefref --Trefnec $Trefnec --Stress LandHumanStress  --fracnecrotic $nec --vd 0 --np 15

	~/meshtool/meshtool/meshtool extract surface \
	-msh=$simoutput/block_i \
	-surf=$simoutput/basal_surf \
	-op=$simoutput/basal_dbc \
	-ofmt=vtk 
	~/meshtool/standalones/reaction_force \
	-msh=$simoutput/block_i \
	-surf=$simoutput/basal_surf \
	-stress=$simoutput/stressTensor.igb \
	-out=$simoutput/basal_force    
	
done
done
done


