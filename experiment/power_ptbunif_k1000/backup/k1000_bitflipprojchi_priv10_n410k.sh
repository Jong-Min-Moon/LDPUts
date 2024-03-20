#!/bin/bash
#
#

project_name="LDPUts"
experiment_name="power_ptbunif_k1000"
extension_code="py"
extension_result="npy"
tool="python -u"

code_dir="/home1/jongminm/${project_name}/experiment/${experiment_name}"


#화면에 텍스트 표시:
echo "code_dir = ${code_dir}"



k=1000
eta=0.0009
privmech=bitflip
statistic=projchi
privlev=1
sample_size=$((410*1000))
memory_multiplier=3

filename=${experiment_name}_${privmech}${statistic}_n${sample_size}_priv${privlev}
filename_code=${code_dir}/${filename}.${extension_code}
filename_job=${code_dir}/${filename}.job
filename_out=${code_dir}/${filename}.out
# code
touch ${code_dir}/temp_code
echo "alphabet_size = ${k}" >> ${code_dir}/temp_code
echo "bump_size = ${eta}" >> ${code_dir}/temp_code
echo "privacy_level = ${privlev}" >> ${code_dir}/temp_code
echo "sample_size = ${sample_size}" >> ${code_dir}/temp_code

#strings
echo "code_dir = '${code_dir}'" >> ${code_dir}/temp_code
echo "priv_mech = '${privmech}'" >> ${code_dir}/temp_code
echo "statistic = '${statistic}'" >> ${code_dir}/temp_code
cat ${code_dir}/skeleton_code.${extension_code} >> ${code_dir}/temp_code
mv ${code_dir}/temp_code ${filename_code}



# job
touch ${code_dir}/temp_job
cat ${code_dir}/skeleton_job.job >> ${code_dir}/temp_job
echo "#SBATCH --mem=$((memory_multiplier*2*8*sample_size*k/1024/1024))mb" >> ${code_dir}/temp_job
echo "#SBATCH --job-name=${filename}" >> ${code_dir}/temp_job
echo "#SBATCH --output=${filename_out}" >> ${code_dir}/temp_job
echo "module purge" >> ${code_dir}/temp_job
echo "load conda" >> ${code_dir}/temp_job
cat ${code_dir}/conda_template >> ${code_dir}/temp_job
echo "${tool}  ${filename_code} > ${filename_out}" >> ${code_dir}/temp_job
mv ${code_dir}/temp_job ${filename_job}
sleep 3
sbatch ${filename_job}





