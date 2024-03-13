#!/bin/bash
#
#

project_name="LDPUts"
experiment_name="power_ptbunif_k1000"
extension_code="py"
extension_result="npy"
tool="python -u"

code_dir="/mnt/nas/users/user213/${project_name}/experiment/${experiment_name}"


#화면에 텍스트 표시:
echo "code_dir = ${code_dir}"


k=1000
eta=0.0009
statistic=chi
privmech=genrr
privlev=1
device=1



# code
touch ${code_dir}/temp_code
echo "alphabet_size = ${k}" >> ${code_dir}/temp_code
echo "bump_size = ${eta}" >> ${code_dir}/temp_code
echo "privacy_level = ${privlev}" >> ${code_dir}/temp_code
echo "device_num = ${device_num}" >> ${code_dir}/temp_code

#strings
echo "code_dir = '${code_dir}'" >> ${code_dir}/temp_code
echo "priv_mech = '${privmech}'" >> ${code_dir}/temp_code
echo "statistic = '${statistic}'" >> ${code_dir}/temp_code
cat ${code_dir}/skeleton_code.${extension_code} >> ${code_dir}/temp_code
mv ${code_dir}/temp_code ${code_dir}/${experiment_name}.${extension_code}



# job
output_file_name=${code_dir}/result/${experiment_name}_priv${privlev}_${privmech}${statistic}.out
touch ${code_dir}/temp_job
cat ${code_dir}/skeleton_job.job >> ${code_dir}/temp_job
echo "#SBATCH --job-name=${experiment_name}" >> ${code_dir}/temp_job
echo "#SBATCH --output=${output_file_name}" >> ${code_dir}/temp_job
cat ${code_dir}/conda_template >> ${code_dir}/temp_job
echo "${tool}  ${code_dir}/${experiment_name}.${extension_code} > ${output_file_name}" >> ${code_dir}/temp_job
mv ${code_dir}/temp_job ${code_dir}/${experiment_name}.job
sbatch ${code_dir}/${experiment_name}.job





