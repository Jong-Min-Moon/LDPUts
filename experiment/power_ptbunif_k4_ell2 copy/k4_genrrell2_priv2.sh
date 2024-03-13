#!/bin/bash
#
#

project_name="LDPUts"
experiment_name="power_ptbunif_k4_ell2"
extension_code="py"
extension_result="npy"
tool="python -u"

code_dir="/mnt/nas/users/user213/${project_name}/experiment/${experiment_name}"


#화면에 텍스트 표시:
echo "code_dir = ${code_dir}"


k=4
eta=0.04
statistic=ell2
privmech=genrr
privlev=2



# code
touch ${code_dir}/temp_code
echo "alphabet_size = ${k}" >> ${code_dir}/temp_code
echo "bump_size = ${eta}" >> ${code_dir}/temp_code
echo "privacy_level = ${privlev}" >> ${code_dir}/temp_code

#strings
echo "code_dir = '${code_dir}'" >> ${code_dir}/temp_code
echo "priv_mech = '${privmech}'" >> ${code_dir}/temp_code
echo "statistic = '${statistic}'" >> ${code_dir}/temp_code
cat ${code_dir}/skeleton_code.${extension_code} >> ${code_dir}/temp_code
mv ${code_dir}/temp_code ${code_dir}/${experiment_name}.${extension_code}



# job
touch ${code_dir}/temp_job
cat ${code_dir}/skeleton_job.job >> ${code_dir}/temp_job
echo "#SBATCH --job-name=${experiment_name}" >> ${code_dir}/temp_job
echo "#SBATCH --output=${code_dir}/result/${experiment_name}.out" >> ${code_dir}/temp_job
cat ${code_dir}/conda_template >> ${code_dir}/temp_job
echo "${tool}  ${code_dir}/${experiment_name}.${extension_code} > ${code_dir}/result/${experiment_name}.out" >> ${code_dir}/temp_job
mv ${code_dir}/temp_job ${code_dir}/${experiment_name}.job
sbatch ${code_dir}/${experiment_name}.job





