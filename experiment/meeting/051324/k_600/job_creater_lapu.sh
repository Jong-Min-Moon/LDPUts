#!/bin/bash
#
#
project_name="LDPUts"
meeting_date="051324"
experiment_name="k_600"
db_name="051324_LDPUts"
table_name="k600"
extension_code="py"
extension_result="npy"
tool="python -u"

code_dir="/home1/jongminm/${project_name}/experiment/meeting/${meeting_date}/${experiment_name}"
db_dir="${code_dir}/${db_name}.db"

#화면에 텍스트 표시:
echo "code_dir = ${code_dir}"



k=600
eta=0.0015
privmech=lapu
statistic=elltwo
privlev=1
memory_multiplier=10
n_permutation=999
n_test=100
for ii in 10 30 50 80 100
do
    sample_size=$((ii*1000))

    filename=${experiment_name}_${privmech}${statistic}_n${sample_size}_priv${privlev}
    filename_code=${code_dir}/${filename}.${extension_code}
    filename_job=${code_dir}/${filename}.job
    filename_out=${code_dir}/${filename}.out
    #scalar parameters
    touch ${code_dir}/temp_code
    echo "alphabet_size = ${k}"             >> ${code_dir}/temp_code
    echo "bump_size     = ${eta}"           >> ${code_dir}/temp_code
    echo "privacy_level = ${privlev}"       >> ${code_dir}/temp_code
    echo "sample_size   = ${sample_size}"   >> ${code_dir}/temp_code
    echo "n_permutation = ${n_permutation}" >> ${code_dir}/temp_code
    echo "n_test        = ${n_test}"        >> ${code_dir}/temp_code

    #strings
    echo "table_name = '${table_name}'"  >> ${code_dir}/temp_code
    echo "code_dir   = '${code_dir}'"  >> ${code_dir}/temp_code
    echo "priv_mech  = '${privmech}'"  >> ${code_dir}/temp_code
    echo "statistic  = '${statistic}'" >> ${code_dir}/temp_code
    echo "db_dir = '${db_dir}'" >> ${code_dir}/temp_code
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
    sleep 6
    sbatch ${filename_job}
done




