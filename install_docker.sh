tar -xvf file/trec_eval-9.0.7.tar.gz -C file
cd file/trec_eval-9.0.7
make
cd ../../


directory=`pwd`

echo $directory

chmod -R 777 $directory

docker import ${directory}/images/IR.tar zhangzhao-ir:v1
