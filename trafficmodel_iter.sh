ss=('2cubes_sphere' 'amazon0312' 'ca-CondMat' 'cage12' 'cit-Patents'
'cop20k_A' 'email-Enron' 'filter3D' 'm133-b3' 'mario002' 'offshore' 'p2p-Gnutella31'
'patents_main' 'poisson3Da' 'roadNet-CA' 'scircuit' 'web-Google' 'webbase-1M' 'wiki-Vote'
'degme' 'EternityII_Etilde' 'Ge87H76' 'Ge99H100' 'gupta2' 'm_t1' 'Maragal_7' 'msc10848'
'nemsemm1' 'NotreDame_actors' 'opt1' 'raefsky3' 'ramage02' 'relat8' 'ship_001' 'sme3Db'
'vsp_bcsstk30_500sep_10in_1Kout' 'x104')
nn=('alexnetconv0' 'alexnetconv1' 'alexnetconv2' 'alexnetconv3' 'alexnetconv4'
'alexnetfc0' 'alexnetfc1' 'alexnetfc2' 'resnet50conv0' 'resnet50layer1_conv1'
'resnet50layer1_conv2' 'resnet50layer1_conv3' 'resnet50layer2_conv1' 'resnet50layer2_conv2'
'resnet50layer2_conv3' 'resnet50layer3_conv1' 'resnet50layer3_conv2' 'resnet50layer3_conv3'
'resnet50layer4_conv1' 'resnet50layer4_conv2' 'resnet50layer4_conv3' 'resnet50fc')

# echo "Executing use $0"
# echo "$1"
# if [["$1" == "ss"]]; then
#     for i in "${ss[@]}"; do
#         nohup ./target/debug/omega-sim trafficmodel $0 $1 i >

cur_date=$(date +'%m_%d_%H')
echo "----Execute use $1 on $2----"
echo "Write output to $3/${cur_date}"
mkdir -p ${3}/${cur_date}/
if [[ "$2" == "ss" ]]; then
    for i in "${ss[@]:0:10}"; do
        echo "* start $i"
        nohup ./target/debug/omega-sim trafficmodel $1 $2 $i > ${3}/${cur_date}/${1}_${i}_${cur_date}.log &
        sleep 2
    done
elif [[ "$2" == "nn" ]]; then
    for i in "${nn[@]:0:1}"; do
        echo "* start $i"
        nohup ./target/debug/omega-sim trafficmodel $1 $2 $i > ${3}/${cur_date}/${1}_${i}_${cur_date}.log &
        sleep 2
    done
else
    echo "Invalid workload type $2."
fi
