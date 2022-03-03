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

improved_ss=('Ge87H76' 'Ge99H100' 'gupta2' 'Maragal_7' 'msc10848' 'ramage02')
improved_nn=('alexnetconv1' 'alexnetconv2' 'alexnetconv3' 'alexnetconv4'
'alexnetfc1' 'alexnetfc2' 'resnet50fc' 'resnet50layer3_conv1' 'resnet50layer3_conv2'
'resnet50layer3_conv3' 'resnet50layer4_conv1' 'resnet50layer4_conv2' 'resnet50layer4_conv3')

reduce_overhead=('192bit' 'brainpc2' 'case9' 'mri2' 'net4-1' 'rajat17' 'shermanACb' 'south31'
'TSOPF_FS_b9_c6' 'Maragal_8' 'TSOPF_FS_b162_c4')
enlarge_advantage=('bas1lp' 'blockqp1' 'c-64' 'GaAsH6' 'gupta2' 'SiO' 'TSOPF_FS_b162_c1')

cur_date=$(date +'%m_%d_%H')
echo "----Execute use $1 on $2----"
echo "----Use config: $4----"
echo "----Write output to $3/${cur_date}----"
mkdir -p ${3}/${cur_date}/
if [[ "$2" == "ss" ]]; then
    for i in "${ss[@]}"; do
        echo "* start $i"
        nohup ./target/debug/omega-sim trafficmodel $1 $2 $i $4 > ${3}/${cur_date}/${1}_${i}_${cur_date}.log &
        sleep 2
    done
elif [[ "$2" == "nn" ]]; then
    for i in "${nn[@]}"; do
        echo "* start $i"
        nohup ./target/debug/omega-sim trafficmodel $1 $2 $i $4 > ${3}/${cur_date}/${1}_${i}_${cur_date}.log &
        sleep 2
    done
elif [[ "$2" == "improve" ]]; then
    for i in "${improved_ss[@]}"; do
        echo "* start $i"
        nohup ./target/debug/omega-sim trafficmodel $1 ss $i $4 > ${3}/${cur_date}/${1}_${i}_${cur_date}.log &
        # sleep 2
    done
    for i in "${improved_nn[@]}"; do
        echo "* start $i"
        nohup ./target/debug/omega-sim trafficmodel $1 nn $i $4 > ${3}/${cur_date}/${1}_${i}_${cur_date}.log &
        # sleep 2
    done
elif [[ "$2" == "reduce_overhead" ]]; then
    for i in "${reduce_overhead[@]}"; do
        echo "* start $i"
        nohup ./target/debug/omega-sim trafficmodel $1 ss $i $4 > ${3}/${cur_date}/${1}_${i}${cur_date}.log &
    done
elif [[ "$2" == "enlarge_advantage" ]]; then
    for i in "${enlarge_advantage[@]}"; do
        echo "* start $i"
        nohup ./target/debug/omega-sim trafficmodel $1 ss $i $4 > ${3}/${cur_date}/${1}_${i}${cur_date}.log &
    done
else
    echo "Invalid workload type $2."
fi
