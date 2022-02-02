exe=/home/sc/research/ORB_SLAM3/bin/rio_renderer
N=8
for d in /media/sc/SSD1TB/dataset/3RScan/data/3RScan/*/sequence/;do #echo "$d"
 ((i=i%N)); ((i++==0)) && wait
 echo $d
 $exe --pth_in $d &
 #break

done
wait
