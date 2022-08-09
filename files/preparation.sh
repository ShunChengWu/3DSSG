wget https://www.campar.in.tum.de/public_datasets/3RScan/rescans.txt
wget https://www.campar.in.tum.de/public_datasets/3RScan/train_ref.txt
wget https://www.campar.in.tum.de/public_datasets/3RScan/val_ref.txt
wget https://www.campar.in.tum.de/public_datasets/3RScan/3RScan.json
wget https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset/relationships.json
wget https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset/relationships.txt
wget https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset/classes.txt -O classes160.txt
cat ./train_ref.txt > references.txt
echo >> references.txt
cat ./val_ref.txt >> references.txt
rm train_ref.txt
rm val_ref.txt
