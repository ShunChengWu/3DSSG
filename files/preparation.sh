wget http://campar.in.tum.de/files/3RScan/rescans.txt
wget http://campar.in.tum.de/files/3RScan/train_ref.txt
wget http://campar.in.tum.de/files/3RScan/val_ref.txt
wget http://campar.in.tum.de/files/3RScan/3RScan.json
wget http://campar.in.tum.de/files/3DSSG/3DSSG_subset/relationships.json
wget http://campar.in.tum.de/files/3DSSG/3DSSG_subset/relationships.txt
wget http://campar.in.tum.de/files/3DSSG/3DSSG_subset/classes.txt -O classes160.txt
cat ./train_ref.txt > references.txt
echo >> references.txt
cat ./val_ref.txt >> references.txt
rm train_ref.txt
rm val_ref.txt
