sudo apt-get update -y
sudo apt-get install --no-install-recommends -y wget rsync git tmux htop
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
/bin/bash ~/miniconda.sh -b -p ~/conda && rm ~/miniconda.sh && \
~/conda/bin/conda clean -tipsy && \
sudo ln -s ~/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
echo ". ~/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
echo "conda activate base" >> ~/.bashrc
echo 'export PATH="~/conda/bin:$PATH"' >> ~/.bashrc
~/conda/bin/conda init && source ~/.bashrc