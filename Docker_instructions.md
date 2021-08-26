# Run with Docker
If you have different operation system than Ubuntu 18, you may face some errors when try to run this repository. One of the solution may be to use Docker.

This instruction is tested on: 
- Ubuntu 20.04.2 LTS
- Docker 20.10.8, build 3967b7d

### 1. Install Docker
`apt install docker.io`

check installation
`docker run hello-world`

### 2. Prepare docker file
Create a new directory and a file without extension in it. Name the file **Dockerfile**.
Content of **Dockerfile**:
```   
FROM ubuntu:18.04
RUN apt update && apt upgrade -y
RUN apt -y install wget
RUN apt -y install ffmpeg libsm6 libxext6 build-essential	 # for gcc and libGL.so.1
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN /bin/bash ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh 
RUN /opt/conda/bin/conda clean -tipsy
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate base" >> ~/.bashrc
   ```

This file initializes docker image with Ubuntu 18 and Conda.

### 3. Build and run Docker image

First build your Docker image using Docker file
```
docker build -t <your_image_name> - < Dockerfile
```
Then run your image and mount directory with repository and data:
```
docker run -it --volume <mount_path_on_your_machine>:<full_mount_path_in_docker> <your_image_name>
#  -it for interactive session (your terminal will go into Docker container)
#  -d for detached session (Docker container will run in the background)
```
for example:
```
docker run -it --volume /home/user/directory_with_code_and_data:/my_directory <your_image_name>
```
when you go into container, the filesystem there won't have access to all your file system, but will have path 
**/my_directory** that leads to **/home/user/directory_with_code_and_data**

### 4. Set up environment

When you are in the docker container, create conda environment and install all the packages:
```
source /opt/conda/bin/activate
conda update conda
conda env create --name <your_name> --file <path-to-repo>/environment.yml 
conda activate <your_name>
```

Now you can run scripts from the repository!

## Save state of the container

It's not recommended, but if you need to save your current state of container (created environments, installed packages), you can do it.

When your container is still runnning, type in your terminal (not in docker container command line):
```
docker ps
```
You'll get something like this:
```
CONTAINER ID            IMAGE                COMMAND   CREATED       STATUS       PORTS     NAMES
<your_container_ID>     <your_image_name>    "bash"    5 hours ago   Up 5 hours             nervous_dhawan
```
Copy container ID, come up with good repository name and run (it will take some time):
```
docker commit <your_container_ID>  <repository_name>/<subname>:<tag>
```
For example:
```
docker commit dc0261a60633 rep_3dssg/testing:set_conda
```
Then, if you run `docker image list`, you'll see this new image in the list. 

To run this container:
```
docker run -it --volume <mount_path_on_your_machine>:<full_mount_path_in_docker> <repository_name>/<subname>:<tag>
```