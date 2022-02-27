docker image build -t s1_to_ndvi-production:latest  \
	--target builder  \
	--build-arg USER_ID=1000  \
	--build-arg GROUP_ID=1000  \
	--build-arg USERNAME=ubuntu  \
	--build-arg PROJECT_DIR=/home/ubuntu/egistic/s1_to_ndvi /home/ubuntu/egistic/s1_to_ndvi