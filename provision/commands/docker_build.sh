docker image build -t s1_to_ndvi-development:latest  \
	--target builder  \
	--build-arg USER_ID=1000  \
	--build-arg GROUP_ID=1000  \
	--build-arg USERNAME=zhamilya  \
	--build-arg PROJECT_DIR=/home/zhamilya/PycharmProjects/s1_to_ndvi /home/zhamilya/PycharmProjects/s1_to_ndvi