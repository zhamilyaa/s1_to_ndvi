docker image build -t sentinel-development:latest  \
	--target builder  \
	--build-arg USER_ID=1000  \
	--build-arg GROUP_ID=1000  \
	--build-arg USERNAME=zhamilya  \
	--build-arg PROJECT_DIR=/home/zhamilya/PycharmProjects/sentinel /home/zhamilya/PycharmProjects/sentinel