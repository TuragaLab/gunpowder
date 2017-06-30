REPOSITORY = funkey
GUNPOWDER_VERSION = v0.2
BUILD_TAG_SUFFIX := $(shell date +%Y%m%d%H%M%S)

GUNPOWDER_TAG = gunpowder:$(GUNPOWDER_VERSION)
GUNPOWDER_BUILD_TAG = $(GUNPOWDER_TAG)-$(BUILD_TAG_SUFFIX)
CAFFE_TAG = caffe:latest
CAFFE_BUILD_TAG = caffe:$(BUILD_TAG_SUFFIX)

.PHONY: default push
default:
	-docker rmi -f $(REPOSITORY)/$(CAFFE_TAG)
	docker build --pull -t $(CAFFE_BUILD_TAG) ./caffe
	docker tag $(CAFFE_BUILD_TAG) $(REPOSITORY)/$(CAFFE_BUILD_TAG)
	docker tag $(CAFFE_BUILD_TAG) $(REPOSITORY)/$(CAFFE_TAG)
	-docker rmi -f $(REPOSITORY)/$(GUNPOWDER_TAG)
	-rm -rf caffe/gunpowder/gunpowder
	cp -r ../gunpowder caffe/gunpowder
	cp ../requirements.txt caffe/gunpowder
	docker build -t $(GUNPOWDER_BUILD_TAG) ./caffe/gunpowder
	docker tag $(GUNPOWDER_BUILD_TAG) $(REPOSITORY)/$(GUNPOWDER_BUILD_TAG)
	docker tag $(GUNPOWDER_BUILD_TAG) $(REPOSITORY)/$(GUNPOWDER_TAG)

push: default
	docker push $(REPOSITORY)/$(CAFFE_BUILD_TAG)
	docker push $(REPOSITORY)/$(CAFFE_TAG)
	docker push $(REPOSITORY)/$(GUNPOWDER_BUILD_TAG)
	docker push $(REPOSITORY)/$(GUNPOWDER_TAG)

test: default
	docker run --rm $(REPOSITORY)/$(GUNPOWDER_TAG)
