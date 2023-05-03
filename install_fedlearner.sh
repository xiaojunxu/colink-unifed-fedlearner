git clone https://github.com/bytedance/fedlearner.git
cd fedlearner
git checkout 86a9f3e7a39a39767bb3913461de4b120f80cacd
pip install --upgrade pip
pip install -r requirements.txt
pip install flbenchmark
pip install scikit-learn
pip install protobuf==3.20.* pytest
cp ../fedlearner.patch ./fedlearner.patch
git apply --whitespace=fix fedlearner.patch
make protobuf && make op
cd ..
