# pose_mapping

### Python3 compatibility

To use Python3 with ROS Kinetic or newer:

```
cd ~/catkin_ws
rm -rf devel build
mkdir src
cd src 
git clone https://github.com/ros/geometry
git clone https://github.com/ros/geometry2
cd ..
python3 -m venv venv
source venv/bin/activate
pip install catkin_pkg pyyaml empy rospkg numpy
catkin build
source devel/setup.bash
```


