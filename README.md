# pose_mapping

### Python3 compatibility

To use Python3 with ROS Kinetic or newer:

```
cd ~/catkin_ws
rm -rf devel build
cd src 
git clone https://github.com/ros/geometry
git clone https://github.com/ros/geometry2
cd ..
python3 -m venv venv
source venv/bin/activate
pip install catkin_pkg pyyaml rospkg numpy
catkin build
# or:
# catkin build --cmake-args -DPYTHON_EXECUTABLE:FILEPATH=/path/to/virtualenv/python
source devel/setup.bash
```
(source https://github.com/ros/geometry2/issues/259)


