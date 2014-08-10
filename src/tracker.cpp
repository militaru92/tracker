
#include <tracker.h>

Tracker::Tracker()
{
  viewer_ptr_.reset(new pcl::visualization::PCLVisualizer("3D Visualizer"));
  viewer_ptr_->setBackgroundColor(0, 0, 0);
  viewer_ptr_->initCameraParameters();

  viewer_ptr_->registerKeyboardCallback <Tracker> (&Tracker::keyboardEventOccurred, *this,(void*)&viewer_ptr_);

  scan_ = false;
  index_ = 0;

  cloud_ptr_.reset( new pcl::PointCloud<pcl::PointXYZ>);


  Eigen::Vector3f volume_size = Eigen::Vector3f::Constant (3.f);
  kinfu_.volume().setSize (volume_size);

  Eigen::Matrix3f R = Eigen::Matrix3f::Identity ();
  Eigen::Vector3f t = volume_size * 0.5f - Eigen::Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

  Eigen::Affine3f pose = Eigen::Translation3f (t) * Eigen::AngleAxisf (R);

  kinfu_.setInitalCameraPose (pose);
  kinfu_.setCameraMovementThreshold(0.001f);


  if( !video_grabber_.open(CV_CAP_OPENNI_ASUS) )
  {
    PCL_ERROR("The camera is disconnected\n");
    exit(-1);
  }

  if( !face_classifier_.load( "haarcascade_frontalface_alt.xml" ) )
  {
    PCL_ERROR("Did not find the XML file\n");
    exit(-1);
  }

}

Tracker::~Tracker() throw ()
{

}


void
Tracker::keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *viewer_void)
{

  char c = event.getKeyCode();

  if(c == 't')
  {
    scan_ = true;

  }

  if(c == 's')
  {
    viewer_ptr_->saveScreenshot("ScreenShot" + boost::lexical_cast<std::string>(index_) + ".png");
  }


}

void
Tracker::show()
{

  cv::Mat frame,frame_gray;
  std::vector<cv::Rect> faces;

  video_grabber_.grab();

  video_grabber_.retrieve( frame, CV_CAP_OPENNI_GRAY_IMAGE );

  cv::equalizeHist( frame, frame_gray);

  face_classifier_.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

  std::pair < int, int > center_coordinates;

  if(faces.size() > 0)
  {
    center_coordinates.first = faces[0].x + faces[0].width/2;
    center_coordinates.second = faces[0].y + faces[0].height/2;
  }


  pcl::gpu::DeviceArray<pcl::PointXYZ> extracted = kinfu_.volume().fetchCloud (cloud_buffer_device_);

  extracted.download (cloud_ptr_->points);

  cloud_ptr_->width = (int)cloud_ptr_->points.size ();
  cloud_ptr_->height = 1;

  //pcl::io::savePCDFile ("cloud_face.pcd", *cloud_depth_ptr_, false);
  //pcl::io::savePCDFile ("cloud_buffer.pcd", *cloud_ptr_, false);

  viewer_ptr_->removeAllPointClouds ();

  if(faces.size() > 0)
  {
    viewer_ptr_->removeAllShapes();
  }



  viewer_ptr_->addPointCloud <pcl::PointXYZ> (cloud_ptr_,"Buffer_point_cloud_" + boost::lexical_cast<std::string>(index_));
  viewer_ptr_->addPointCloud <pcl::PointXYZ> (cloud_depth_ptr_, "Ordered_pointcloud_" + boost::lexical_cast<std::string>(index_));

  if(faces.size() > 0)
  {
    viewer_ptr_->addSphere <pcl::PointXYZ> (cloud_depth_ptr_->at(center_coordinates.first,center_coordinates.second),0.1, "Sphere_" + boost::lexical_cast<std::string>(index_));
  }



  ++index_;
}


void
Tracker::source_cb1_device(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper)
{
  {


    boost::mutex::scoped_try_lock lock(data_ready_mutex_);
    if (!lock)
        return;


    depth_.cols = depth_wrapper->getWidth();
    depth_.rows = depth_wrapper->getHeight();
    depth_.step = depth_.cols * depth_.elemSize();


    source_depth_data_.resize(depth_.cols * depth_.rows);
    depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
    depth_.data = &source_depth_data_[0];



    cloud_depth_ptr_ = this->convertToXYZPointCloud(depth_wrapper);

  }

  data_ready_cond_.notify_one();
}

void
Tracker::execute(bool has_data)
{

  if (has_data)
  {
    depth_device_.upload (depth_.data, depth_.step, depth_.rows, depth_.cols);
    kinfu_ (depth_device_);

    if(scan_)
    {
      scan_ = false;
      show();
    }

  }


}

void
Tracker::startMainLoop()
{
  boost::function<void (const DepthImagePtr&)> func2_dev = boost::bind (&Tracker::source_cb1_device, this, _1);
  boost::signals2::connection c = this->registerCallback (func2_dev);

  {
    boost::unique_lock<boost::mutex> lock(data_ready_mutex_);

    this->start ();

    while (true)
    {
      bool has_data = data_ready_cond_.timed_wait (lock, boost::posix_time::millisec(100));

      this->execute (has_data);

      this->viewer_ptr_->spinOnce(100);

    }

    this->stop ();
  }

  c.disconnect();

}
