#ifndef TRACKER_H
#define TRACKER_H

#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/openni_camera/openni_depth_image.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/opencv.hpp>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



class Tracker : public pcl::OpenNIGrabber
{

  public:

    typedef boost::shared_ptr<openni_wrapper::DepthImage> DepthImagePtr;

    Tracker();

    ~Tracker() throw ();


    void startMainLoop();

    void execute(bool has_data);




  private:

    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void);

    void source_cb1_device(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper);

    void show();


    cv::VideoCapture video_grabber_;
    cv::CascadeClassifier face_classifier_;

    pcl::gpu::KinfuTracker kinfu_;
    pcl::gpu::KinfuTracker::DepthMap depth_device_;

    pcl::visualization::PCLVisualizer::Ptr viewer_ptr_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr_;
    pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_depth_ptr_;

    pcl::PointXYZ face_center_;
    pcl::gpu::DeviceArray<pcl::PointXYZ> cloud_buffer_device_;


    pcl::gpu::PtrStepSz<const unsigned short> depth_;

    boost::mutex data_ready_mutex_;
    boost::condition_variable data_ready_cond_;

    std::vector<unsigned short> source_depth_data_;

    bool scan_;

    int index_;








};

#endif // TRACKER_H
