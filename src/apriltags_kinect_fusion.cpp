#include <iostream>
#include <memory>
#include <atomic>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <arc_utilities/ros_helpers.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <visualization_msgs/MarkerArray.h>

#include <apriltagscpp/TagDetector.h>
#include <apriltagscpp/TagDetection.h>
#include <apriltagscpp/TagFamily.h>

#include "apriltags_kinect_fusion/AprilTagDetections.h"


typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class ApriltagsKinectFusion
{
public:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, PointCloud> SyncPolicy;

    ApriltagsKinectFusion()
        : nh_("")
        , ph_("~")
        , image_filter_sub_      (nh_, ROSHelpers::GetParamDebugLog<std::string>(ph_, "image_topic",       DEFAULT_IMAGE_TOPIC),       1, ros::TransportHints().tcpNoDelay())
        , point_cloud_filter_sub_(nh_, ROSHelpers::GetParamDebugLog<std::string>(ph_, "point_cloud_topic", DEFAULT_POINT_CLOUD_TOPIC), 1, ros::TransportHints().tcpNoDelay())
        // ApproximateTime takes a queue size as its constructor argument, hence SyncPolicy(10)
        , sync_(SyncPolicy(10), image_filter_sub_, point_cloud_filter_sub_)
        , has_camera_info_(false)
        , tag_family_(ROSHelpers::GetParamDebugLog<std::string>(ph_, "tag_family", DEFAULT_TAG_FAMILY))
        , publish_detections_image_(ROSHelpers::GetParamDebugLog<bool>(ph_, "publish_detections_image", true))
    {
        info_subscriber_ = nh_.subscribe(
                    ROSHelpers::GetParamDebugLog<std::string>(ph_, "camera_info_topic", DEFAULT_CAMERA_INFO_TOPIC),
                    1, &ApriltagsKinectFusion::cameraInfoCallback, this);

        tag_params_.newQuadAlgorithm = true;
        tag_detector_ = std::make_shared<const TagDetector>(tag_family_, tag_params_);

        sync_.registerCallback(&ApriltagsKinectFusion::syncCallback, this);
    }

    static constexpr auto DEFAULT_IMAGE_TOPIC       = "kinect2/hd/image_color_rect";
    static constexpr auto DEFAULT_POINT_CLOUD_TOPIC = "kinect2/hd/points";
    static constexpr auto DEFAULT_CAMERA_INFO_TOPIC = "kinect2/hd/camera_info";

    static constexpr auto DEFAULT_MARKER_TOPIC = "marker_array";
    static constexpr auto DEFAULT_DETECTIONS_TOPIC = "detections";
    static constexpr auto DEFAULT_DETECTIONS_IMAGE_TOPIC = "detections_image";
    static constexpr auto DEFAULT_DISPLAY_TYPE = "CUBE";

    static constexpr auto DEFAULT_TAG_FAMILY = "Tag36h11";
    static constexpr double SMALL_TAG_SIZE = 0.0358968;
    static constexpr double MED_TAG_SIZE = 0.06096;
    static constexpr double PAGE_TAG_SIZE = 0.165;
    static constexpr double DEFAULT_TAG_SIZE = MED_TAG_SIZE;

private:

    void syncCallback(const sensor_msgs::ImageConstPtr& ros_image, const PointCloud::ConstPtr& cloud)
    {
        std::cerr << "Processing callback\n";

        if (!has_camera_info_.load())
        {
            ROS_WARN("No Camera Info Received Yet");
            return;
        }

        // Get the image
        cv_bridge::CvImagePtr cv_image;
        try
        {
            cv_image = cv_bridge::toCvCopy(ros_image, "mono8");
        }
        catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        const cv::Mat cv_image_gray = cv_image->image;

        cv::Point2d opticalCenter;
        if ((camera_info_.K[2] > 1.0) && (camera_info_.K[5] > 1.0))
        {
            // cx,cy from intrinsic matric look reasonable, so we'll use that
            opticalCenter = cv::Point2d(camera_info_.K[5], camera_info_.K[2]);
        }
        else
        {
            opticalCenter = cv::Point2d(0.5 * cv_image_gray.rows, 0.5 * cv_image_gray.cols);
        }

        // Detect AprilTag markers in the image
        TagDetectionArray detections;
        tag_detector_->process(cv_image_gray, opticalCenter, detections);

        // Build the visualization message
        visualization_msgs::MarkerArray marker_transforms;
        apriltags_kinect_fusion::AprilTagDetections apriltag_detections;
        apriltag_detections.header.frame_id = cloud->header.frame_id;
        apriltag_detections.header.stamp = ros_image->header.stamp;

        for(size_t i = 0; i < detections.size(); ++i)
        {
            // skip bad detections
            if (!detections[i].good)
            {
                continue;
            }

            const TagDetection &det = detections[i];

            const auto pose_and_size = getTagTransformAndSize(det, cloud);
            const auto pose = pose_and_size.first;
            const auto tag_size = pose_and_size.second;

            visualization_msgs::Marker marker;
            marker.header.frame_id = apriltag_detections.header.frame_id;
            marker.header.stamp = apriltag_detections.header.stamp;

            marker.ns = "tag" + std::to_string(det.id);
            marker.id = static_cast<int>(det.id);
            marker.action = visualization_msgs::Marker::ADD;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.scale.x = tag_size;
            marker.scale.y = tag_size;
            marker.scale.z = tag_size / 5.0;
            marker.pose = EigenHelpersConversions::EigenIsometry3dToGeometryPose(pose);
            marker.color = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>::MakeFromFloatColors(1.0f, 0.0f, 1.0f, 1.0f);

            marker_transforms.markers.push_back(marker);
        }
    }

    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info)
    {
        camera_info_ = (*camera_info);
        has_camera_info_.store(true);
    }

    std::pair<Eigen::Isometry3d, double> getTagTransformAndSize(const TagDetection& tag, const PointCloud::ConstPtr& cloud)
    {
        return {Eigen::Isometry3d::Identity(), 0.23};
    }

    ros::NodeHandle nh_;
    ros::NodeHandle ph_;

    message_filters::Subscriber<sensor_msgs::Image> image_filter_sub_;
    message_filters::Subscriber<PointCloud> point_cloud_filter_sub_;
    message_filters::Synchronizer<SyncPolicy> sync_;

    ros::Subscriber info_subscriber_;
    std::atomic<bool> has_camera_info_;
    sensor_msgs::CameraInfo camera_info_;

    const TagFamily tag_family_;
    std::shared_ptr<const TagDetector> tag_detector_;
    TagDetectorParams tag_params_;

    const bool publish_detections_image_;
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "apriltags_kinect_fusion");

    ApriltagsKinectFusion tags;
    ROS_INFO("AprilTags Kinect Fusion ready for data");
    ros::spin();

    return EXIT_SUCCESS;
}
