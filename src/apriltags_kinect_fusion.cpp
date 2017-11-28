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

#include <opencv2/highgui/highgui.hpp>

#include <apriltagscpp/TagDetector.h>
#include <apriltagscpp/TagDetection.h>
#include <apriltagscpp/TagFamily.h>

#include "apriltags_kinect_fusion/AprilTagDetections.h"


typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class ApriltagsKinectFusion
{
public:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, PointCloud> SyncPolicy;

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

    static constexpr int IMAGE_WIDTH = 1920;
    static constexpr int IMAGE_HEIGHT = 1080;

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


        // Desired pattern:
        //     x_ind    y_ind     1
        //     --------------------
        //         0        0     1
        //         0        1     1
        //         0        2     1
        //      ...
        //         0     1077     1
        //         0     1078     1
        //         0     1079     1
        //         1        0     1
        //         1        1     1
        //         1        2     1
        //      ...
        //      1919     1077     1
        //      1919     1078     1
        //      1919     1079     1
        test_matrix_.create(IMAGE_WIDTH * IMAGE_HEIGHT, 3, CV_32FC1);
        for (int x_ind = 0; x_ind < IMAGE_WIDTH; ++x_ind)
        {
            for (int y_ind = 0; y_ind < IMAGE_HEIGHT; ++y_ind)
            {
                test_matrix_.at<float>(x_ind * IMAGE_HEIGHT + y_ind, 0) = (float)x_ind;
                test_matrix_.at<float>(x_ind * IMAGE_HEIGHT + y_ind, 1) = (float)y_ind;
                test_matrix_.at<float>(x_ind * IMAGE_HEIGHT + y_ind, 2) = 1.0f;
            }
        }

//        std::cout << "Test matrix size: " << test_matrix_.size() << std::endl;
//        std::cout << "Test matrix:\n" << test_matrix_(cv::Rect(0, 0, 3, 20)) << std::endl;
//        exit(EXIT_FAILURE);
    }

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

        for (size_t i = 0; i < detections.size(); ++i)
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

            std::cerr << "Detection id: " << det.id << "\n"
                      << det.p[0] << "\n"
                      << det.p[1] << "\n"
                      << det.p[2] << "\n"
                      << det.p[3] << "\n\n";
        }
    }

    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info)
    {
        camera_info_ = (*camera_info);
        has_camera_info_.store(true);
    }

    std::pair<Eigen::Isometry3d, double> getTagTransformAndSize(const TagDetection& tag, const PointCloud::ConstPtr& cloud)
    {
        // Iterate through each line, finding the valid region of the image for each one
        for (size_t first_corner_ind = 1; first_corner_ind < 4; ++first_corner_ind)
        {
            const size_t second_corner_ind = (first_corner_ind + 1) % 4;
            const size_t opposite_corner_ind = (first_corner_ind + 2) % 4;

            // Find the nullspace of A to determine the equation for the line
            Eigen::MatrixXf A(2, 3);
            A << tag.p[first_corner_ind].x,  tag.p[first_corner_ind].y,  1.0f,
                 tag.p[second_corner_ind].x, tag.p[second_corner_ind].y, 1.0f;
            const auto lu_decomp = Eigen::FullPivLU<Eigen::MatrixXf>(A);
            const Eigen::Vector3f nullspace_eigen = lu_decomp.kernel();

            std::cout << "A:\n" << A << std::endl << std::endl;
            std::cout << "nullspace_eigen:" << nullspace_eigen.transpose() << std::endl << std::endl;

            cv::Mat abc(3, 1, CV_32FC1);
            abc.at<float>(0, 0) = nullspace_eigen(0);
            abc.at<float>(1, 0) = nullspace_eigen(1);
            abc.at<float>(2, 0) = nullspace_eigen(2);

//            std::cout << "Eigen: " << nullspace_eigen.transpose() << std::endl;
//            std::cout << "Eigen manual: " << nullspace_eigen(0) << " " << nullspace_eigen(1) << " " << nullspace_eigen(2) << std::endl;
//            std::cout << "CV:    " << abc.t() << std::endl;

            const cv::Mat raw_math_vals = test_matrix_ * abc;

            std::cout << "Raw math vals: " << raw_math_vals(cv::Rect(0, 0, 1, 5)).t() << std::endl;

            // Determine which side of the line we should count as "in"
            const float value =
                    abc.at<float>(0, 0) * tag.p[opposite_corner_ind].x +
                    abc.at<float>(1, 0) * tag.p[opposite_corner_ind].y +
                    abc.at<float>(2, 0);
            cv::Mat valid_region;
            if (value < 0.0f)
            {
                valid_region = raw_math_vals < 0.0f;
            }
            else
            {
                valid_region = raw_math_vals > 0.0f;
            }
            cv::Mat valid_region_mask = valid_region.reshape(0, IMAGE_WIDTH).t();

            std::cout << "First 20 elements of valid region:\n" << valid_region(cv::Rect(0, 0, 1, 20)) << std::endl;
            std::cout << "First 20 elements of valid region mask:\n" << valid_region_mask(cv::Rect(0, 0, 1, 20)) << std::endl;

            cv::imshow("Display window", valid_region.reshape(0, IMAGE_WIDTH).t());         // Show our image inside it.
            while (ros::ok())
            {
                cv::waitKey(10);                                     // Wait for a keystroke in the window
            }

//            std::cout << valid_region << std::endl << std::endl << std::endl << std::endl;

        }

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


    cv::Mat test_matrix_;
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "apriltags_kinect_fusion");

    ApriltagsKinectFusion tags;

    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display

    ROS_INFO("AprilTags Kinect Fusion ready for data");
    ros::spin();

    return EXIT_SUCCESS;
}
