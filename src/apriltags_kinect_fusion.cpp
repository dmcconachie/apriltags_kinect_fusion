#include <iostream>
#include <memory>
#include <atomic>

#include <ros/ros.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <arc_utilities/ros_helpers.hpp>
#include <arc_utilities/timing.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <visualization_msgs/MarkerArray.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include <apriltagscpp/TagDetector.h>
#include <apriltagscpp/TagDetection.h>
#include <apriltagscpp/TagFamily.h>

#include "apriltags_kinect_fusion/AprilTagDetections.h"

using namespace pcl;

class ApriltagsKinectFusion
{
public:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, PointCloud<PointXYZRGB>> SyncPolicy;

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
//        , publish_detections_image_(ROSHelpers::GetParamDebugLog<bool>(ph_, "publish_detections_image", true))
        , viewer_(nullptr)
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

//        cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display
//        cv::namedWindow("Test window", cv::WINDOW_AUTOSIZE);// Create a window for display

        marker_publisher_ = nh_.advertise<visualization_msgs::MarkerArray>(
                    ROSHelpers::GetParamDebugLog<std::string>(ph_, "marker_topic", DEFAULT_MARKER_TOPIC), 1);

        detections_publisher_ = nh_.advertise<apriltags_kinect_fusion::AprilTagDetections>(
                    ROSHelpers::GetParamDebugLog<std::string>(ph_, "detections_topic", DEFAULT_DETECTIONS_TOPIC), 1);


//        viewer_ = simpleVis();
    }

private:


    pcl::visualization::PCLVisualizer::Ptr simpleVis()
    {
        // --------------------------------------------
        // -----Open 3D viewer and add point cloud-----
        // --------------------------------------------
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(0, 0, 0);
//        viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
//        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
        viewer->addCoordinateSystem(0.1);
        viewer->initCameraParameters();

        // Focused on the original template
    //    viewer->setCameraPosition(
    //                2.19043, 0.35053, 1.08629, // position
    //                2.19043, 0.0734896, 0.90981, // focal
    //                0.0, 0.0, 1.0 // view up
    //                );

        // Focused on the origin
    //    viewer->setCameraPosition(
    //                40.0, 50,0, // x and y of position
    //                60.0, 10.0, 20.0, // focal
    //                30.0, // no idea
    //                0.0, 0.0, 1.0 // view up - no idea what this really is
    //                );

//        viewer->setCameraPosition(
//                    0.0, 0.0, 0.4, // position
//                    0.0, 0.0, 0.0, // focal
//                    0.0, 1.0, 0.0 // view up
//                    );

        return viewer;
    }





    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info)
    {
        camera_info_ = (*camera_info);
        has_camera_info_.store(true);
    }

    void syncCallback(const sensor_msgs::ImageConstPtr& ros_image, const PointCloud<PointXYZRGB>::ConstPtr& cloud)
    {
        auto function_wide_stopwatch = arc_utilities::Stopwatch();

        if (!has_camera_info_.load())
        {
            ROS_WARN("No Camera Info Received Yet");
            return;
        }

        // Get the image
        cv_bridge::CvImagePtr cv_image_gray_wrapper;
        try
        {
            cv_image_gray_wrapper = cv_bridge::toCvCopy(ros_image, "mono8");
        }
        catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        const cv::Mat cv_image_gray = cv_image_gray_wrapper->image;
//        cv::imshow("Test window", cv_image_gray);

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

        std::cerr << "post tag detection:           " << function_wide_stopwatch(arc_utilities::READ) << " seconds" << std::endl;

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

            auto stopwatch = arc_utilities::Stopwatch();

            const auto pose_and_size = getTagTransformAndSize(det, cloud);
            const auto pose = pose_and_size.first;
            const auto tag_size = pose_and_size.second;

            std::cerr << "post tag pose/size:       " << stopwatch(arc_utilities::READ) << " seconds" << std::endl;

            // Fill in the visualization marker
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
            marker.pose = EigenHelpersConversions::EigenAffine3fToGeometryPose(pose);
            marker.color = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>::MakeFromFloatColors(1.0f, 0.0f, 1.0f, 1.0f);

            marker_transforms.markers.push_back(marker);

            // Fill in AprilTag detection.
            apriltags_kinect_fusion::AprilTagDetection apriltag_det;
            apriltag_det.header = marker.header;
            apriltag_det.id = marker.id;
            apriltag_det.tag_size = tag_size;
            apriltag_det.pose = marker.pose;
            for(size_t pt_i = 0; pt_i < 4; ++pt_i)
            {
                geometry_msgs::Point32 img_pt;
                img_pt.x = det.p[pt_i].x;
                img_pt.y = det.p[pt_i].y;
                img_pt.z = 1.0f;
                apriltag_det.corners2d[pt_i] = img_pt;
            }
            apriltag_detections.detections.push_back(apriltag_det);

//            if (viewer_ != nullptr)
//            {
//                while (ros::ok())
//                {
//                    viewer_->spinOnce(100);
//                }
//            }
        }


        std::cerr << "sync callback total time:     " << function_wide_stopwatch(arc_utilities::READ) << " seconds" << std::endl << std::endl;


        marker_publisher_.publish(marker_transforms);
        detections_publisher_.publish(apriltag_detections);
    }



    /////////// Step 0 /////////////
    std::pair<Eigen::Affine3f, float> getTagTransformAndSize(const TagDetection& tag, const PointCloud<PointXYZRGB>::ConstPtr& cloud)
    {
        // Get a mask which has non-zero values only in the region that the tag occupies
        cv::Mat valid_region_mask = getRegionMask(tag);

        // Extract the points correspinding to the mask from the point cloud
        const PointCloud<PointXYZRGB>::ConstPtr tag_xyzrgb = extractPoints(cloud, valid_region_mask);

        // Publish the data as a point cloud
        if (detections_cloud_publishers_.find(tag.id) == detections_cloud_publishers_.end())
        {
            detections_cloud_publishers_[tag.id] = nh_.advertise<PointCloud<PointXYZRGB>>("detections_as_clouds_" + std::to_string(tag.id), 1);
        }
        detections_cloud_publishers_.at(tag.id).publish(tag_xyzrgb);

        // Convert to just XYZ data
        PointCloud<PointXYZ>::Ptr tag_xyz(new PointCloud<PointXYZ>());
        copyPointCloud(*tag_xyzrgb, *tag_xyz);

        if (viewer_ != nullptr)
        {
            viewer_->removeAllPointClouds();

            viewer_->addPointCloud(tag_xyz, "starting_tag");;
            viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.9, 0.0, 0.9, "starting_tag");
            viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "starting_tag");
            viewer_->spinOnce(100);
        }

        // Fit a plane to the points, used to then find a bounding rectangle in the plane
        const ModelCoefficients::ConstPtr plane_coefficients = fitPlane(tag_xyz);
        const auto bbox = getBoundingBox(tag_xyz, plane_coefficients);

        return {bbox.first, (bbox.second[0] + bbox.second[0]) / 2.0};
    }

    /////////// Step 1 /////////////
    cv::Mat getRegionMask(const TagDetection& tag)
    {
        cv::Mat valid_region_mask(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255));

        // Iterate through each line, finding the valid region of the image for each one
        for (size_t first_corner_ind = 0; first_corner_ind < 4; ++first_corner_ind)
        {
            const size_t second_corner_ind = (first_corner_ind + 1) % 4;
            const size_t opposite_corner_ind = (first_corner_ind + 2) % 4;

            // Find the nullspace of A to determine the equation for the line
            Eigen::MatrixXf A(2, 3);
            A << tag.p[first_corner_ind].x,  tag.p[first_corner_ind].y,  1.0f,
                 tag.p[second_corner_ind].x, tag.p[second_corner_ind].y, 1.0f;
            const auto lu_decomp = Eigen::FullPivLU<Eigen::MatrixXf>(A);
            const Eigen::Vector3f nullspace_eigen = lu_decomp.kernel();

//            std::cout << "A:\n" << A << std::endl << std::endl;
//            std::cout << "nullspace_eigen:" << nullspace_eigen.transpose() << std::endl << std::endl;

            cv::Mat abc(3, 1, CV_32FC1);
            abc.at<float>(0, 0) = nullspace_eigen(0);
            abc.at<float>(1, 0) = nullspace_eigen(1);
            abc.at<float>(2, 0) = nullspace_eigen(2);

//            std::cout << "Eigen: " << nullspace_eigen.transpose() << std::endl;
//            std::cout << "Eigen manual: " << nullspace_eigen(0) << " " << nullspace_eigen(1) << " " << nullspace_eigen(2) << std::endl;
//            std::cout << "CV:    " << abc.t() << std::endl;

            const cv::Mat raw_math_vals = test_matrix_ * abc;

//            std::cout << "Raw math vals: " << raw_math_vals(cv::Rect(0, 0, 1, 5)).t() << std::endl;

            // Determine which side of the line we should count as "in"
            const float value =
                    abc.at<float>(0, 0) * tag.p[opposite_corner_ind].x +
                    abc.at<float>(1, 0) * tag.p[opposite_corner_ind].y +
                    abc.at<float>(2, 0);
            cv::Mat valid_region;
            if (value < 0.0f)
            {
                valid_region = raw_math_vals <= 0.0f;
            }
            else
            {
                valid_region = raw_math_vals >= 0.0f;
            }

            const cv::Mat tmp = valid_region.reshape(0, IMAGE_WIDTH).t();
            valid_region_mask &= tmp;

//            std::cout << "First 20 elements of valid region:\n" << valid_region(cv::Rect(0, 0, 1, 20)) << std::endl;
//            std::cout << "First 20 elements of valid region mask:\n" << valid_region_mask(cv::Rect(0, 0, 1, 20)) << std::endl;

//            cv::imshow("Display window", valid_region_mask);         // Show our image inside it.
//            cv::waitKey(1000);                                       // Wait for a keystroke in the window

//            std::cout << valid_region << std::endl << std::endl << std::endl << std::endl;

        }

        return valid_region_mask;
    }

    /////////// Step 2 /////////////
    PointCloud<PointXYZRGB>::Ptr extractPoints(const PointCloud<PointXYZRGB>::ConstPtr& input_cloud, const cv::Mat& region_mask)
    {
        // build the indices that we will extract
        PointIndices::Ptr inliers (new PointIndices());
        for (int x_ind = 0; x_ind < IMAGE_WIDTH; ++x_ind)
        {
            for (int y_ind = 0; y_ind < IMAGE_HEIGHT; ++y_ind)
            {
                if (region_mask.at<uchar>(y_ind, x_ind) != 0)
                {
                    inliers->indices.push_back(y_ind * IMAGE_WIDTH + x_ind);
                }
            }
        }

        // Perform the extraction itself
        PointCloud<PointXYZRGB>::Ptr extracted_cloud(new PointCloud<PointXYZRGB>());
        ExtractIndices<PointXYZRGB> extract;
        extract.setInputCloud(input_cloud);
        extract.setIndices(inliers);
        extract.filter(*extracted_cloud);

        // Filter out any NaN values
        PointCloud<PointXYZRGB>::Ptr output_cloud(new PointCloud<PointXYZRGB>());
        std::vector<int> index; // just used to satisfy the interface, not actually consumed
        pcl::removeNaNFromPointCloud(*extracted_cloud, *output_cloud, index);

        return output_cloud;
    }

    /////////// Step 3 /////////////
    ModelCoefficients::Ptr fitPlane(const PointCloud<PointXYZ>::ConstPtr& cloud)
    {
        ModelCoefficients::Ptr coefficients(new ModelCoefficients);
        PointIndices::Ptr inliers(new PointIndices);
        // Create the segmentation object
        SACSegmentation<PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory
        seg.setModelType(SACMODEL_PLANE);
        seg.setMethodType(SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        return coefficients;
    }

    /////////// Step 4.0 /////////////
    std::pair<Eigen::Affine3f, Eigen::Vector3f> getBoundingBox(
            const PointCloud<PointXYZ>::ConstPtr& cloud,
            const ModelCoefficients::ConstPtr& plane_coefficients)
    {
        assert(cloud != nullptr);
        assert(plane_coefficients != nullptr);
        assert(plane_coefficients->values.size() == 4);

        const auto reduce_result = reduceCloudToXY(cloud, plane_coefficients);
        const PointCloud<PointXYZ>::ConstPtr cloud_reduced = reduce_result.first;
        Eigen::Matrix4f rotation_to_reduced_frame = Eigen::Matrix4f::Identity();
        rotation_to_reduced_frame.col(0) = reduce_result.second.first;;
        rotation_to_reduced_frame.col(1) = reduce_result.second.second;
        rotation_to_reduced_frame.col(2) = Eigen::Vector4f(plane_coefficients->values[0],
                                                           plane_coefficients->values[1],
                                                           plane_coefficients->values[2],
                                                           0.0f);

        if (viewer_ != nullptr)
        {
            viewer_->addPointCloud<pcl::PointXYZ>(cloud_reduced, "reduced_points");
            viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.9, 0.0, 0.0, "reduced_points");
            viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reduced_points");
            viewer_->spinOnce(100);
        }

        ConvexHull<PointXYZ> hull;
        PointCloud<PointXYZ>::Ptr cloud_reduced_hull(new PointCloud<PointXYZ>);
        hull.setInputCloud(cloud_reduced);
//        hull.setDimension(2);
        hull.reconstruct(*cloud_reduced_hull);

        if (viewer_ != nullptr)
        {
            viewer_->addPointCloud<pcl::PointXYZ>(cloud_reduced_hull, "projected_hull");
            viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.9, 0.0, 0.9, "projected_hull");
            viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "projected_hull");
            viewer_->spinOnce(100);
        }

        const Eigen::Affine3f transform_to_aligned_box = getMinimumBoundingRectangleTransform(cloud_reduced_hull);
        PointCloud<PointXYZ>::Ptr bounding_rectangle_aligned_cloud(new PointCloud<PointXYZ>);
        transformPointCloud(*cloud_reduced_hull, *bounding_rectangle_aligned_cloud, transform_to_aligned_box.inverse());

        // Set the rotational component of the transform, we will overwrite the translational component momentariliy
        const Eigen::Matrix4f transform_to_bb_aligned_frame = rotation_to_reduced_frame * transform_to_aligned_box.matrix();
        PointCloud<PointXYZ>::Ptr complete_cloud_aligned(new PointCloud<PointXYZ>);
        transformPointCloud(*cloud, *complete_cloud_aligned, transform_to_bb_aligned_frame.inverse());
        Eigen::Vector4f min_point, max_point;
        getMinMax3D(*complete_cloud_aligned, min_point, max_point);
        const Eigen::Translation3f final_offset(((max_point + min_point) / 2.0f).head<3>());
        Eigen::Affine3f final_transform;
        final_transform = transform_to_bb_aligned_frame;
        final_transform *= final_offset;
        const Eigen::Vector3f box_dimensions((max_point - min_point).head<3>());

        if (viewer_ != nullptr)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr final_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*cloud, *final_cloud, final_transform.inverse());

            viewer_->addPointCloud<pcl::PointXYZ>(final_cloud, "final_cloud");
            viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.9, 0.0, "final_cloud");
            viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "final_cloud");
            viewer_->spinOnce(100);
        }

        return std::make_pair(final_transform, box_dimensions);
    }

    /////////// Step 4.1 /////////////
    std::pair<PointCloud<PointXYZ>::Ptr, std::pair<Eigen::Vector4f, Eigen::Vector4f>> reduceCloudToXY(
            const PointCloud<PointXYZ>::ConstPtr& cloud,
            const ModelCoefficients::ConstPtr& plane_coefficients)
    {
        // Whichever unit vector (x, y, or z) that points least along the plane normal is the unit vector that we
        // project to the plane and becomes the new x-axis; the y-axis will be define using the right hand rule
        const Eigen::Vector4f plane_normal(
                    plane_coefficients->values[0],
                    plane_coefficients->values[1],
                    plane_coefficients->values[2],
                    0.0);
        const std::vector<float> abs_normal =
        {
            std::fabs(plane_coefficients->values[0]),
            std::fabs(plane_coefficients->values[1]),
            std::fabs(plane_coefficients->values[2])
        };
        const int min_idx = (int)(std::min_element(abs_normal.begin(), abs_normal.end()) - abs_normal.begin());
        Eigen::Vector4f base_vector;
        float displacement;
        switch (min_idx)
        {
            case 0:
                base_vector = Eigen::Vector4f(1.0f, 0.0f, 0.0f, 0.0f);
                displacement = plane_coefficients->values[0];
                break;

            case 1:
                base_vector = Eigen::Vector4f(0.0f, 1.0f, 0.0f, 0.0f);
                displacement = plane_coefficients->values[1];
                break;

            case 2:
                base_vector = Eigen::Vector4f(0.0f, 0.0f, 1.0f, 0.0f);
                displacement = plane_coefficients->values[2];
                break;

            default:
                assert(false && "The min index should be x, y, or z");
        }

        const Eigen::Vector4f& z_axis = plane_normal;
        const Eigen::Vector4f x_axis = (base_vector - displacement * plane_normal).normalized();
        const auto y_axis_3vector = z_axis.head<3>().cross(x_axis.head<3>());
        const Eigen::Vector4f y_axis(y_axis_3vector(0), y_axis_3vector(1), y_axis_3vector(2), 0.0f);

        // Build the new x-y only point cloud from the original
        PointCloud<PointXYZ>::Ptr reduced_cloud(new PointCloud<PointXYZ>(cloud->width, cloud->height));
        for (size_t point_idx = 0; point_idx < cloud->points.size(); ++point_idx)
        {
            const auto point = cloud->points[point_idx].getVector4fMap();
            reduced_cloud->points[point_idx].x = x_axis.dot(point);
            reduced_cloud->points[point_idx].y = y_axis.dot(point);
            reduced_cloud->points[point_idx].z = 0.0f;
        }

        return std::make_pair(reduced_cloud, std::make_pair<>(x_axis, y_axis));
    }

    /////////// Step 4.2 /////////////
    // It is assumed that the cloud is already in a XY-plane, i.e. all Z values are zero.
    // Uses the caliper algorithm to find the smallest bounding rectangle in the plane:
    // http://datagenetics.com/blog/march12014/index.html
    Eigen::Affine3f getMinimumBoundingRectangleTransform(
            const PointCloud<PointXYZ>::ConstPtr& cloud)
    {
        Eigen::Affine3f best_transform;
        float smallest_area = std::numeric_limits<float>::infinity();
        for (size_t base_point_idx = 0; base_point_idx + 1 < cloud->points.size(); ++base_point_idx)
        {
            // Transform the cloud to the basepoint, aligning it with the next 2 indices of the hull
            const auto& base_point = cloud->points[base_point_idx];
            const auto& alignment_point = cloud->points[base_point_idx + 1];
            const auto transform_result = transformCloudToAlignedXAxis(cloud, base_point, alignment_point);
            const PointCloud<PointXYZ>::Ptr transformed_cloud = transform_result.first;
            const Eigen::Affine3f transform = transform_result.second;

            // Determine the area of the resulting rectangle
            Eigen::Vector4f min_point, max_point;
            getMinMax3D(*transformed_cloud, min_point, max_point);
            const float area = (max_point(0) - min_point(0)) * (max_point(1) - min_point(1));
            if (area < smallest_area)
            {
                smallest_area = area;
                best_transform = transform;
            }
        }
        return best_transform;
    }

    /////////// Step 4.3 /////////////
    // It is assumed that the cloud is already in a XY-plane, i.e. all Z values are zero
    std::pair<PointCloud<PointXYZ>::Ptr, Eigen::Affine3f> transformCloudToAlignedXAxis(
            const PointCloud<PointXYZ>::ConstPtr& cloud,
            const PointXYZ& base_point,
            const PointXYZ& alignment_point)
    {
        PointCloud<PointXYZ>::Ptr transformed_cloud(new PointCloud<PointXYZ>());
        const Eigen::Vector4f aligned_x_axis = (alignment_point.getVector4fMap() - base_point.getVector4fMap()).normalized();
        const Eigen::Vector4f aligned_y_axis(-aligned_x_axis(1), aligned_x_axis(0), 0.0f, 0.0f);
        const Eigen::Vector4f aligned_z_axis(0.0f, 0.0f, 1.0f, 0.0f);
        const Eigen::Vector4f translation = base_point.getVector4fMap();

        Eigen::Affine3f transform;
        transform.matrix().col(0) = aligned_x_axis;
        transform.matrix().col(1) = aligned_y_axis;
        transform.matrix().col(2) = aligned_z_axis;
        transform.matrix().col(3) = translation;

        // Executing the (inverse) transformation
        transformPointCloud(*cloud, *transformed_cloud, transform.inverse());

        return {transformed_cloud, transform};
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ros::NodeHandle nh_;
    ros::NodeHandle ph_;

    message_filters::Subscriber<sensor_msgs::Image> image_filter_sub_;
    message_filters::Subscriber<PointCloud<PointXYZRGB>> point_cloud_filter_sub_;
    message_filters::Synchronizer<SyncPolicy> sync_;

    ros::Subscriber info_subscriber_;
    std::atomic<bool> has_camera_info_;
    sensor_msgs::CameraInfo camera_info_;

    const TagFamily tag_family_;
    std::shared_ptr<const TagDetector> tag_detector_;
    TagDetectorParams tag_params_;

//    const bool publish_detections_image_;
    std::map<size_t, ros::Publisher> detections_cloud_publishers_;
    ros::Publisher marker_publisher_;
    ros::Publisher detections_publisher_;

    cv::Mat test_matrix_;



    pcl::visualization::PCLVisualizer::Ptr viewer_;
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "apriltags_kinect_fusion");

    ApriltagsKinectFusion tags;
    ROS_INFO("AprilTags Kinect Fusion ready for data");
    ros::spin();

    return EXIT_SUCCESS;
}
