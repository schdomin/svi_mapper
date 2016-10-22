#ifndef CTRACKERSV_H
#define CTRACKERSV_H

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"

#include "vision/CStereoCameraIMU.h"
#include "CTriangulator.h"
#include "CFundamentalMatcher.h"
#include "types/CKeyFrame.h"
#include "optimization/Cg2oOptimizer.h"



class CTrackerSV
{

//ds ctor/dtor
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CTrackerSV( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                    const EPlaybackMode& p_eMode,
                    const uint32_t& p_uWaitKeyTimeoutMS = 1 );
    ~CTrackerSV( );

//ds members
private:

    //ds vision setup
    uint32_t m_uWaitKeyTimeoutMS;
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;
    const std::shared_ptr< CStereoCamera > m_pCameraSTEREO;

    //ds SLAM structures
    std::shared_ptr< std::vector< CLandmark* > > m_vecLandmarks;
    std::shared_ptr< std::vector< CKeyFrame* > > m_vecKeyFrames;

    //ds reference information
    UIDFrame m_uFrameCount = 0;
    Eigen::Isometry3d m_matTransformationWORLDtoLEFTLAST;
    Eigen::Isometry3d m_matTransformationLEFTLASTtoLEFTNOW;
    double m_dTimestampLASTSeconds                      = 0.0;
    CPoint3DWORLD m_vecPositionKeyFrameLAST;
    Eigen::Vector3d m_vecCameraOrientationAccumulated   = {0.0, 0.0, 0.0};
    const double m_dTranslationDeltaForKeyFrameMetersL2 = 25.0; //5.0; //0.25;
    const double m_dAngleDeltaForKeyFrameRadiansL2      = 0.025;
    const UIDFrame m_uFrameDifferenceForKeyFrame        = 1e6; //100;
    UIDFrame m_uFrameKeyFrameLAST                       = 0;
    CPoint3DWORLD m_vecPositionCurrent;
    CPoint3DWORLD m_vecPositionLAST;
    double m_dTranslationDeltaSquaredNormCurrent = 0.0;

    //ds feature related
    //const uint32_t m_uKeyPointSize;
    const std::shared_ptr< cv::FeatureDetector > m_pDetector;
    const std::shared_ptr< cv::DescriptorExtractor > m_pExtractor;

    const uint8_t m_uVisibleLandmarksMinimum;
    const UIDFrame m_uMaximumNumberOfFramesWithoutDetection = 2; //1e6; //20;
    UIDFrame m_uNumberOfFramesWithoutDetection              = 0;

    std::shared_ptr< CTriangulator > m_pTriangulator;
    CFundamentalMatcher m_cMatcher;
    Cg2oOptimizer m_cOptimizer;

    //ds tracking (we use the ID counter instead of accessing the vector size every time for speed)
    std::vector< CLandmark* >::size_type m_uAvailableLandmarkID = 0;
    int32_t m_uNumberofVisibleLandmarksLAST                     = 0;
    const double m_dMaximumMotionScalingForOptimization = 1.5;
    double m_dMotionScalingLAST                         = 1.0;
    uint32_t m_uCountInstability                        = 0;
    std::vector< Eigen::Vector3d > m_vecRotations;

    //ds g2o optimization
    const std::vector< CLandmark* >::size_type m_uMinimumLandmarksForKeyFrame    = 50;
    const uint8_t m_uLandmarkOptimizationEveryNFrames                            = 10;
    std::vector< CKeyFrame* >::size_type m_uIDOptimizedKeyFrameLAST              = 0;
    const std::vector< CKeyFrame* >::size_type m_uIDDeltaKeyFrameForOptimization = 20; //10

    //ds loop closing
    const int64_t m_uMinimumLoopClosingKeyFrameDistance                              = 20; //20
    const double m_dMinimumRelativeMatchesLoopClosure                                = 0.5;
    const std::vector< CKeyFrame* >::size_type m_uLoopClosingKeyFrameWaitingQueue    = 1;
    std::vector< CKeyFrame* >::size_type m_uLoopClosingKeyFramesInQueue              = 0;
    std::vector< CKeyFrame* >::size_type m_uIDLoopClosureOptimizedLAST               = 0;
    const double m_dLoopClosingRadiusSquaredMetersL2                                 = 25.0;

    //ds robocentric world frame refreshing
    std::vector< Eigen::Vector3d > m_vecTranslationDeltas;
    const std::vector< Eigen::Vector3d >::size_type m_uIMULogbackSize = 200;
    Eigen::Vector3d m_vecGradientXYZ      = {0.0, 0.0, 0.0};
    Eigen::Vector3d m_vecTranslationToG2o = {0.0, 0.0, 0.0};

    //ds control
    EPlaybackMode m_eMode = ePlaybackStepwise;
    bool m_bIsShutdownRequested = false;

    //ds info display
    cv::Mat m_matDisplayLowerReference;
    uint32_t m_uFramesCurrentCycle = 0;
    double m_dPreviousFrameRate    = 0.0;
    double m_dPreviousFrameTime    = 0.0;
    double m_dDistanceTraveledMeters = 0.0;
    const std::string m_strVersionInfo;
    double m_dDurationTotalSecondsLoopClosing = 0.0;
    Eigen::MatrixXd m_matClosureMap;

#if defined USING_BOW
    const std::shared_ptr< BriefDatabase > m_pBoWDatabase;
#endif

//ds accessors
public:

    void process( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
                  const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageRIGHT );

    const UIDFrame getFrameCount( ) const { return m_uFrameCount; }
    const bool isShutdownRequested( ) const { return m_bIsShutdownRequested; }
    void finalize( );
    void sanitizeFiletree( ){ m_cOptimizer.clearFilesUNIX( ); }
    const double getDistanceTraveled( ) const { return m_dDistanceTraveledMeters; }
    const double getTotalDurationOptimizationSeconds( ) const { return 0; /*m_cGraphOptimizer.getTotalOptimizationDurationSeconds( );*/ }
    const double getDurationTotalSecondsStereoPosit( ) const { return m_cMatcher.getDurationTotalSecondsStereoPosit( ); }
    const double getDurationTotalSecondsRegionalTracking( ) const { return m_cMatcher.getDurationTotalSecondsRegionalTrackingFailed( ); }
    const double getDurationTotalSecondsEpipolarTracking( ) const { return m_cMatcher.getDurationTotalSecondsEpipolarTracking( ); }
    const double getDurationTotalSecondsLoopClosing( ) const { return m_dDurationTotalSecondsLoopClosing; }
    const double getDurationTotalSecondsOptimization( ) const { return m_cOptimizer.getDurationTotalSecondsOptimization( ); }

//ds helpers
private:

    void _trackLandmarks( const cv::Mat& p_matImageLEFT,
                          const cv::Mat& p_matImageRIGHT,
                          const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                          const Eigen::Isometry3d& p_matTransformationEstimateParallelWORLDtoLEFT,
                          const CLinearAccelerationIMU& p_vecLinearAcceleration,
                          const CAngularVelocityIMU& p_vecAngularVelocity,
                          const Eigen::Vector3d& p_vecRotationTotal,
                          const Eigen::Vector3d& p_vecTranslationTotal,
                          const double& p_dDeltaTimeSeconds );

    void _addNewLandmarks( const cv::Mat& p_matImageLEFT,
                           const cv::Mat& p_matImageRIGHT,
                           const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                           const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                           cv::Mat& p_matDisplaySTEREO );

    const std::vector< const CKeyFrame::CMatchICP* > _getLoopClosuresForKeyFrame( const CKeyFrame* p_pKeyFrameQUERY,
                                                                                  const double& p_dSearchRadiusMeters,
                                                                                  const double& p_dMinimumRelativeMatchesLoopClosure );

    const CMatchCloud _getMatchNN( std::list< CMatchCloud >& p_lMatches ) const;

    //ds reference frame update TODO implement
    //void _updateWORLDFrame( const Eigen::Vector3d& p_vecTranslationWORLD );

    //ds translation window to detect steady states
    void _initializeTranslationWindow( );

    //ds control
    void _shutDown( );
    void _updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange = 10 );
    void _drawInfoBox( cv::Mat& p_matDisplay, const double& p_dMotionScaling ) const;

#if defined USING_BOW

    //ds snippet: https://github.com/dorian3d/DLoopDetector/blob/master/include/DLoopDetector/TemplatedLoopDetector.h
    void _getMatches_neighratio( const std::vector< boost::dynamic_bitset<>> &A,
                                 const std::vector<unsigned int> &i_A,
                                 const std::vector<boost::dynamic_bitset<>> &B,
                                 const std::vector<unsigned int> &i_B,
                                 std::vector<unsigned int> &i_match_A,
                                 std::vector<unsigned int> &i_match_B ) const;

#endif

};

#endif //#define CTRACKERSV_H
