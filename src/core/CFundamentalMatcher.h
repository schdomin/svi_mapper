#ifndef CFUNDAMENTALMATCHER_H
#define CFUNDAMENTALMATCHER_H

#include "CTriangulator.h"
#include "types/CLandmark.h"
#include "types/TypesCloud.h"
#include "optimization/CSolverStereoPosit.h"
#include "optimization/Cg2oOptimizer.h"



class CFundamentalMatcher
{

//ds eigen memory alignment
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//ds custom types
private:

    struct CDetectionPoint
    {
        const UIDDetectionPoint uID;
        const Eigen::Isometry3d matTransformationLEFTtoWORLD;
        const std::shared_ptr< std::vector< CLandmark* > > vecLandmarks;

        CDetectionPoint( const UIDDetectionPoint& p_uID,
                         const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                         const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks ): uID( p_uID ),
                                                                                              matTransformationLEFTtoWORLD( p_matTransformationLEFTtoWORLD ),
                                                                                              vecLandmarks( p_vecLandmarks )
        {
            //ds nothing to do
        }
        ~CDetectionPoint( )
        {
            //ds nothing to do
        }
    };

//ds ctor/dtor
public:

    CFundamentalMatcher( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO );
    ~CFundamentalMatcher( );

//ds members
private:

    //ds triangulation
    std::shared_ptr< CTriangulator > m_pTriangulator;

    //ds cameras (not necessarily used in stereo here)
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;
    const std::shared_ptr< CStereoCamera > m_pCameraSTEREO;

    //ds matching
    const cv::Ptr< cv::Feature2D > m_pDetector;
    const cv::Ptr< cv::Feature2D > m_pExtractor;
    const std::shared_ptr< cv::DescriptorMatcher > m_pMatcher;
    const double m_dMinimumDepthMeters;
    const double m_dMaximumDepthMeters;
    const double m_dMatchingDistanceCutoffTrackingStage1;
    const double m_dMatchingDistanceCutoffTrackingStage2;
    const double m_dMatchingDistanceCutoffTrackingStage3;
    const double m_dMatchingDistanceCutoffOriginal;
    const uint8_t m_uFeatureRadiusForMask = 7;

    //ds measurement point storage (we use the ID counter instead of accessing the vector size every time for speed)
    UIDDetectionPoint m_uAvailableDetectionPointID;
    std::vector< CDetectionPoint > m_vecDetectionPointsActive;
    std::vector< CLandmark* > m_vecVisibleLandmarks;
    std::vector< CLandmark* > m_vecLandmarksGRAPH;
    std::vector< CLandmark* > m_vecLandmarksWINDOW;
    std::vector< const CMeasurementLandmark* > m_vecMeasurementsVisible;
    UIDLandmark m_uAvailableLandmarkID = 0;
    const UIDFrame m_uMaximumAgeWithoutKeyFraming = 100;

    //ds internal
    const uint8_t m_uMaximumFailedSubsequentTrackingsPerLandmark = 5;
    const uint8_t m_uRecursionLimitEpipolarLines               = 2;
    const uint8_t m_uRecursionStepSize                         = 2;
    UIDLandmark m_uNumberOfFailedLandmarkOptimizationsTotal    = 0;
    UIDLandmark m_uNumberOfInvalidLandmarksTotal               = 0;
    UIDLandmark m_uNumberOfTracksStage1   = 0;
    UIDLandmark m_uNumberOfTracksStage2_1 = 0;
    UIDLandmark m_uNumberOfTracksStage3   = 0;
    UIDLandmark m_uNumberOfTracksStage2_2 = 0;
    const double m_dEpipolarLineBaseLength = 15.0; //15.0;

    //ds posit solving
    const uint8_t m_uSearchBlockSizePoseOptimization = 15; //15

    //ds posit solver
    CSolverStereoPosit m_cSolverSterePosit;

    //ds timing
    double m_dDurationTotalSecondsRegionalTrackingL1 = 0.0;
    double m_dDurationTotalSecondsRegionalTrackingR1 = 0.0;
    double m_dDurationTotalSecondsRegionalTrackingL2 = 0.0;
    double m_dDurationTotalSecondsRegionalTrackingR2 = 0.0;
    double m_dDurationTotalSecondsRegionalTrackingFailed = 0.0;
    double m_dDurationTotalSecondsEpipolarTracking = 0.0;

//ds api
public:

    //ds adds new landmarks to the SLAM system
    const std::vector< CLandmark* >::size_type addNewLandmarks( const cv::Mat& p_matImageLEFT,
                                                                const cv::Mat& p_matImageRIGHT,
                                                                const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                                                const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                const UIDFrame& p_uIDFrame,
                                                                cv::Mat& p_matDisplaySTEREO );

    //ds add current detected landmarks to the matcher
    void addDetectionPoint( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks );

    //ds adds landmarks in the current WINDOW to the passed graph optimizer
    void addLandmarksToGraph( Cg2oOptimizer& p_cGraphOptimizer, const Eigen::Vector3d& p_vecTranslationToG2o, const UIDFrame& p_uIDFrame );

    //ds routine that resets the visibility of all active landmarks
    void resetVisibilityActiveLandmarks( );

    //ds register keyframing on currently visible landmarks
    void setKeyFrameToVisibleLandmarks( );

    //ds trigger optimization
    void optimizeActiveLandmarks( const UIDFrame& p_uFrame ) const;

    //ds returns a handle to all currently visible landmarks
    const std::shared_ptr< const std::vector< CLandmark* > > getVisibleOptimizedLandmarks( ) const;

    //ds returns cloud version of currently visible landmarks
    const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > getCloudForVisibleOptimizedLandmarks( const UIDFrame& p_uFrame ) const;

    //ds wrapper for measurement vector
    const std::vector< const CMeasurementLandmark* > getMeasurementsForVisibleLandmarks( );

    const Eigen::Isometry3d getPoseStereoPosit( const UIDFrame p_uFrame,
                                                    cv::Mat& p_matDisplayLEFT,
                                                    cv::Mat& p_matDisplayRIGHT,
                                                    const cv::Mat& p_matImageLEFT,
                                                    const cv::Mat& p_matImageRIGHT,
                                                    const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                                                    const Eigen::Isometry3d& p_matTransformationWORLDtoLEFTLAST,
                                                    const Eigen::Vector3d& p_vecRotationTotal,
                                                    const Eigen::Vector3d& p_vecTranslationTotal,
                                                    const double& p_dMotionScaling );

    void trackEpipolar( const UIDFrame p_uFrame,
                       const cv::Mat& p_matImageLEFT,
                       const cv::Mat& p_matImageRIGHT,
                       const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                       const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                       const double& p_dMotionScaling,
                       cv::Mat& p_matDisplayLEFT,
                       cv::Mat& p_matDisplayRIGHT );

    void trackManual( const UIDFrame p_uFrame,
                       const cv::Mat& p_matImageLEFT,
                       const cv::Mat& p_matImageRIGHT,
                       const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                       const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                       const double& p_dMotionScaling,
                       cv::Mat& p_matDisplayLEFT,
                       cv::Mat& p_matDisplayRIGHT );

    //ds returns an image mask containing the currently visible landmarks (used to avoid re-detection of identical features)
    const cv::Mat getMaskVisibleLandmarks( ) const;

    //ds returns a mask containing all reprojections of currently active landmarks (more than visible)
    const cv::Mat getMaskActiveLandmarks( const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT, cv::Mat& p_matDisplayLEFT ) const;

    //ds draws currently visible landmarks to the screen
    void drawVisibleLandmarks( cv::Mat& p_matDisplayLEFT, cv::Mat& p_matDisplayRIGHT, const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT ) const;

    //ds shift active landmarks
    void shiftActiveLandmarks( const Eigen::Vector3d& p_vecTranslation );
    void rotateActiveLandmarks( const Eigen::Matrix3d& p_matRotation );
    void clearActiveLandmarksMeasurements( const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT );
    void refreshActiveLandmarksMeasurements( const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT );

    //ds returns copy of the vector holding the currently visible landmarks
    const std::vector< CLandmark* > getVisibleLandmarks( ) const { return m_vecVisibleLandmarks; }

    //ds informative
    const std::vector< CDetectionPoint >::size_type getNumberOfDetectionPointsActive( ) const { return m_vecDetectionPointsActive.size( ); }
    const UIDDetectionPoint getNumberOfDetectionPointsTotal( ) const { return m_uAvailableDetectionPointID; }
    const UIDLandmark getNumberOfInvalidLandmarksTotal( ) const { return m_uNumberOfInvalidLandmarksTotal; }
    const UIDLandmark getNumberOfFailedLandmarkOptimizations( ) const { return m_uNumberOfFailedLandmarkOptimizationsTotal; }
    const UIDLandmark getNumberOfTracksStage1( ) const { return m_uNumberOfTracksStage1; }
    const UIDLandmark getNumberOfTracksStage2_1( ) const { return m_uNumberOfTracksStage2_1; }
    const UIDLandmark getNumberOfTracksStage3( ) const { return m_uNumberOfTracksStage3; }
    const UIDLandmark getNumberOfTracksStage2_2( ) const { return m_uNumberOfTracksStage2_2; }
    const std::vector< CLandmark* >::size_type getNumberOfVisibleLandmarks( ) const { return m_vecVisibleLandmarks.size( ); }
    const UIDLandmark getNumberOfLandmarksInGRAPH( ) const { return m_vecLandmarksGRAPH.size( ); }
    const UIDLandmark getNumberOfLandmarksInWINDOW( ) const { return m_vecLandmarksWINDOW.size( ); }
    const UIDLandmark getNumberOfLandmarksTotal( ) const { return m_uAvailableLandmarkID; }
    const double getDurationTotalSecondsStereoPosit( ) const { return m_cSolverSterePosit.getDurationTotalSeconds( ); }
    const double getDurationTotalSecondsRegionalTrackingL1( ) const { return m_dDurationTotalSecondsRegionalTrackingL1; }
    const double getDurationTotalSecondsRegionalTrackingR1( ) const { return m_dDurationTotalSecondsRegionalTrackingR1; }
    const double getDurationTotalSecondsRegionalTrackingL2( ) const { return m_dDurationTotalSecondsRegionalTrackingL2; }
    const double getDurationTotalSecondsRegionalTrackingR2( ) const { return m_dDurationTotalSecondsRegionalTrackingR2; }
    const double getDurationTotalSecondsRegionalTrackingFailed( ) const { return m_dDurationTotalSecondsRegionalTrackingFailed; }
    const double getDurationTotalSecondsEpipolarTracking( ) const { return m_dDurationTotalSecondsEpipolarTracking; }

private:

    const std::shared_ptr< CMatchTracking > _getMatchSampleRecursiveU( cv::Mat& p_matDisplay,
                                                                const cv::Mat& p_matImage,
                                                                const double& p_dUMinimum,
                                                                const uint32_t& p_uDeltaU,
                                                                const Eigen::Vector3d& p_vecCoefficients,
                                                                const CDescriptor& p_matReferenceDescriptor,
                                                                const CDescriptor& p_matOriginalDescriptor,
                                                                const float& p_fKeyPointSize,
                                                                const uint8_t& p_uRecursionDepth ) const;

    const std::shared_ptr< CMatchTracking > _getMatchSampleRecursiveV( cv::Mat& p_matDisplay,
                                                                const cv::Mat& p_matImage,
                                                                const double& p_dVMinimum,
                                                                const uint32_t& p_uDeltaV,
                                                                const Eigen::Vector3d& p_vecCoefficients,
                                                                const CDescriptor& p_matReferenceDescriptor,
                                                                const CDescriptor& p_matOriginalDescriptor,
                                                                const float& p_fKeyPointSize,
                                                                const uint8_t& p_uRecursionDepth ) const;

    const std::shared_ptr< CMatchTracking > _getMatch( const cv::Mat& p_matImage,
                                     std::vector< cv::KeyPoint >& p_vecPoolKeyPoints,
                                     const cv::Point2f& p_ptOffsetROI,
                                     const CDescriptor& p_matDescriptorReference,
                                     const CDescriptor& p_matDescriptorOriginal ) const;

    void _addMeasurementToLandmarkLEFT( const UIDFrame p_uFrame,
                                   CLandmark* p_pLandmark,
                                   const cv::Mat& p_matImageRIGHT,
                                   const cv::KeyPoint& p_cKeyPoint,
                                   const CDescriptor& p_matDescriptorNew,
                                   const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                   const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                   const MatrixProjection& p_matProjectionWORLDtoLEFT,
                                   const MatrixProjection& p_matProjectionWORLDtoRIGHT,
                                   const double& p_dMotionScaling );

    void _addMeasurementToLandmarkSTEREO( const UIDFrame p_uFrame,
                                    const CSolverStereoPosit::CMatch& p_cMatchSTEREO,
                                    const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                    const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                    const MatrixProjection& p_matProjectionWORLDtoLEFT,
                                    const MatrixProjection& p_matProjectionWORLDtoRIGHT );

    void _addMeasurementToLandmarkSTEREO( const UIDFrame p_uFrame,
                                          CLandmark* p_pLandmark,
                                          const cv::Point2d& p_ptUVLEFT,
                                          const cv::Point2d& p_ptUVRIGHT,
                                          const CPoint3DCAMERA& p_vecPointXYZLEFT,
                                          const CDescriptor& p_matDescriptorLEFT,
                                          const CDescriptor& p_matDescriptorRIGHT,
                                          const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                          const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                          const MatrixProjection& p_matProjectionWORLDtoLEFT,
                                          const MatrixProjection& p_matProjectionWORLDtoRIGHT );

    inline const double _getCurveU( const Eigen::Vector3d& p_vecCoefficients, const double& p_dV ) const;
    inline const double _getCurveV( const Eigen::Vector3d& p_vecCoefficients, const double& p_dU ) const;

};

#endif //#define CFUNDAMENTALMATCHER_H
