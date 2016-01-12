#include "CTrackerStereo.h"

#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>

#include "gui/CConfigurationOpenCV.h"
#include "exceptions/CExceptionPoseOptimization.h"
#include "exceptions/CExceptionNoMatchFound.h"

CTrackerStereo::CTrackerStereo( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                                const std::shared_ptr< CIMUInterpolator > p_pIMUInterpolator,
                                std::shared_ptr< CHandleLandmarks > p_hLandmarks,
                                std::shared_ptr< CHandleMapping > p_hMappingThread,
                                const EPlaybackMode& p_eMode,
                                const uint32_t& p_uWaitKeyTimeoutMS ): m_uWaitKeyTimeoutMS( p_uWaitKeyTimeoutMS ),
                                                                           m_pCameraLEFT( p_pCameraSTEREO->m_pCameraLEFT ),
                                                                           m_pCameraRIGHT( p_pCameraSTEREO->m_pCameraRIGHT ),
                                                                           m_pCameraSTEREO( p_pCameraSTEREO ),

                                                                           m_hLandmarks( p_hLandmarks ),
                                                                           m_hMapper( p_hMappingThread ),

                                                                           m_matTransformationWORLDtoLEFTLAST( p_pIMUInterpolator->getTransformationWORLDtoCAMERA( m_pCameraLEFT->m_matRotationIMUtoCAMERA ) ),
                                                                           m_matTransformationLEFTLASTtoLEFTNOW( Eigen::Matrix4d::Identity( ) ),
                                                                           m_vecVelocityAngularFilteredLAST( 0.0, 0.0, 0.0 ),
                                                                           m_vecLinearAccelerationFilteredLAST( 0.0, 0.0, 0.0 ),
                                                                           m_vecPositionKeyFrameLAST( m_matTransformationWORLDtoLEFTLAST.inverse( ).translation( ) ),
                                                                           m_vecCameraOrientationAccumulated( 0.0, 0.0, 0.0 ),
                                                                           m_vecPositionCurrent( m_vecPositionKeyFrameLAST ),
                                                                           m_vecPositionLAST( m_vecPositionCurrent ),

                                                                           //ds BRIEF (calibrated 2015-05-31)
                                                                           // m_uKeyPointSize( 7 ),
                                                                           m_pDetector( std::make_shared< cv::GoodFeaturesToTrackDetector >( 200, 0.01, 7.0, 7, true ) ),
                                                                           m_pExtractor( std::make_shared< cv::BriefDescriptorExtractor >( 32 ) ),
                                                                           m_pMatcher( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) ),
                                                                           m_dMatchingDistanceCutoffTriangulation( 100.0 ),
                                                                           m_dMatchingDistanceCutoffPoseOptimization( 50.0 ),
                                                                           m_dMatchingDistanceCutoffEpipolar( 50.0 ),
                                                                           m_uVisibleLandmarksMinimum( 100 ),

                                                                           m_pTriangulator( std::make_shared< CTriangulator >( m_pCameraSTEREO, m_pExtractor, m_pMatcher, m_dMatchingDistanceCutoffTriangulation ) ),
                                                                           m_cMatcher( m_pTriangulator, m_hLandmarks, m_pDetector, m_dMinimumDepthMeters, m_dMaximumDepthMeters, m_dMatchingDistanceCutoffPoseOptimization, m_dMatchingDistanceCutoffEpipolar, m_uMaximumFailedSubsequentTrackingsPerLandmark ),

                                                                           m_pIMU( p_pIMUInterpolator ),

                                                                           m_eMode( p_eMode )
{
    m_vecRotations.clear( );

    //ds windows
    _initializeTranslationWindow( );

    //ds set opencv parallelization threads (0: disabled)
    cv::setNumThreads( 0 );

    //ds initialize reference frames with black images
    m_matDisplayLowerReference = cv::Mat::zeros( m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );

    //ds initialize the window
    cv::namedWindow( "vi_mapper [L|R]", cv::WINDOW_AUTOSIZE );

    CLogger::openBox( );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) <OpenCV> available CPUs: %i\n", m_uFrameCount, cv::getNumberOfCPUs( ) );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) <OpenCV> available threads: %i\n", m_uFrameCount, cv::getNumThreads( ) );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) feature detector: %s\n", m_uFrameCount, m_pDetector->name( ).c_str( ) );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) descriptor extractor: %s\n", m_uFrameCount, m_pExtractor->name( ).c_str( ) );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) descriptor matcher: %s\n", m_uFrameCount, m_pMatcher->name( ).c_str( ) );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) descriptor size: %i bytes\n", m_uFrameCount, m_pExtractor->descriptorSize( ) );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) <CIMUInterpolator> maximum timestamp delta: %f\n", m_uFrameCount, CIMUInterpolator::dMaximumDeltaTimeSeconds );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) <CIMUInterpolator> imprecision angular velocity: %f\n", m_uFrameCount, CIMUInterpolator::m_dImprecisionAngularVelocity );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) <CIMUInterpolator> imprecision linear acceleration: %f\n", m_uFrameCount, CIMUInterpolator::m_dImprecisionLinearAcceleration );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) <CIMUInterpolator> bias linear acceleration x/y/z: %4.2f/%4.2f/%4.2f\n", m_uFrameCount, CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[0],
                                                                                                                                                  CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[1],
                                                                                                                                                  CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[2] );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) <Landmark> cap iterations: %u\n", m_uFrameCount, CLandmark::uCapIterations );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) <Landmark> convergence delta: %f\n", m_uFrameCount, CLandmark::dConvergenceDelta );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) <Landmark> maximum error L2 inlier: %f\n", m_uFrameCount, CLandmark::dKernelMaximumErrorSquaredPixels );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) <Landmark> maximum error L2 average: %f\n", m_uFrameCount, CLandmark::dMaximumErrorSquaredAveragePixels );
    std::printf( "[0][%06lu]<CTrackerStereo>(CTrackerStereo) instance allocated\n", m_uFrameCount );
    CLogger::closeBox( );
}

CTrackerStereo::~CTrackerStereo( )
{
    /*ds close loggers
    CLogger::CLogLandmarkCreation::close( );
    CLogger::CLogLandmarkFinal::close( );
    CLogger::CLogLandmarkFinalOptimized::close( );
    CLogger::CLogTrajectory::close( );
    CLogger::CLogIMUInput::close( );*/

    std::printf( "[0][%06lu]<CTrackerStereo>(~CTrackerStereo) instance deallocated\n", m_uFrameCount );
}

void CTrackerStereo::receivevDataVI( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
                                                const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageRIGHT,
                                                const std::shared_ptr< txt_io::CIMUMessage > p_pIMU )
{
    //ds preprocessed images
    cv::Mat matPreprocessedLEFT( p_pImageLEFT->image( ) );
    cv::Mat matPreprocessedRIGHT( p_pImageRIGHT->image( ) );

    //ds preprocess images
    cv::equalizeHist( p_pImageLEFT->image( ), matPreprocessedLEFT );
    cv::equalizeHist( p_pImageRIGHT->image( ), matPreprocessedRIGHT );
    m_pCameraSTEREO->undistortAndrectify( matPreprocessedLEFT, matPreprocessedRIGHT );

    //ds current timestamp
    const double dTimestampSeconds      = p_pIMU->timestamp( );
    const double dDeltaTimestampSeconds = dTimestampSeconds - m_dTimestampLASTSeconds;

    assert( 0 < dDeltaTimestampSeconds );

    //ds parallel transformation with erased translation
    Eigen::Isometry3d matTransformationRotationOnlyLEFTLASTtoLEFTNOW( m_matTransformationLEFTLASTtoLEFTNOW );
    matTransformationRotationOnlyLEFTLASTtoLEFTNOW.translation( ) = Eigen::Vector3d::Zero( );

    //ds if the delta is acceptable
    if( CIMUInterpolator::dMaximumDeltaTimeSeconds > dDeltaTimestampSeconds )
    {
        //ds compute total rotation
        const Eigen::Vector3d vecRotationTotal( m_vecVelocityAngularFilteredLAST*dDeltaTimestampSeconds );
        const Eigen::Vector3d vecTranslationTotal( 0.5*m_vecLinearAccelerationFilteredLAST*dDeltaTimestampSeconds*dDeltaTimestampSeconds );

        //ds integrate imu input: overwrite rotation
        m_matTransformationLEFTLASTtoLEFTNOW.linear( ) = CMiniVisionToolbox::fromOrientationRodrigues( vecRotationTotal );

        //ds add acceleration
        m_matTransformationLEFTLASTtoLEFTNOW.translation( ) += vecTranslationTotal;

        //ds process images (fed with IMU prior pose)
        _trackLandmarks( matPreprocessedLEFT,
                         matPreprocessedRIGHT,
                         m_matTransformationLEFTLASTtoLEFTNOW*m_matTransformationWORLDtoLEFTLAST,
                         matTransformationRotationOnlyLEFTLASTtoLEFTNOW*m_matTransformationWORLDtoLEFTLAST,
                         p_pIMU->getLinearAcceleration( ),
                         p_pIMU->getAngularVelocity( ),
                         vecRotationTotal,
                         vecTranslationTotal,
                         dDeltaTimestampSeconds );
    }
    else
    {
        //ds compute reduced entities
        const Eigen::Vector3d vecRotationTotalDamped( m_vecVelocityAngularFilteredLAST*CIMUInterpolator::dMaximumDeltaTimeSeconds );
        const Eigen::Vector3d vecTranslationTotalDamped( Eigen::Vector3d::Zero( ) );

        //ds use full angular velocity
        std::printf( "[0][%06lu]<CTrackerStereo>(receivevDataVI) using reduced IMU input, timestamp delta: %f\n", m_uFrameCount, dDeltaTimestampSeconds );

        //ds integrate imu input: overwrite rotation with limited IMU input
        m_matTransformationLEFTLASTtoLEFTNOW.linear( ) = CMiniVisionToolbox::fromOrientationRodrigues( vecRotationTotalDamped );

        //ds process images (fed with IMU prior pose: damped input)
        _trackLandmarks( matPreprocessedLEFT,
                         matPreprocessedRIGHT,
                         m_matTransformationLEFTLASTtoLEFTNOW*m_matTransformationWORLDtoLEFTLAST,
                         matTransformationRotationOnlyLEFTLASTtoLEFTNOW*m_matTransformationWORLDtoLEFTLAST,
                         p_pIMU->getLinearAcceleration( ),
                         p_pIMU->getAngularVelocity( ),
                         vecRotationTotalDamped,
                         vecTranslationTotalDamped,
                         dDeltaTimestampSeconds );
    }

    //ds update timestamp
    m_dTimestampLASTSeconds = dTimestampSeconds;
}

void CTrackerStereo::updateMap( const CMapUpdate& p_cMapUpdate )
{
    //ds info
    std::printf( "[1][%06lu]<CTrackerStereo>(updateMap) received map update - [key frame delta: %lu][landmark delta: %lu]\n", m_uFrameCount, m_uNumberOfKeyFrames-p_cMapUpdate.uIDKeyFrame-1, m_hLandmarks->vecLandmarks->size( )-p_cMapUpdate.uIDLandmark-1 );

    //ds buffer previous value
    const Eigen::Isometry3d m_matTransformationWORLDtoLEFTLASTBeforeOptimization( m_matTransformationWORLDtoLEFTLAST );

    //ds compute original transformation
    const Eigen::Isometry3d matTransformationKeyFrameLASTtoNOW( m_matTransformationWORLDtoLEFTLAST*p_cMapUpdate.matPoseLEFTtoWORLDBeforeOptimization );

    //ds compute new LAST based on updated key frame
    m_matTransformationWORLDtoLEFTLAST = matTransformationKeyFrameLASTtoNOW*p_cMapUpdate.matPoseLEFTtoWORLD.inverse( );

    //ds "world frame transformation" before and after optimization
    const Eigen::Isometry3d matTransformationFromOptimization( m_matTransformationWORLDtoLEFTLAST.inverse( )*m_matTransformationWORLDtoLEFTLASTBeforeOptimization );

    //ds precompute intrinsics
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*m_matTransformationWORLDtoLEFTLAST.matrix( ) );
    const MatrixProjection matProjectionWORLDtoRIGHT( m_pCameraRIGHT->m_matProjection*m_matTransformationWORLDtoLEFTLAST.matrix( ) );

    //ds transform landmarks which were not in the optimization
    for( std::vector< CLandmark* >::size_type uID = p_cMapUpdate.uIDLandmark+1; uID < m_hLandmarks->vecLandmarks->size( ); ++uID )
    {
        //ds buffer landmark
        CLandmark* pLandmark = m_hLandmarks->vecLandmarks->at( uID );

        //ds update position
        pLandmark->vecPointXYZOptimized = matTransformationFromOptimization*pLandmark->vecPointXYZOptimized;
        pLandmark->clearMeasurements( pLandmark->vecPointXYZOptimized, m_matTransformationWORLDtoLEFTLAST, matProjectionWORLDtoLEFT, matProjectionWORLDtoRIGHT );
    }
}

void CTrackerStereo::finalize( )
{
    //ds if tracker GUI is still open - otherwise run the optimization right away
    if( !m_bIsShutdownRequested )
    {
        //ds inform
        std::printf( "[0][%06lu]<CTrackerStereo>(finalize) press any key to perform final optimization\n", m_uFrameCount );

        //ds also display on image
        cv::Mat matDisplayComplete = cv::Mat( 2*m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3, cv::Scalar( 0.0 ) );
        cv::putText( matDisplayComplete, "DATASET COMPLETE - PRESS ANY KEY TO EXIT" , cv::Point2i( 50, 50 ), cv::FONT_HERSHEY_PLAIN, 2.0, CColorCodeBGR( 255, 255, 255 ) );
        cv::imshow( "vi_mapper [L|R]", matDisplayComplete );

        //ds wait for any user input
        cv::waitKey( 0 );
    }
    else
    {
        std::printf( "[0][%06lu]<CTrackerStereo>(finalize) running final optimization\n", m_uFrameCount );
    }

    //ds trigger shutdown
    m_bIsShutdownRequested = true;
}

void CTrackerStereo::_trackLandmarks( const cv::Mat& p_matImageLEFT,
                                                 const cv::Mat& p_matImageRIGHT,
                                                 const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                                                 const Eigen::Isometry3d& p_matTransformationEstimateParallelWORLDtoLEFT,
                                                 const CLinearAccelerationIMU& p_vecLinearAcceleration,
                                                 const CAngularVelocityIMU& p_vecAngularVelocity,
                                                 const Eigen::Vector3d& p_vecRotationTotal,
                                                 const Eigen::Vector3d& p_vecTranslationTotal,
                                                 const double& p_dDeltaTimeSeconds )
{
    //ds get images into triple channel mats (display only)
    cv::Mat matDisplayLEFT;
    cv::Mat matDisplayRIGHT;

    //ds get images to triple channel for colored display
    cv::cvtColor( p_matImageLEFT, matDisplayLEFT, cv::COLOR_GRAY2BGR );
    cv::cvtColor( p_matImageRIGHT, matDisplayRIGHT, cv::COLOR_GRAY2BGR );

    //ds get clean copies
    const cv::Mat matDisplayLEFTClean( matDisplayLEFT.clone( ) );
    const cv::Mat matDisplayRIGHTClean( matDisplayRIGHT.clone( ) );

    //ds compute motion scaling (capped)
    const double dMotionScaling = std::min( 1.0+100*( p_vecRotationTotal.squaredNorm( )+p_vecTranslationTotal.squaredNorm( ) ), 2.0 );

    //ds get landmark lock mutex (locks at construction)
    std::unique_lock< std::mutex > cLockLandmarks( m_hLandmarks->cMutex );

    //ds refresh landmark states
    m_cMatcher.resetVisibilityActiveLandmarks( );

    //ds initial transformation
    Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationEstimateWORLDtoLEFT );

    try
    {
        //ds get the optimized pose
        matTransformationWORLDtoLEFT = m_cMatcher.getPoseStereoPosit( m_uFrameCount,
                                                                          matDisplayLEFT,
                                                                          matDisplayRIGHT,
                                                                          p_matImageLEFT,
                                                                          p_matImageRIGHT,
                                                                          p_matTransformationEstimateWORLDtoLEFT,
                                                                          m_matTransformationWORLDtoLEFTLAST,
                                                                          p_vecRotationTotal,
                                                                          p_vecTranslationTotal,
                                                                          dMotionScaling );

        //ds if still here it went okay - reduce instability count if necessary
        if( 0 < m_uCountInstability )
        {
            --m_uCountInstability;
        }
    }
    catch( const CExceptionPoseOptimization& p_cException )
    {
        //std::printf( "<CTrackerStereo>(_trackLandmarks) pose optimization failed [RAW PRIOR]: '%s'\n", p_cException.what( ) );
        try
        {
            //ds get the optimized pose on constant motion
            matTransformationWORLDtoLEFT = m_cMatcher.getPoseStereoPosit( m_uFrameCount,
                                                                              matDisplayLEFT,
                                                                              matDisplayRIGHT,
                                                                              p_matImageLEFT,
                                                                              p_matImageRIGHT,
                                                                              m_matTransformationWORLDtoLEFTLAST,
                                                                              m_matTransformationWORLDtoLEFTLAST,
                                                                              p_vecRotationTotal,
                                                                              p_vecTranslationTotal,
                                                                              dMotionScaling );
        }
        catch( const CExceptionPoseOptimization& p_cException )
        {
            //ds if not capped already
            if( 20 > m_uCountInstability )
            {
                m_uCountInstability += 5;
            }
            //std::printf( "[0][%06lu]<CTrackerStereo>(_trackLandmarks) pose optimization failed [DAMPED PRIOR]: '%s' - running on damped IMU only\n", m_uFrameCount, p_cException.what( ) );

            //ds compute damped rotation
            Eigen::Vector3d vecRotationYZ( p_vecRotationTotal );
            vecRotationYZ.x( ) = 0.0;
            matTransformationWORLDtoLEFT = CMiniVisionToolbox::fromOrientationRodrigues( vecRotationYZ )*m_matTransformationWORLDtoLEFTLAST;
        }
    }

    //ds get inverse pose
    Eigen::Isometry3d matTransformationLEFTtoWORLD( matTransformationWORLDtoLEFT.inverse( ) );

    //ds get current measurements (including landmarks already detected in the pose optimization)
    m_cMatcher.trackEpipolar( m_uFrameCount,
                              p_matImageLEFT,
                              p_matImageRIGHT,
                              matTransformationWORLDtoLEFT,
                              matTransformationLEFTtoWORLD,
                              dMotionScaling,
                              matDisplayLEFT,
                              matDisplayRIGHT );
    int32_t uNumberOfVisibleLandmarks = m_cMatcher.getNumberOfVisibleLandmarks( );

    //ds unlock landmarks
    cLockLandmarks.unlock( );

    //ds compute landmark lost since last (negative if we see more landmarks than before)
    const double iLandmarksLost = m_uNumberofVisibleLandmarksLAST-uNumberOfVisibleLandmarks;

    //ds if we lose more than 75% landmarks in one frame
    if( 0.75 < iLandmarksLost/m_uNumberofVisibleLandmarksLAST )
    {
        //ds if not capped already
        if( 20 > m_uCountInstability )
        {
            m_uCountInstability += 5;
        }

        std::printf( "[0][%06lu]<CTrackerStereo>(_trackLandmarks) lost track (landmarks visible: %3i lost: %3i), total delta: %f (%f %f %f), motion scaling: %f\n",
                     m_uFrameCount, uNumberOfVisibleLandmarks, static_cast< int32_t >( iLandmarksLost ), m_vecVelocityAngularFilteredLAST.squaredNorm( ), m_vecVelocityAngularFilteredLAST.x( ), m_vecVelocityAngularFilteredLAST.y( ), m_vecVelocityAngularFilteredLAST.z( ), dMotionScaling );
        m_uWaitKeyTimeoutMS = 0;
    }

    //ds estimate acceleration in current WORLD frame (necessary to filter gravity)
    const Eigen::Matrix3d matRotationIMUtoWORLD( matTransformationLEFTtoWORLD.linear( )*m_pCameraLEFT->m_matRotationIMUtoCAMERA );
    const CLinearAccelerationWORLD vecLinearAccelerationWORLD( matRotationIMUtoWORLD*p_vecLinearAcceleration );
    const CLinearAccelerationWORLD vecLinearAccelerationWORLDFiltered( CIMUInterpolator::getLinearAccelerationFiltered( vecLinearAccelerationWORLD ) );

    //ds get angular velocity filtered
    const CAngularVelocityIMU vecAngularVelocityFiltered( CIMUInterpolator::getAngularVelocityFiltered( p_vecAngularVelocity ) );

    //ds update IMU input references
    m_vecLinearAccelerationFilteredLAST = matTransformationWORLDtoLEFT.linear( )*vecLinearAccelerationWORLDFiltered;
    m_vecVelocityAngularFilteredLAST    = m_pCameraLEFT->m_matRotationIMUtoCAMERA*vecAngularVelocityFiltered;

    //ds current translation
    m_vecPositionLAST    = m_vecPositionCurrent;
    m_vecPositionCurrent = matTransformationLEFTtoWORLD.translation( );

    //ds update reference
    m_uNumberofVisibleLandmarksLAST = uNumberOfVisibleLandmarks;

    //ds display measurements (blocks)
    cLockLandmarks.lock( );
    m_cMatcher.drawVisibleLandmarks( matDisplayLEFT, matDisplayRIGHT, matTransformationWORLDtoLEFT );

    //ds optimize landmarks periodically
    if( 0 == m_uFrameCount%m_uLandmarkOptimizationEveryNFrames )
    {
        //ds blocks
        m_cMatcher.optimizeActiveLandmarks( m_uFrameCount );
    }
    cLockLandmarks.unlock( );

    //ds update scene in viewer
    m_prFrameLEFTtoWORLD = std::pair< bool, Eigen::Isometry3d >( false, matTransformationLEFTtoWORLD );

    //ds accumulate orientation
    m_vecCameraOrientationAccumulated += p_vecRotationTotal;

    //ds position delta
    m_dTranslationDeltaSquaredNormCurrent = ( m_vecPositionCurrent-m_vecPositionKeyFrameLAST ).squaredNorm( );

    //ds translation history
    m_vecTranslationDeltas.push_back( m_vecPositionCurrent-m_vecPositionLAST );
    m_vecRotations.push_back( p_vecRotationTotal );

    //ds current element count
    const std::vector< Eigen::Vector3d >::size_type uMaximumIndex = m_vecTranslationDeltas.size( )-1;
    const std::vector< Eigen::Vector3d >::size_type uMinimumIndex = uMaximumIndex-m_uIMULogbackSize+1;

    //ds remove first and add recent
    m_vecGradientXYZ -= m_vecTranslationDeltas[uMinimumIndex];
    m_vecGradientXYZ += m_vecTranslationDeltas[uMaximumIndex];

    //ds add a keyframe if valid
    if( m_dTranslationDeltaForKeyFrameMetersL2 < m_dTranslationDeltaSquaredNormCurrent       ||
        m_dAngleDeltaForKeyFrameRadiansL2 < m_vecCameraOrientationAccumulated.squaredNorm( ) ||
        m_uFrameDifferenceForKeyFrame < m_uFrameCount-m_uFrameKeyFrameLAST                   )
    {
        //ds compute cloud for current keyframe (also optimizes landmarks!)
        cLockLandmarks.lock( );
        const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > vecCloud = m_cMatcher.getCloudForVisibleOptimizedLandmarks( m_uFrameCount );

        //ds if the number of points in the cloud is sufficient
        if( m_uMinimumLandmarksForKeyFrame < vecCloud->size( ) )
        {
            //ds register keyframing to matcher
            m_cMatcher.setKeyFrameToVisibleLandmarks( );

            {
                //ds signal thread with new key frame and go on
                std::lock_guard< std::mutex > cLockGuardMapper( m_hMapper->cMutex );
                m_hMapper->vecKeyFramesToAdd.push_back( new CKeyFrame( m_uNumberOfKeyFrames,
                                                                              m_uFrameCount,
                                                                              matTransformationLEFTtoWORLD,
                                                                              p_vecLinearAcceleration.normalized( ),
                                                                              m_cMatcher.getMeasurementsForVisibleLandmarks( ),
                                                                              vecCloud,
                                                                              m_uCountInstability,
                                                                              dMotionScaling ) );
            }
            m_hMapper->cConditionVariable.notify_one( );
            ++m_uNumberOfKeyFrames;

            //ds update references
            m_vecPositionKeyFrameLAST         = m_vecPositionCurrent;
            m_vecCameraOrientationAccumulated = Eigen::Vector3d::Zero( );
            m_uFrameKeyFrameLAST              = m_uFrameCount;

            //ds update scene in viewer with keyframe transformation
            m_prFrameLEFTtoWORLD = std::pair< bool, Eigen::Isometry3d >( true, matTransformationLEFTtoWORLD );
        }
        else
        {
            std::printf( "[0][%06lu]<CTrackerStereo>(_trackLandmarks) not enough points to create key frame: %lu < %lu\n", m_uFrameCount, vecCloud->size( ), m_uMinimumLandmarksForKeyFrame );
        }
        cLockLandmarks.unlock( );
    }

    //ds frame available for viewer
    m_bIsFrameAvailable = true;

    //ds check if we have to detect new landmarks
    if( m_uVisibleLandmarksMinimum > m_uNumberofVisibleLandmarksLAST || m_uMaximumNumberOfFramesWithoutDetection < m_uNumberOfFramesWithoutDetection )
    {
        //ds clean the lower display (to show detection details)
        cv::hconcat( matDisplayLEFTClean, matDisplayRIGHTClean, m_matDisplayLowerReference );

        //ds detect new landmarks (blocks partially)
        _addNewLandmarks( p_matImageLEFT, p_matImageRIGHT, matTransformationWORLDtoLEFT, matTransformationLEFTtoWORLD, m_matDisplayLowerReference );

        //ds reset counter
        m_uNumberOfFramesWithoutDetection = 0;
    }
    else
    {
        ++m_uNumberOfFramesWithoutDetection;
    }

    //ds build display mat
    cv::Mat matDisplayUpper = cv::Mat( m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );
    cv::hconcat( matDisplayLEFT, matDisplayRIGHT, matDisplayUpper );
    _drawInfoBox( matDisplayUpper, dMotionScaling );
    cv::Mat matDisplayComplete = cv::Mat( 2*m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );
    cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );

    //ds check if unstable
    if( 0 < m_uCountInstability )
    {
        //ds draw red frame around display
        cv::rectangle( matDisplayComplete, cv::Point2i( 0, 17 ), cv::Point2i( 2*m_pCameraSTEREO->m_uPixelWidth, m_pCameraSTEREO->m_uPixelHeight ), CColorCodeBGR( 0, 0, 255 ), m_uCountInstability );
    }

    //ds display
    cv::imshow( "vi_mapper [L|R]", matDisplayComplete );

    //ds if there was a keystroke
    int iLastKeyStroke( cv::waitKey( m_uWaitKeyTimeoutMS ) );
    if( -1 != iLastKeyStroke )
    {
        //ds user input - reset frame rate counting
        m_uFramesCurrentCycle = 0;

        //ds evaluate keystroke
        switch( iLastKeyStroke )
        {
            case CConfigurationOpenCV::KeyStroke::iEscape:
            {
                _shutDown( );
                return;
            }
            case CConfigurationOpenCV::KeyStroke::iBackspace:
            {
                if( 0 < m_uWaitKeyTimeoutMS )
                {
                    //ds switch to stepwise mode
                    m_uWaitKeyTimeoutMS = 0;
                    m_eMode = ePlaybackStepwise;
                    std::printf( "[0][%06lu]<CTrackerStereo>(_trackLandmarks) switched to stepwise mode\n", m_uFrameCount );
                }
                else
                {
                    //ds switch to benchmark mode
                    m_uWaitKeyTimeoutMS = 1;
                    m_eMode = ePlaybackBenchmark;
                    std::printf( "[0][%06lu]<CTrackerStereo>(_trackLandmarks) switched back to benchmark mode\n", m_uFrameCount );
                }
                break;
            }
            default:
            {
                break;
            }
        }
    }
    else
    {
        _updateFrameRateForInfoBox( );
    }

    //ds update references
    ++m_uFrameCount;
    m_matTransformationLEFTLASTtoLEFTNOW = matTransformationWORLDtoLEFT*m_matTransformationWORLDtoLEFTLAST.inverse( );
    m_matTransformationWORLDtoLEFTLAST   = matTransformationWORLDtoLEFT;
    m_dDistanceTraveledMeters           += m_vecTranslationDeltas.back( ).norm( );
    m_dMotionScalingLAST                 = dMotionScaling;

    //ds log final status (after potential optimization)
    //CLogger::CLogIMUInput::addEntry( m_uFrameCount, vecLinearAccelerationWORLD, vecLinearAccelerationWORLDFiltered, p_vecAngularVelocity, vecAngularVelocityFiltered );
    //CLogger::CLogTrajectory::addEntry( m_uFrameCount, m_vecPositionCurrent, Eigen::Quaterniond( matTransformationLEFTtoWORLD.linear( ) ) );
}

void CTrackerStereo::_addNewLandmarks( const cv::Mat& p_matImageLEFT,
                                                  const cv::Mat& p_matImageRIGHT,
                                                  const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                                  const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                  cv::Mat& p_matDisplaySTEREO )
{
    //ds precompute intrinsics
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*p_matTransformationWORLDtoLEFT.matrix( ) );
    const MatrixProjection matProjectionWORLDtoRIGHT( m_pCameraRIGHT->m_matProjection*p_matTransformationWORLDtoLEFT.matrix( ) );

    //ds solution holder
    std::shared_ptr< std::vector< CLandmark* > > vecNewLandmarks( std::make_shared< std::vector< CLandmark* > >( ) );

    //ds key points buffer
    std::vector< cv::KeyPoint > vecKeyPoints;

    //const std::shared_ptr< std::vector< cv::KeyPoint > > vecKeyPoints( m_cDetector.detectKeyPointsTilewise( p_matImageLEFT, matMask ) );
    m_pDetector->detect( p_matImageLEFT, vecKeyPoints, m_cMatcher.getMaskActiveLandmarks( p_matTransformationWORLDtoLEFT, p_matDisplaySTEREO ) );

    //ds compute descriptors for the keypoints
    CDescriptors matReferenceDescriptors;
    //m_pExtractor->compute( p_matImageLEFT, *vecKeyPoints, matReferenceDescriptors );
    m_pExtractor->compute( p_matImageLEFT, vecKeyPoints, matReferenceDescriptors );

    //ds process the keypoints and see if we can use them as landmarks
    for( uint32_t u = 0; u < vecKeyPoints.size( ); ++u )
    {
        //ds current points
        const cv::KeyPoint cKeyPointLEFT( vecKeyPoints[u] );
        const cv::Point2f ptLandmarkLEFT( cKeyPointLEFT.pt );
        const CDescriptor matDescriptorLEFT( matReferenceDescriptors.row(u) );

        try
        {
            //ds triangulate the point
            const CMatchTriangulation cMatchRIGHT( m_pTriangulator->getPointTriangulatedInRIGHTFull( p_matDisplaySTEREO, p_matImageRIGHT,
                                                                                                 std::max( 0.0f, ptLandmarkLEFT.x-CTriangulator::fMinimumSearchRangePixels-4*cKeyPointLEFT.size ),
                                                                                                 ptLandmarkLEFT.y-4*cKeyPointLEFT.size,
                                                                                                 cKeyPointLEFT.size,
                                                                                                 ptLandmarkLEFT,
                                                                                                 matDescriptorLEFT ) );
            const CPoint3DCAMERA vecPointTriangulatedLEFT( cMatchRIGHT.vecPointXYZCAMERA );

            //ds check depth
            const double dDepthMeters( vecPointTriangulatedLEFT.z( ) );

            //ds check if point is in front of camera an not more than a defined distance away
            if( m_dMinimumDepthMeters < dDepthMeters && m_dMaximumDepthMeters > dDepthMeters )
            {
                //ds landmark right
                const cv::Point2f ptLandmarkRIGHT( cMatchRIGHT.ptUVCAMERA );

                //ds allocate a new landmark and add the current position
                CLandmark* pLandmark( new CLandmark( m_uAvailableLandmarkID,
                                                     matDescriptorLEFT,
                                                     cMatchRIGHT.matDescriptorCAMERA,
                                                     cKeyPointLEFT.size,
                                                     ptLandmarkLEFT,
                                                     ptLandmarkRIGHT,
                                                     vecPointTriangulatedLEFT,
                                                     p_matTransformationLEFTtoWORLD,
                                                     p_matTransformationWORLDtoLEFT,
                                                     m_pCameraLEFT->m_matProjection,
                                                     m_pCameraRIGHT->m_matProjection,
                                                     matProjectionWORLDtoLEFT,
                                                     matProjectionWORLDtoRIGHT,
                                                     m_uFrameCount ) );

                //ds log creation
                //CLogger::CLogLandmarkCreation::addEntry( m_uFrameCount, pLandmark, dDepthMeters, ptLandmarkLEFT, ptLandmarkRIGHT );

                //ds add to newly detected
                vecNewLandmarks->push_back( pLandmark );

                //ds next landmark id
                ++m_uAvailableLandmarkID;

                //ds draw detected point
                cv::line( p_matDisplaySTEREO, ptLandmarkLEFT, cv::Point2f( ptLandmarkLEFT.x+m_pCameraSTEREO->m_uPixelWidth, ptLandmarkLEFT.y ), CColorCodeBGR( 175, 175, 175 ) );
                cv::circle( p_matDisplaySTEREO, ptLandmarkLEFT, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
                cv::putText( p_matDisplaySTEREO, std::to_string( pLandmark->uID ) , cv::Point2d( ptLandmarkLEFT.x+pLandmark->dKeyPointSize, ptLandmarkLEFT.y+pLandmark->dKeyPointSize ), cv::FONT_HERSHEY_PLAIN, 0.5, CColorCodeBGR( 0, 0, 255 ) );

                //ds draw reprojection of triangulation
                cv::circle( p_matDisplaySTEREO, cv::Point2d( ptLandmarkRIGHT.x+m_pCameraSTEREO->m_uPixelWidth, ptLandmarkRIGHT.y ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );
            }
            else
            {
                cv::circle( p_matDisplaySTEREO, ptLandmarkLEFT, 3, CColorCodeBGR( 0, 255, 255 ), -1 );
                std::printf( "[0][%06lu]<CTrackerStereo>(_addNewLandmarks) could not find match for keypoint (invalid depth: %f m)\n", m_uFrameCount, dDepthMeters );
            }
        }
        catch( const CExceptionNoMatchFound& p_cException )
        {
            cv::circle( p_matDisplaySTEREO, ptLandmarkLEFT, 3, CColorCodeBGR( 0, 0, 255 ), -1 );
            //std::printf( "[0][%06lu]<CTrackerStereo>(_addNewLandmarks) could not find match for keypoint: '%s'\n", m_uFrameCount, p_cException.what( ) );
        }
    }

    //ds if we couldnt find new landmarks
    if( 0 == vecNewLandmarks->size( ) )
    {
        std::printf( "[0][%06lu]<CTrackerStereo>(_getNewLandmarks) unable to detect new landmarks\n", m_uFrameCount );
    }
    else
    {
        //ds all visible in this frame
        m_uNumberofVisibleLandmarksLAST += vecNewLandmarks->size( );

        {
            //ds lock - blocking - RAII
            std::lock_guard< std::mutex > cLockGuard( m_hLandmarks->cMutex );

            //ds add to permanent reference holder (this will copy the landmark references)
            m_hLandmarks->vecLandmarks->insert( m_hLandmarks->vecLandmarks->end( ), vecNewLandmarks->begin( ), vecNewLandmarks->end( ) );

            //ds add this measurement point to the epipolar matcher (which will remove references from its detection point -> does not affect the landmarks main vector)
            m_cMatcher.addDetectionPoint( p_matTransformationLEFTtoWORLD, vecNewLandmarks );
        }

        //std::printf( "<CTrackerStereo>(_getNewLandmarks) added new landmarks: %lu/%lu\n", vecNewLandmarks->size( ), vecKeyPoints.size( ) );
    }
}


void CTrackerStereo::_updateWORLDFrame( const Eigen::Vector3d& p_vecTranslationWORLD )
{
    assert( false );

    std::printf( "[0][%06lu]<CTrackerStereo>(_updateWORLDFrame) refreshing WORLD frame - ", m_uFrameCount );

    //ds set initial frame in g2o
    //m_cGraphOptimizer.updateSTART( p_vecTranslationWORLD );

    /*ds move all keyframes
    for( CKeyFrame* pKeyFrame: *m_vecKeyFrames )
    {
        pKeyFrame->matTransformationLEFTtoWORLD.translation( ) -= p_vecTranslationWORLD;

        //ds update in g2o
        //m_cGraphOptimizer.updateEstimate( pKeyFrame );
    }*/

    /*ds move all landmarks
    for( CLandmark* pLandmark: *m_vecLandmarks )
    {
        pLandmark->vecPointXYZOptimized -= p_vecTranslationWORLD;

        //ds update in g2o
        //m_cGraphOptimizer.updateEstimate( pLandmark );
    }*/

    //ds done
    std::printf( "complete!\n" );
}

void CTrackerStereo::_initializeTranslationWindow( )
{
    //ds reinitialize translation window
    m_vecTranslationDeltas.clear( );
    for( std::vector< Eigen::Vector3d >::size_type u = 0; u < m_uIMULogbackSize; ++u )
    {
        m_vecTranslationDeltas.push_back( Eigen::Vector3d::Zero( ) );
    }
    m_vecGradientXYZ = Eigen::Vector3d::Zero( );
}

void CTrackerStereo::_shutDown( )
{
    m_bIsShutdownRequested = true;
    std::printf( "[0][%06lu]<CTrackerStereo>(_shutDown) termination requested, <CTrackerStereo> disabled\n", m_uFrameCount );
}

void CTrackerStereo::_updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange )
{
    //ds check if we can compute the frame rate
    if( p_uFrameProbeRange == m_uFramesCurrentCycle )
    {
        //ds get time delta
        const double dDuration = CLogger::getTimeSeconds( )-m_dPreviousFrameTime;

        //ds compute framerate
        m_dPreviousFrameRate = p_uFrameProbeRange/dDuration;

        //ds enable new measurement (will enter the following if case)
        m_uFramesCurrentCycle = 0;
    }

    //ds check if its the first frame since the last count
    if( 0 == m_uFramesCurrentCycle )
    {
        //ds stop time
        m_dPreviousFrameTime = CLogger::getTimeSeconds( );
    }

    //ds count frames
    ++m_uFramesCurrentCycle;
}

void CTrackerStereo::_drawInfoBox( cv::Mat& p_matDisplay, const double& p_dMotionScaling ) const
{
    char chBuffer[1024];

    switch( m_eMode )
    {
        case ePlaybackStepwise:
        {
            std::snprintf( chBuffer, 1024, "[%13.2f|%05lu] STEPWISE | X: %5.1f Y: %5.1f Z: %5.1f DELTA L2: %4.2f MOTION: %4.2f | LANDMARKS V: %3i (%3lu,%3lu,%3lu,%3lu) F: %4lu I: %4lu TOTAL: %4lu | DETECTIONS: %2lu(%3lu) | KFs: %3lu",
                           m_dTimestampLASTSeconds, m_uFrameCount,
                           m_vecPositionCurrent.x( ), m_vecPositionCurrent.y( ), m_vecPositionCurrent.z( ), m_dTranslationDeltaSquaredNormCurrent, p_dMotionScaling,
                           m_uNumberofVisibleLandmarksLAST, m_cMatcher.getNumberOfTracksStage1( ), m_cMatcher.getNumberOfTracksStage2_1( ), m_cMatcher.getNumberOfTracksStage3( ), m_cMatcher.getNumberOfTracksStage2_2( ), m_cMatcher.getNumberOfFailedLandmarkOptimizations( ), m_cMatcher.getNumberOfInvalidLandmarksTotal( ), m_uAvailableLandmarkID,
                           m_cMatcher.getNumberOfDetectionPointsActive( ), m_cMatcher.getNumberOfDetectionPointsTotal( ),
                           m_uNumberOfKeyFrames ); //m_cGraphOptimizer.getNumberOfOptimizations( )
            break;
        }
        case ePlaybackBenchmark:
        {
            std::snprintf( chBuffer, 1024, "[%13.2f|%05lu] BENCHMARK FPS: %4.1f | X: %5.1f Y: %5.1f Z: %5.1f DELTA L2: %4.2f SCALING: %4.2f | LANDMARKS V: %3i (%3lu,%3lu,%3lu,%3lu) F: %4lu I: %4lu TOTAL: %4lu | DETECTIONS: %2lu(%3lu) | KFs: %3lu",
                           m_dTimestampLASTSeconds, m_uFrameCount, m_dPreviousFrameRate,
                           m_vecPositionCurrent.x( ), m_vecPositionCurrent.y( ), m_vecPositionCurrent.z( ), m_dTranslationDeltaSquaredNormCurrent, p_dMotionScaling,
                           m_uNumberofVisibleLandmarksLAST, m_cMatcher.getNumberOfTracksStage1( ), m_cMatcher.getNumberOfTracksStage2_1( ), m_cMatcher.getNumberOfTracksStage3( ), m_cMatcher.getNumberOfTracksStage2_2( ), m_cMatcher.getNumberOfFailedLandmarkOptimizations( ), m_cMatcher.getNumberOfInvalidLandmarksTotal( ), m_uAvailableLandmarkID,
                           m_cMatcher.getNumberOfDetectionPointsActive( ), m_cMatcher.getNumberOfDetectionPointsTotal( ),
                           m_uNumberOfKeyFrames );
            break;
        }
        default:
        {
            std::printf( "[0][%06lu]<CTrackerStereo>(_drawInfoBox) unsupported playback mode, no info box displayed\n", m_uFrameCount );
            break;
        }
    }


    p_matDisplay( cv::Rect( 0, 0, 2*m_pCameraSTEREO->m_uPixelWidth, 17 ) ).setTo( CColorCodeBGR( 0, 0, 0 ) );
    cv::putText( p_matDisplay, chBuffer , cv::Point2i( 2, 12 ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );
}
