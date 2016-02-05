#include "CTrackerSVI.h"

#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>

#include "../gui/CConfigurationOpenCV.h"
#include "../exceptions/CExceptionPoseOptimization.h"
#include "../exceptions/CExceptionNoMatchFound.h"



CTrackerSVI::CTrackerSVI( const std::shared_ptr< CStereoCameraIMU > p_pCameraSTEREO,
                                const std::shared_ptr< CIMUInterpolator > p_pIMUInterpolator,
                                const EPlaybackMode& p_eMode,
                                const uint32_t& p_uWaitKeyTimeoutMS ): m_uWaitKeyTimeoutMS( p_uWaitKeyTimeoutMS ),
                                                                           m_pCameraLEFT( p_pCameraSTEREO->m_pCameraLEFT ),
                                                                           m_pCameraRIGHT( p_pCameraSTEREO->m_pCameraRIGHT ),
                                                                           m_pCameraSTEREO( p_pCameraSTEREO ),

                                                                           m_vecLandmarks( std::make_shared< std::vector< CLandmark* > >( ) ),
                                                                           m_vecKeyFrames( std::make_shared< std::vector< CKeyFrame* > >( ) ),

                                                                           m_matTransformationWORLDtoLEFTLAST( p_pIMUInterpolator->getTransformationWORLDtoCAMERA( m_pCameraLEFT->m_matRotationIMUtoCAMERA ) ),
                                                                           m_matTransformationLEFTLASTtoLEFTNOW( Eigen::Matrix4d::Identity( ) ),
                                                                           m_vecPositionKeyFrameLAST( m_matTransformationWORLDtoLEFTLAST.inverse( ).translation( ) ),
                                                                           m_vecPositionCurrent( m_vecPositionKeyFrameLAST ),
                                                                           m_vecPositionLAST( m_vecPositionCurrent ),

                                                                           //ds BRIEF (calibrated 2015-05-31)
                                                                           // m_uKeyPointSize( 7 ),
                                                                           m_pDetector( std::make_shared< cv::GoodFeaturesToTrackDetector >( 1000, 0.01, 7.0, 7, true ) ),
                                                                           m_pExtractor( std::make_shared< cv::BriefDescriptorExtractor >( DESCRIPTOR_SIZE_BYTES ) ),
                                                                           m_uVisibleLandmarksMinimum( 100 ),

                                                                           m_pTriangulator( std::make_shared< CTriangulator >( m_pCameraSTEREO, m_pExtractor ) ),
                                                                           m_cMatcher( m_pTriangulator, m_pDetector ),
                                                                           m_cOptimizer( m_pCameraSTEREO, m_vecLandmarks, m_vecKeyFrames, m_matTransformationWORLDtoLEFTLAST.inverse( ) ),
                                                                           //m_pMatcherLoopClosing( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) ), m_dMinimumRelativeMatchesLoopClosure( 0.5 ),
                                                                           //m_pMatcherLoopClosing( std::make_shared< cv::FlannBasedMatcher >( new cv::flann::LshIndexParams( 1, 20, 2 ) ) ), m_dMinimumRelativeMatchesLoopClosure( 0.4 ),
                                                                           //m_pMatcherLoopClosing( std::make_shared< cv::CBTreeMatcher< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( ) ), m_dMinimumRelativeMatchesLoopClosure( 0.45 ),

                                                                           m_pIMU( p_pIMUInterpolator ),

                                                                           m_eMode( p_eMode ),
                                                                           m_strVersionInfo( "CTrackerSVI [" + std::to_string( m_pCameraSTEREO->m_uPixelWidth )
                                                                                                             + "|" + std::to_string( m_pCameraSTEREO->m_uPixelHeight ) + "]" )
#if defined USING_BOW
                                                                          ,m_pBoWDatabase( std::make_shared< BriefDatabase >( BriefVocabulary( "vocabulary_BRIEF_75.yml.gz" ), true, 0 ) )
                                                                          ,m_pBoWVocabulary( std::make_shared< BriefVocabulary >( "vocabulary_BRIEF_75.yml.gz" ) )
#endif
{
    m_vecLandmarks->clear( );
    m_vecKeyFrames->clear( );
    m_vecRotations.clear( );

    //ds windows
    _initializeTranslationWindow( );

    //ds set opencv parallelization threads
    cv::setNumThreads( 1 );
    cv::setUseOptimized( false );

    //ds initialize reference frames with black images
    m_matDisplayLowerReference = cv::Mat::zeros( m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );

    //ds initialize the window
    cv::namedWindow( m_strVersionInfo, cv::WINDOW_AUTOSIZE );

    CLogger::openBox( );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) <OpenCV> available CPUs: %i\n", m_uFrameCount, cv::getNumberOfCPUs( ) );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) <OpenCV> available threads: %i\n", m_uFrameCount, cv::getNumThreads( ) );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) feature detector: %s\n", m_uFrameCount, m_pDetector->name( ).c_str( ) );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) descriptor extractor: %s\n", m_uFrameCount, m_pExtractor->name( ).c_str( ) );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) descriptor size: %i bytes\n", m_uFrameCount, m_pExtractor->descriptorSize( ) );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) <CIMUInterpolator> maximum timestamp delta: %f\n", m_uFrameCount, CIMUInterpolator::dMaximumDeltaTimeSeconds );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) <CIMUInterpolator> imprecision angular velocity: %f\n", m_uFrameCount, CIMUInterpolator::m_dImprecisionAngularVelocity );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) <CIMUInterpolator> imprecision linear acceleration: %f\n", m_uFrameCount, CIMUInterpolator::m_dImprecisionLinearAcceleration );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) <CIMUInterpolator> bias linear acceleration x/y/z: %4.2f/%4.2f/%4.2f\n", m_uFrameCount, CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[0],
                                                                                                                                                  CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[1],
                                                                                                                                                  CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[2] );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) <Landmark> cap iterations: %u\n", m_uFrameCount, CLandmark::uCapIterations );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) <Landmark> convergence delta: %f\n", m_uFrameCount, CLandmark::dConvergenceDelta );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) <Landmark> maximum error L2 inlier: %f\n", m_uFrameCount, CLandmark::dKernelMaximumErrorSquaredPixels );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) <Landmark> maximum error L2 average: %f\n", m_uFrameCount, CLandmark::dMaximumErrorSquaredAveragePixels );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) loop closing minimum relative matches: %f\n", m_uFrameCount, m_dMinimumRelativeMatchesLoopClosure );
    std::printf( "[0][%06lu]<CTrackerSVI>(CTrackerSVI) instance allocated\n", m_uFrameCount );
    CLogger::closeBox( );
}

CTrackerSVI::~CTrackerSVI( )
{
    /*ds close loggers
    CLogger::CLogLandmarkCreation::close( );
    CLogger::CLogLandmarkFinal::close( );
    CLogger::CLogLandmarkFinalOptimized::close( );
    CLogger::CLogTrajectory::close( );
    CLogger::CLogIMUInput::close( );*/

    //ds total data structure size
    uint64_t uSizeBytesLandmarks = 0;

    //ds free all landmarks
    for( const CLandmark* pLandmark: *m_vecLandmarks )
    {
        //ds write final state to file before deleting
        //CLogger::CLogLandmarkFinal::addEntry( pLandmark );

        //ds save optimized landmarks to separate file
        if( pLandmark->bIsOptimal && 1 < pLandmark->uNumberOfKeyFramePresences )
        {
            //CLogger::CLogLandmarkFinalOptimized::addEntry( pLandmark );
        }

        //ds accumulate size information
        uSizeBytesLandmarks += pLandmark->getSizeBytes( );

        assert( 0 != pLandmark );
        delete pLandmark;
    }
    std::printf( "[0][%06lu]<CTrackerSVI>(~CTrackerSVI) deallocated landmarks: %lu (%.0fMB)\n", m_uFrameCount, m_vecLandmarks->size( ), uSizeBytesLandmarks/1e6 );

    //ds total data structure size
    uint64_t uSizeBytesKeyFrames = 0;

    //ds free keyframes
    for( const CKeyFrame* pKeyFrame: *m_vecKeyFrames )
    {
        uSizeBytesKeyFrames += pKeyFrame->getSizeBytes( );
        assert( 0 != pKeyFrame );
        delete pKeyFrame;
    }
    std::printf( "[0][%06lu]<CTrackerSVI>(~CTrackerSVI) deallocated key frames: %lu (%.0fMB)\n", m_uFrameCount, m_vecKeyFrames->size( ), uSizeBytesKeyFrames/1e6 );
    std::printf( "[0][%06lu]<CTrackerSVI>(~CTrackerSVI) instance deallocated\n", m_uFrameCount );
}

void CTrackerSVI::process( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
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

    assert( 0.0 <= dDeltaTimestampSeconds );

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
        std::printf( "[0][%06lu]<CTrackerSVI>(receivevDataVI) using reduced IMU input, timestamp delta: %f\n", m_uFrameCount, dDeltaTimestampSeconds );

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

void CTrackerSVI::finalize( )
{

/*#ifdef USING_BOW

    //ds instantiate controller
    BriefVocabulary cVoc( 9, 3, DBoW2::BINARY, DBoW2::L1_NORM );

    //ds data structure containing key frames with descriptor info
    std::vector< std::vector< boost::dynamic_bitset< > > > vecDescriptorsKeyFrames;
    vecDescriptorsKeyFrames.reserve( m_vecKeyFrames->size( ) );

    //ds for each
    for( const CKeyFrame* pKeyFrame: *m_vecKeyFrames )
    {
        vecDescriptorsKeyFrames.push_back( pKeyFrame->vecDescriptorPool );
    }

    std::printf( "[0][%06lu]<CTrackerSVI>(finalize) creating vocabulary\n", m_uFrameCount );

    //ds create the vocabulary
    const double dTimeStartSeconds = CTimer::getTimeSeconds( );
    cVoc.create( vecDescriptorsKeyFrames );
    const double dDurationSeconds = CTimer::getTimeSeconds( )-dTimeStartSeconds;

    std::printf( "[0][%06lu]<CTrackerSVI>(finalize) creation complete - duration: %fs\n", m_uFrameCount, dDurationSeconds );

    //ds filename
    const std::string strVocabularyName( "vocabulary_BRIEF_"+ std::to_string( vecDescriptorsKeyFrames.size( ) ) +".yml.gz" );

    //ds save to disk
    cVoc.save( strVocabularyName );

    std::printf( "[0][%06lu]<CTrackerSVI>(finalize) saved DBoW2 vocabulary to: '%s'\n", m_uFrameCount, strVocabularyName.c_str( ) );

#endif*/

    //ds if tracker GUI is still open - otherwise run the optimization right away
    if( !m_bIsShutdownRequested )
    {
        //ds inform
        std::printf( "[0][%06lu]<CTrackerSVI>(finalize) press any key to perform final optimization\n", m_uFrameCount );

        //ds also display on image
        cv::Mat matDisplayComplete = cv::Mat( 2*m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3, cv::Scalar( 0.0 ) );
        cv::putText( matDisplayComplete, "DATASET COMPLETE - PRESS ANY KEY TO EXIT" , cv::Point2i( 50, 50 ), cv::FONT_HERSHEY_PLAIN, 2.0, CColorCodeBGR( 255, 255, 255 ) );
        cv::imshow( "vi_mapper [L|R]", matDisplayComplete );

        //ds wait for any user input
        cv::waitKey( 0 );
    }
    else
    {
        std::printf( "[0][%06lu]<CTrackerSVI>(finalize) running final optimization\n", m_uFrameCount );
    }

    //ds trigger shutdown
    m_bIsShutdownRequested = true;
}

void CTrackerSVI::_trackLandmarks( const cv::Mat& p_matImageLEFT,
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
        std::printf( "[0][%06lu]<CTrackerSVI>(_trackLandmarks) pose optimization failed [RAW PRIOR]: '%s'\n", m_uFrameCount, p_cException.what( ) );
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
            std::printf( "[0][%06lu]<CTrackerSVI>(_trackLandmarks) pose optimization failed [DAMPED PRIOR]: '%s' - running on damped IMU only\n", m_uFrameCount, p_cException.what( ) );

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

        std::printf( "[0][%06lu]<CTrackerSVI>(_trackLandmarks) lost track (landmarks visible: %3i lost: %3i), total delta: %f (%f %f %f), motion scaling: %f\n",
                     m_uFrameCount, uNumberOfVisibleLandmarks, static_cast< int32_t >( iLandmarksLost ), m_vecVelocityAngularFilteredLAST.squaredNorm( ), m_vecVelocityAngularFilteredLAST.x( ), m_vecVelocityAngularFilteredLAST.y( ), m_vecVelocityAngularFilteredLAST.z( ), dMotionScaling );
        //m_uWaitKeyTimeoutMS = 0;
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
    m_cMatcher.drawVisibleLandmarks( matDisplayLEFT, matDisplayRIGHT, matTransformationWORLDtoLEFT );

    //ds optimize landmarks periodically
    if( 0 == m_uFrameCount%m_uLandmarkOptimizationEveryNFrames )
    {
        //ds blocks
        m_cMatcher.optimizeActiveLandmarks( m_uFrameCount );
    }

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
        const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > vecCloud = m_cMatcher.getCloudForVisibleOptimizedLandmarks( m_uFrameCount );

        //ds if the number of points in the cloud is sufficient
        if( m_uMinimumLandmarksForKeyFrame < vecCloud->size( ) )
        {
            //ds register keyframing to matcher
            m_cMatcher.setKeyFrameToVisibleLandmarks( );

            //ds create key frame
            CKeyFrame* pKeyFrameNEW = new CKeyFrame( m_vecKeyFrames->size( ),
                                                  m_uFrameCount,
                                                  matTransformationLEFTtoWORLD,
                                                  p_vecLinearAcceleration.normalized( ),
                                                  m_cMatcher.getMeasurementsForVisibleLandmarks( ),
                                                  vecCloud,
                                                  m_uCountInstability,
                                                  dMotionScaling );

#if defined USING_BOW
            pKeyFrameNEW->setBoWVectors( m_pBoWDatabase );
#endif

            //ds set loop closures
            pKeyFrameNEW->vecLoopClosures = _getLoopClosuresForKeyFrame( pKeyFrameNEW, m_dLoopClosingRadiusSquaredMetersL2, m_dMinimumRelativeMatchesLoopClosure );

            //ds if we found closures
            if( 0 < pKeyFrameNEW->vecLoopClosures.size( ) )
            {
                //ds register closed key frame (ignore actual number of closures for this frame)
                ++m_uLoopClosingKeyFramesInQueue;
            }

            //ds add the new key frame to our stack
            m_vecKeyFrames->push_back( pKeyFrameNEW );

            //ds check if we are not in a critical situation before triggering an optimization
            if( m_dMaximumMotionScalingForOptimization > ( dMotionScaling+m_dMotionScalingLAST )/2.0 && 0 == m_uCountInstability )
            {
                //ds current key frame id
                const std::vector< CLandmark* >::size_type uIDKeyFrameCurrent = m_vecKeyFrames->back( )->uID;

                //ds check if optimization is required (based on key frame id or loop closing) TODO beautify this case
                if( m_uIDDeltaKeyFrameForOptimization < uIDKeyFrameCurrent-m_uIDOptimizedKeyFrameLAST                                                                              ||
                   ( m_uLoopClosingKeyFrameWaitingQueue < m_uLoopClosingKeyFramesInQueue && m_uIDDeltaKeyFrameForOptimization < uIDKeyFrameCurrent-m_uIDLoopClosureOptimizedLAST ) )
                {
                    //ds trigger optimization
                    m_cOptimizer.optimize( m_uFrameCount, m_uIDOptimizedKeyFrameLAST, m_uLoopClosingKeyFramesInQueue, m_vecTranslationToG2o );

                    //ds if the optimization contained loop closures
                    if( 0 < m_uLoopClosingKeyFramesInQueue )
                    {
                        m_uIDLoopClosureOptimizedLAST = uIDKeyFrameCurrent;
                    }

                    //ds update counters
                    m_uLoopClosingKeyFramesInQueue = 0;
                    m_uIDOptimizedKeyFrameLAST     = m_vecKeyFrames->back( )->uID+1;
                    assert( m_vecKeyFrames->back( )->bIsOptimized );

                    //ds integrate optimization
                    matTransformationLEFTtoWORLD = m_vecKeyFrames->back( )->matTransformationLEFTtoWORLD;
                    matTransformationWORLDtoLEFT = matTransformationLEFTtoWORLD.inverse( );
                    m_vecPositionCurrent         = matTransformationLEFTtoWORLD.translation( );
                }
            }

            //ds update references
            m_vecPositionKeyFrameLAST         = m_vecPositionCurrent;
            m_vecCameraOrientationAccumulated = Eigen::Vector3d::Zero( );
            m_uFrameKeyFrameLAST              = m_uFrameCount;
        }
    }

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
    cv::imshow( m_strVersionInfo, matDisplayComplete );

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
                    std::printf( "[0][%06lu]<CTrackerSVI>(_trackLandmarks) switched to stepwise mode\n", m_uFrameCount );
                }
                else
                {
                    //ds switch to benchmark mode
                    m_uWaitKeyTimeoutMS = 1;
                    m_eMode = ePlaybackBenchmark;
                    std::printf( "[0][%06lu]<CTrackerSVI>(_trackLandmarks) switched back to benchmark mode\n", m_uFrameCount );
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

    /*if( 75 == m_vecKeyFrames->size( ) )
    {
        finalize( );
    }*/
}

void CTrackerSVI::_addNewLandmarks( const cv::Mat& p_matImageLEFT,
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

            //ds landmark right
            const cv::Point2f ptLandmarkRIGHT( cMatchRIGHT.ptUVCAMERA );

            //ds allocate a new landmark and add the current position
            CLandmark* pLandmarkNEW = new CLandmark( m_uAvailableLandmarkID,
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
                                                     m_uFrameCount );

            //ds log creation
            //CLogger::CLogLandmarkCreation::addEntry( m_uFrameCount, pLandmark, dDepthMeters, ptLandmarkLEFT, ptLandmarkRIGHT );

            //ds add to newly detected
            vecNewLandmarks->push_back( pLandmarkNEW );

            //ds next landmark id
            ++m_uAvailableLandmarkID;

            //ds draw detected point
            //cv::line( p_matDisplaySTEREO, ptLandmarkLEFT, cv::Point2f( ptLandmarkLEFT.x+m_pCameraSTEREO->m_uPixelWidth, ptLandmarkLEFT.y ), CColorCodeBGR( 175, 175, 175 ) );
            cv::circle( p_matDisplaySTEREO, ptLandmarkLEFT, 2, CColorCodeBGR( 0, 255, 0 ), -1 );

            //ds draw acquisition information
            char chBuffer[10];
            std::snprintf( chBuffer, 10, "%lu|%.1f", pLandmarkNEW->uID, vecPointTriangulatedLEFT.z( ) );
            cv::putText( p_matDisplaySTEREO, chBuffer , cv::Point2d( ptLandmarkLEFT.x+pLandmarkNEW->dKeyPointSize, ptLandmarkLEFT.y+pLandmarkNEW->dKeyPointSize ), cv::FONT_HERSHEY_PLAIN, 0.5, CColorCodeBGR( 0, 0, 255 ) );

            //ds draw reprojection of triangulation
            cv::circle( p_matDisplaySTEREO, cv::Point2d( ptLandmarkRIGHT.x+m_pCameraSTEREO->m_uPixelWidth, ptLandmarkRIGHT.y ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );
        }
        catch( const CExceptionNoMatchFound& p_cException )
        {
            cv::circle( p_matDisplaySTEREO, ptLandmarkLEFT, 3, CColorCodeBGR( 0, 0, 255 ), -1 );
            //std::printf( "[0][%06lu]<CTrackerSVI>(_addNewLandmarks) could not find match for keypoint: '%s'\n", m_uFrameCount, p_cException.what( ) );
        }
    }

    //ds if we couldn't find new landmarks
    if( 0 == vecNewLandmarks->size( ) )
    {
        std::printf( "[0][%06lu]<CTrackerSVI>(_getNewLandmarks) unable to detect new landmarks\n", m_uFrameCount );
    }
    else
    {
        //ds all visible in this frame
        m_uNumberofVisibleLandmarksLAST += vecNewLandmarks->size( );

        {
            //ds add to permanent reference holder (this will copy the landmark references)
            m_vecLandmarks->insert( m_vecLandmarks->end( ), vecNewLandmarks->begin( ), vecNewLandmarks->end( ) );

            //ds add this measurement point to the epipolar matcher (which will remove references from its detection point -> does not affect the landmarks main vector)
            m_cMatcher.addDetectionPoint( p_matTransformationLEFTtoWORLD, vecNewLandmarks );
        }

        //std::printf( "<CTrackerSVI>(_getNewLandmarks) added new landmarks: %lu/%lu\n", vecNewLandmarks->size( ), vecKeyPoints.size( ) );
    }
}

//ds locked key frames from upper scope
const std::vector< const CKeyFrame::CMatchICP* > CTrackerSVI::_getLoopClosuresForKeyFrame( const CKeyFrame* p_pKeyFrameQUERY,
                                                                                           const double& p_dSearchRadiusMetersL2,
                                                                                           const double& p_dMinimumRelativeMatchesLoopClosure )
{
    //ds overall timing
    const double dTimeStartSeconds = CTimer::getTimeSeconds( );

    //ds potential closures list
    std::vector< std::map< UIDLandmark, std::list< CMatchCloud > > > vecPotentialClosures( m_vecKeyFrames->size( ) );

    //ds total matching duration for this query cloud
    double dDurationMatchingSeconds = 0.0;

#ifdef USING_BTREE

    const std::string strOutFileTiming( "logs/matching_time_closures_btree.txt" );
    const std::string strOutFileClosureMap( "logs/closure_map_btree.txt" );

    //ds query descriptors
    const std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > vecDescriptorPoolQUERY = p_pKeyFrameQUERY->vecDescriptorPool;

    //ds loop over all past key frames (EXTREMELY COSTLY
    for( const CKeyFrame* pKeyFrameREFERENCE: *m_vecKeyFrames )
    {
        //ds matches within the current reference
        std::vector< cv::DMatch > vecMatches( 0 );

        //ds match
        const double dTimeStartMatchingSeconds = CTimer::getTimeSeconds( );
        pKeyFrameREFERENCE->m_pBTree->match( vecDescriptorPoolQUERY, vecMatches );
        dDurationMatchingSeconds += CTimer::getTimeSeconds( )-dTimeStartMatchingSeconds;

        //ds evaluate all matches for this reference cloud
        for( const cv::DMatch& cMatch: vecMatches )
        {
            //ds if distance is acceptable (set for BTree, THIS IS REDUNDANT)
            if( MAXIMUM_DISTANCE_HAMMING == cMatch.distance )
            {
                //ds buffer query point
                const CDescriptorVectorPoint3DWORLD* pPointQUERY = p_pKeyFrameQUERY->mapDescriptorToPoint.at( cMatch.queryIdx );

                try
                {
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].at( pPointQUERY->uID ).push_back( CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) );
                }
                catch( const std::out_of_range& p_cException )
                {
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].insert( std::make_pair( pPointQUERY->uID, std::list< CMatchCloud >( 1, CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) ) ) );
                }
            }
        }
    }

#elif defined USING_BF

    const std::string strOutFileTiming( "logs/matching_time_closures_bf.txt" );
    const std::string strOutFileClosureMap( "logs/closure_map_bf.txt" );

    //ds query descriptors
    const CDescriptors vecDescriptorPoolQUERY = p_pKeyFrameQUERY->vecDescriptorPool;

    //ds loop over all past key frames (EXTREMELY COSTLY
    for( const CKeyFrame* pKeyFrameREFERENCE: *m_vecKeyFrames )
    {
        //ds matches within the current reference
        std::vector< cv::DMatch > vecMatches( 0 );

        //ds match
        const double dTimeStartMatchingSeconds = CTimer::getTimeSeconds( );
        pKeyFrameREFERENCE->m_pMatcherBF->match( vecDescriptorPoolQUERY, vecMatches );
        dDurationMatchingSeconds += CTimer::getTimeSeconds( )-dTimeStartMatchingSeconds;

        //ds evaluate all matches for this reference cloud
        for( const cv::DMatch& cMatch: vecMatches )
        {
            //ds if distance is acceptable
            if( MAXIMUM_DISTANCE_HAMMING > cMatch.distance )
            {
                //ds buffer query point
                const CDescriptorVectorPoint3DWORLD* pPointQUERY = p_pKeyFrameQUERY->mapDescriptorToPoint.at( cMatch.queryIdx );

                try
                {
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].at( pPointQUERY->uID ).push_back( CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) );
                }
                catch( const std::out_of_range& p_cException )
                {
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].insert( std::make_pair( pPointQUERY->uID, std::list< CMatchCloud >( 1, CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) ) ) );
                }
            }
        }
    }

#elif defined USING_LSH

    const std::string strOutFileTiming( "logs/matching_time_closures_lsh.txt" );
    const std::string strOutFileClosureMap( "logs/closure_map_lsh.txt" );

    //ds query descriptors
    const CDescriptors vecDescriptorPoolQUERY = p_pKeyFrameQUERY->vecDescriptorPool;

    //ds loop over all past key frames (EXTREMELY COSTLY
    for( const CKeyFrame* pKeyFrameREFERENCE: *m_vecKeyFrames )
    {
        //ds matches within the current reference
        std::vector< cv::DMatch > vecMatches( 0 );

        //ds match
        const double dTimeStartMatchingSeconds = CTimer::getTimeSeconds( );
        pKeyFrameREFERENCE->m_pMatcherLSH->match( vecDescriptorPoolQUERY, vecMatches );
        dDurationMatchingSeconds += CTimer::getTimeSeconds( )-dTimeStartMatchingSeconds;

        //ds evaluate all matches for this reference cloud
        for( const cv::DMatch& cMatch: vecMatches )
        {
            //ds if distance is acceptable
            if( MAXIMUM_DISTANCE_HAMMING > cMatch.distance )
            {
                //ds buffer query point
                const CDescriptorVectorPoint3DWORLD* pPointQUERY = p_pKeyFrameQUERY->mapDescriptorToPoint.at( cMatch.queryIdx );

                try
                {
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].at( pPointQUERY->uID ).push_back( CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) );
                }
                catch( const std::out_of_range& p_cException )
                {
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].insert( std::make_pair( pPointQUERY->uID, std::list< CMatchCloud >( 1, CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) ) ) );
                }
            }
        }
    }

#elif defined USING_BOW

    const std::string strOutFileTiming( "logs/matching_time_closures_dbow2.txt" );
    const std::string strOutFileClosureMap( "logs/closure_map_dbow2.txt" );

    //ds query descriptors
    const DBoW2::FeatureVector& vecDescriptorPoolFQUERY = p_pKeyFrameQUERY->vecDescriptorPoolF;

    //ds results
    DBoW2::QueryResults vecResultsQUERY;

    //ds get the query results
    const double dTimeStartMatchingSeconds = CTimer::getTimeSeconds( );
    m_pBoWDatabase->query( p_pKeyFrameQUERY->vecDescriptorPoolB, vecResultsQUERY, m_vecKeyFrames->size( ), m_vecKeyFrames->size( ) );

    //ds get actual matches from results
    for( const DBoW2::Result& cResult: vecResultsQUERY )
    {
        const DBoW2::FeatureVector& vecDescriptorPoolFREFERENCE = m_pBoWDatabase->retrieveFeatures( cResult.Id );

        std::cerr << vecDescriptorPoolFQUERY.size( ) << std::endl;
        std::cerr << vecDescriptorPoolFREFERENCE.size( ) << std::endl;

        std::vector<unsigned int> i_old, i_cur;

        DBoW2::FeatureVector::const_iterator old_it, cur_it;
        const DBoW2::FeatureVector::const_iterator old_end = vecDescriptorPoolFREFERENCE.end( );
        const DBoW2::FeatureVector::const_iterator cur_end = vecDescriptorPoolFQUERY.end( );

        old_it = vecDescriptorPoolFREFERENCE.begin();
        cur_it = vecDescriptorPoolFQUERY.begin( );

        while(old_it != old_end && cur_it != cur_end)
        {
        if(old_it->first == cur_it->first)
        {
        // compute matches between
        // features old_it->second of m_image_keys[old_entry] and
        // features cur_it->second of keys
            std::vector<unsigned int> i_old_now, i_cur_now;

        _getMatches_neighratio( m_vecKeyFrames->at( cResult.Id )->vecDescriptorPool, old_it->second, p_pKeyFrameQUERY->vecDescriptorPool, cur_it->second, i_old_now, i_cur_now);

        i_old.insert(i_old.end(), i_old_now.begin(), i_old_now.end());
        i_cur.insert(i_cur.end(), i_cur_now.begin(), i_cur_now.end());

        // move old_it and cur_it forward
        ++old_it;
        ++cur_it;
        }
        else if(old_it->first < cur_it->first)
        {
        // move old_it forward
        old_it = vecDescriptorPoolFREFERENCE.lower_bound(cur_it->first);
        // old_it = (first element >= cur_it.id)
        }
        else
        {
        // move cur_it forward
        cur_it = vecDescriptorPoolFQUERY.lower_bound(old_it->first);
        // cur_it = (first element >= old_it.id)
        }
        }

        std::cerr << "reference: " << i_old.size( ) << "/" << m_vecKeyFrames->at( cResult.Id )->vecDescriptorPool.size( ) << std::endl;
        std::cerr << "query: " << i_cur.size( ) << "/" << p_pKeyFrameQUERY->vecDescriptorPool.size( ) << std::endl;

    }
    dDurationMatchingSeconds += CTimer::getTimeSeconds( )-dTimeStartMatchingSeconds;

        /*ds evaluate all matches for this reference cloud
        for( const cv::DMatch& cMatch: vecMatches )
        {
            //ds if distance is acceptable
            if( MAXIMUM_DISTANCE_HAMMING > cMatch.distance )
            {
                //ds buffer query point
                const CDescriptorVectorPoint3DWORLD* pPointQUERY = p_pKeyFrameQUERY->mapDescriptorToPoint.at( cMatch.queryIdx );

                try
                {
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].at( pPointQUERY->uID ).push_back( CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) );
                }
                catch( const std::out_of_range& p_cException )
                {
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].insert( std::make_pair( pPointQUERY->uID, std::list< CMatchCloud >( 1, CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) ) ) );
                }
            }
        }*/

#endif

    //ds validate design
    assert( p_pKeyFrameQUERY->uID == vecPotentialClosures.size( ) );

    //ds current count
    uint64_t uNumberOfClosures = 0;

    //ds update stats matrix (adding the new key frame index)
    Eigen::MatrixXd matClosureMapNew( m_matClosureMap.rows( )+1, m_matClosureMap.cols( )+1 );
    matClosureMapNew.setZero( );
    matClosureMapNew.block( 0, 0, m_matClosureMap.rows( ), m_matClosureMap.cols( ) ) = m_matClosureMap;
    matClosureMapNew( m_vecKeyFrames->size( ), m_vecKeyFrames->size( ) ) = 1.0;

    double dRelativeMatchesBest = 0.0;

    //ds check all keyframes for distance
    for( int64_t uIDREFERENCE = 0; uIDREFERENCE < static_cast< int64_t >( vecPotentialClosures.size( )-m_uMinimumLoopClosingKeyFrameDistance ); ++uIDREFERENCE )
    {
        assert( 0 <= uIDREFERENCE );

        //ds compute relative matches
        const double dRelativeMatches = static_cast< double >( vecPotentialClosures[uIDREFERENCE].size( ) )/p_pKeyFrameQUERY->vecCloud->size( );

        if( dRelativeMatchesBest < dRelativeMatches )
        {
            dRelativeMatchesBest = dRelativeMatches;
        }

        //ds if we have a suffient amount of matches
        if( p_dMinimumRelativeMatchesLoopClosure < dRelativeMatches )
        {
            matClosureMapNew( m_vecKeyFrames->size( ), uIDREFERENCE ) = 1.0;
            matClosureMapNew( uIDREFERENCE, m_vecKeyFrames->size( ) ) = 1.0;

            ++uNumberOfClosures;
            std::printf( "[0][%06lu]<CTrackerSVI>(_getLoopClosuresForKeyFrame) found closure: [%06lu] > [%06lu] relative matches: %f (%lu/%lu)\n",
                         m_uFrameCount, p_pKeyFrameQUERY->uID, uIDREFERENCE, dRelativeMatches, vecPotentialClosures[uIDREFERENCE].size( ), p_pKeyFrameQUERY->vecCloud->size( ) );

            /*ds spatial matches for ICP loop closure computation
            std::vector< CMatchCloud > vecMatchesICP;
            vecMatchesICP.reserve( vecPotentialClosures[uIDREFERENCE].size( ) );

            //ds filter actual spatial matches
            for( std::pair< UIDLandmark, std::list< CMatchCloud > > prMatch: vecPotentialClosures[uIDREFERENCE] )
            {
                vecMatchesICP.push_back( _getMatchNN( prMatch.second ) );
            }*/
        }
    }

    //ds log stats
    std::ofstream ofLogfileTiming( strOutFileTiming, std::ofstream::out | std::ofstream::app );
    ofLogfileTiming << p_pKeyFrameQUERY->uID << " " << dDurationMatchingSeconds << " " << uNumberOfClosures << " " << dRelativeMatchesBest << "\n";
    ofLogfileTiming.close( );

    //ds update stats
    m_matClosureMap.swap( matClosureMapNew );

    //ds write stats to file (every time)
    std::ofstream ofLogfilePPL( strOutFileClosureMap, std::ofstream::out );

    //ds loop over eigen matrix and dump the values
    for( int64_t u = 0; u < m_matClosureMap.rows( ); ++u )
    {
        for( int64_t v = 0; v < m_matClosureMap.cols( ); ++v )
        {
            ofLogfilePPL << m_matClosureMap( u, v ) << " ";
        }

        ofLogfilePPL << "\n";
    }

    //ds save file
    ofLogfilePPL.close( );

    /*ds potential keyframes for loop closing
    std::vector< const CKeyFrame* > vecPotentialClosureKeyFrames;

    //ds check all keyframes for distance
    for( const CKeyFrame* pKeyFrameReference: *m_vecKeyFrames )
    {
        //ds break if near to current id (forward closing)
        if( m_uMinimumLoopClosingKeyFrameDistance > uIDQuery-pKeyFrameReference->uID )
        {
            break;
        }

        //ds if the distance is acceptable
        if( p_dSearchRadiusMetersL2 > ( pKeyFrameReference->matTransformationLEFTtoWORLD.translation( )-vecPositionQuery ).squaredNorm( ) )
        {
            //ds add the keyframe to the loop closing pool
            vecPotentialClosureKeyFrames.push_back( pKeyFrameReference );
        }
    }

    //ds closure vector to align
    std::vector< std::pair< const CKeyFrame*, const std::vector< CMatchCloud > > > vecClosuresToCompute( 0 );

    std::printf( "[0][%06lu]<CTrackerSVI>(_getLoopClosuresForKeyFrame) checking for closure in potential keyframes: %lu\n", m_uFrameCount, vecPotentialClosureKeyFrames.size( ) );

    //ds timing
    const double dTimeStartMatchingSeconds = CTimer::getTimeSeconds( );

    //ds compare current cloud against previous ones to enable loop closure (skipping the keyframes added just before)
    for( const CKeyFrame* pKeyFrameReference: vecPotentialClosureKeyFrames )
    {
        //ds get matches
        std::vector< cv::DMatch > vecMatchesCV( 0 );
        m_pMatcherLoopClosing->match( vecDescriptorPoolCVQUERY, vecMatchesCV );
        //m_pMatcherLoopClosing->match( vecDescriptorPoolCVQUERY, pKeyFrameReference->vecDescriptorPoolCV, vecMatchesCV );

        //ds custom matches
        std::vector< CMatchCloud > vecMatches;

        //ds filter real matches
        for( const cv::DMatch cMatch: vecMatchesCV )
        {
            //ds if acceptable (a requirement if done with BTree)
            assert( MAXIMUM_DISTANCE_HAMMING == cMatch.distance );
            //if( MAXIMUM_DISTANCE_HAMMING > cMatch.distance )
            //{
            //    try
            //    {
                    //ds match points
                    vecMatches.push_back( CMatchCloud( p_pKeyFrameQUERY->mapDescriptorToPoint.at( cMatch.queryIdx ), pKeyFrameReference->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) );
            //    }
            //    catch( const std::out_of_range& p_cException )
            //    {
            //        std::printf( "[0][%06lu]<CTrackerSVI>(_getLoopClosuresForKeyFrame) unable to match points over descriptor-to-point map\n", m_uFrameCount );
            //    }
            //}
        }

        //ds compute relative matches
        const double dRelativeMatches = static_cast< double >( vecMatches.size( ) )/vecDescriptorPoolQUERY.size( );

        //ds if we have a suffient amount of matches
        if( p_dMinimumRelativeMatchesLoopClosure < dRelativeMatches )
        {
            std::printf( "[0][%06lu]<CTrackerSVI>(_getLoopClosuresForKeyFrame) found closure: [%06lu] > [%06lu] relative matches: %f (%lu%lu)\n",
                         m_uFrameCount, p_pKeyFrameQUERY->uID, pKeyFrameReference->uID, dRelativeMatches, p_pKeyFrameQUERY->vecDescriptorPool.size( ), pKeyFrameReference->vecDescriptorPool.size( ) );
            vecClosuresToCompute.push_back( std::make_pair( pKeyFrameReference, vecMatches ) );
        }
    }*/

    //ds solution vector
    std::vector< const CKeyFrame::CMatchICP* > vecLoopClosures( 0 );

            /*ds transformation between this keyframe and the loop closure one (take current measurement as prior)
            Eigen::Isometry3d matTransformationToClosure( pKeyFrameReference->matTransformationLEFTtoWORLD.inverse( )*matTransformationLEFTtoWORLDQuery );
            matTransformationToClosure.translation( ) = Eigen::Vector3d::Zero( );
            //const Eigen::Isometry3d matTransformationToClosureInitial( matTransformationToClosure );

            //ds 1mm for convergence
            const double dErrorDeltaForConvergence      = 1e-5;
            double dErrorSquaredTotalPrevious           = 0.0;
            const double dMaximumErrorForInlier         = 0.25; //0.25
            const double dMaximumErrorAverageForClosure = 0.2; //0.1

            //ds LS setup
            Eigen::Matrix< double, 6, 6 > matH;
            Eigen::Matrix< double, 6, 1 > vecB;
            Eigen::Matrix3d matOmega( Eigen::Matrix3d::Identity( ) );

            //std::printf( "<CTrackerStereo>(_getLoopClosureKeyFrameFCFS) t: %4.1f %4.1f %4.1f > ", matTransformationToClosure.translation( ).x( ), matTransformationToClosure.translation( ).y( ), matTransformationToClosure.translation( ).z( ) );

            //ds run least-squares maximum 100 times
            for( uint8_t uLS = 0; uLS < 100; ++uLS )
            {
                //ds error
                double dErrorSquaredTotal = 0.0;
                uint32_t uInliers         = 0;

                //ds initialize setup
                matH.setZero( );
                vecB.setZero( );

                //ds for all the points
                for( const CMatchCloud& cMatch: *vecMatches )
                {
                    //ds compute projection into closure
                    const CPoint3DCAMERA vecPointXYZQuery( matTransformationToClosure*cMatch.cPointQuery.vecPointXYZCAMERA );
                    if( 0.0 < vecPointXYZQuery.z( ) )
                    {
                        assert( 0.0 < cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds adjust omega to inverse depth value (the further away the point, the less weight)
                        matOmega(2,2) = 1.0/( cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds compute error
                        const Eigen::Vector3d vecError( vecPointXYZQuery-cMatch.cPointReference.vecPointXYZCAMERA );

                        //ds update chi
                        const double dErrorSquared = vecError.transpose( )*matOmega*vecError;

                        //ds check if outlier
                        double dWeight = 1.0;
                        if( dMaximumErrorForInlier < dErrorSquared )
                        {
                            dWeight = dMaximumErrorForInlier/dErrorSquared;
                        }
                        else
                        {
                            ++uInliers;
                        }
                        dErrorSquaredTotal += dWeight*dErrorSquared;

                        //ds get the jacobian of the transform part = [I 2*skew(T*modelPoint)]
                        Eigen::Matrix< double, 3, 6 > matJacobianTransform;
                        matJacobianTransform.setZero( );
                        matJacobianTransform.block<3,3>(0,0).setIdentity( );
                        matJacobianTransform.block<3,3>(0,3) = -2*CMiniVisionToolbox::getSkew( vecPointXYZQuery );

                        //ds precompute transposed
                        const Eigen::Matrix< double, 6, 3 > matJacobianTransformTransposed( matJacobianTransform.transpose( ) );

                        //ds accumulate
                        matH += dWeight*matJacobianTransformTransposed*matOmega*matJacobianTransform;
                        vecB += dWeight*matJacobianTransformTransposed*matOmega*vecError;
                    }
                }

                //ds solve the system and update the estimate
                matTransformationToClosure = CMiniVisionToolbox::getTransformationFromVector( matH.ldlt( ).solve( -vecB ) )*matTransformationToClosure;

                //ds enforce rotation symmetry
                const Eigen::Matrix3d matRotation        = matTransformationToClosure.linear( );
                Eigen::Matrix3d matRotationSquared       = matRotation.transpose( )*matRotation;
                matRotationSquared.diagonal( ).array( ) -= 1;
                matTransformationToClosure.linear( )    -= 0.5*matRotation*matRotationSquared;

                //ds check if converged (no descent required)
                if( dErrorDeltaForConvergence > std::fabs( dErrorSquaredTotalPrevious-dErrorSquaredTotal ) )
                {
                    //ds compute average error
                    const double dErrorAverage = dErrorSquaredTotal/vecMatches->size( );

                    //ds if the solution is acceptable
                    if( dMaximumErrorAverageForClosure > dErrorAverage && p_uMinimumNumberOfMatchesLoopClosure < uInliers )
                    {
                        //std::printf( "<CMapper>(_getLoopClosuresForKeyFrame) found closure: [%06lu] > [%06lu] (matches: %3lu, iterations: %2u, average error: %5.3f, inliers: %2u)\n",
                        //             uIDQuery, pKeyFrameReference->uID, vecMatches->size( ), uLS, dErrorAverage, uInliers );
                        vecLoopClosures.push_back( new CKeyFrame::CMatchICP( pKeyFrameReference, matTransformationToClosure, vecMatches ) );
                        break;
                    }
                    else
                    {
                        //ds keep looping through keyframes
                        //std::printf( "%4.1f %4.1f %4.1f | ", matTransformationToClosure.translation( ).x( ), matTransformationToClosure.translation( ).y( ), matTransformationToClosure.translation( ).z( ) );
                        //std::printf( "system converged in %2u iterations, average error: %5.3f (inliers: %2u) - discarded\n", uLS, dErrorAverage, uInliers );
                        break;
                    }
                }
                else
                {
                    dErrorSquaredTotalPrevious = dErrorSquaredTotal;
                }

                //ds if not converged
                if( 99 == uLS )
                {
                    //std::printf( "system did not converge\n" );
                }
            }*/

    //ds update closure matcher with current key frame
    //m_pMatcherLoopClosing->add( std::vector< cv::Mat >( 1, vecDescriptorPoolCVQUERY ) );

#if defined USING_BOW
    m_pBoWDatabase->add( p_pKeyFrameQUERY->vecDescriptorPoolB, vecDescriptorPoolFQUERY );
#endif

    //ds info
    m_dDurationTotalSecondsLoopClosing += CTimer::getTimeSeconds( )-dTimeStartSeconds;

    //ds return found closures
    return vecLoopClosures;
}

const CMatchCloud CTrackerSVI::_getMatchNN( std::list< CMatchCloud >& p_lMatches ) const
{
    //ds sort the list
    p_lMatches.sort( []( const CMatchCloud& prLHS, const CMatchCloud& prRHS )
                     {
                        assert( prLHS.pPointQuery->uID == prRHS.pPointQuery->uID );
                        return prLHS.pPointReference->uID > prRHS.pPointReference->uID;
                     } );

    return p_lMatches.front( );
}

void CTrackerSVI::_initializeTranslationWindow( )
{
    //ds reinitialize translation window
    m_vecTranslationDeltas.clear( );
    for( std::vector< Eigen::Vector3d >::size_type u = 0; u < m_uIMULogbackSize; ++u )
    {
        m_vecTranslationDeltas.push_back( Eigen::Vector3d::Zero( ) );
    }
    m_vecGradientXYZ = Eigen::Vector3d::Zero( );
}

void CTrackerSVI::_shutDown( )
{
    m_bIsShutdownRequested = true;
    std::printf( "[0][%06lu]<CTrackerSVI>(_shutDown) termination requested, <CTrackerSVI> disabled\n", m_uFrameCount );
}

void CTrackerSVI::_updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange )
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

void CTrackerSVI::_drawInfoBox( cv::Mat& p_matDisplay, const double& p_dMotionScaling ) const
{
    char chBuffer[1024];

    switch( m_eMode )
    {
        case ePlaybackStepwise:
        {
            std::snprintf( chBuffer, 1024, "[%13.2f|%05lu] STEPWISE | X: %5.1f Y: %5.1f Z: %5.1f DELTA L2: %4.2f MOTION: %4.2f | LANDMARKS V: %3i (%3lu,%3lu,%3lu,%3lu) F: %4lu I: %4lu T: %4lu | DETECTIONS: %2lu(%3lu) | KFs: %lu | g2o: %lu",
                           m_dTimestampLASTSeconds, m_uFrameCount,
                           m_vecPositionCurrent.x( ), m_vecPositionCurrent.y( ), m_vecPositionCurrent.z( ), m_dTranslationDeltaSquaredNormCurrent, p_dMotionScaling,
                           m_uNumberofVisibleLandmarksLAST, m_cMatcher.getNumberOfTracksStage1( ), m_cMatcher.getNumberOfTracksStage2_1( ), m_cMatcher.getNumberOfTracksStage3( ), m_cMatcher.getNumberOfTracksStage2_2( ), m_cMatcher.getNumberOfFailedLandmarkOptimizations( ), m_cMatcher.getNumberOfInvalidLandmarksTotal( ), m_uAvailableLandmarkID,
                           m_cMatcher.getNumberOfDetectionPointsActive( ), m_cMatcher.getNumberOfDetectionPointsTotal( ),
                           m_vecKeyFrames->size( ),
                           m_cOptimizer.getNumberOfOptimizations( ) );
            break;
        }
        case ePlaybackBenchmark:
        {
            std::snprintf( chBuffer, 1024, "[%13.2f|%05lu] BENCHMARK FPS: %4.1f | X: %5.1f Y: %5.1f Z: %5.1f DELTA L2: %4.2f SCALING: %4.2f | LANDMARKS V: %3i (%3lu,%3lu,%3lu,%3lu) F: %4lu I: %4lu T: %4lu | DETECTIONS: %2lu(%3lu) | KFs: %lu | g2o: %lu",
                           m_dTimestampLASTSeconds, m_uFrameCount, m_dPreviousFrameRate,
                           m_vecPositionCurrent.x( ), m_vecPositionCurrent.y( ), m_vecPositionCurrent.z( ), m_dTranslationDeltaSquaredNormCurrent, p_dMotionScaling,
                           m_uNumberofVisibleLandmarksLAST, m_cMatcher.getNumberOfTracksStage1( ), m_cMatcher.getNumberOfTracksStage2_1( ), m_cMatcher.getNumberOfTracksStage3( ), m_cMatcher.getNumberOfTracksStage2_2( ), m_cMatcher.getNumberOfFailedLandmarkOptimizations( ), m_cMatcher.getNumberOfInvalidLandmarksTotal( ), m_uAvailableLandmarkID,
                           m_cMatcher.getNumberOfDetectionPointsActive( ), m_cMatcher.getNumberOfDetectionPointsTotal( ),
                           m_vecKeyFrames->size( ),
                           m_cOptimizer.getNumberOfOptimizations( ) );
            break;
        }
        default:
        {
            std::printf( "[0][%06lu]<CTrackerSVI>(_drawInfoBox) unsupported playback mode, no info box displayed\n", m_uFrameCount );
            break;
        }
    }


    p_matDisplay( cv::Rect( 0, 0, 2*m_pCameraSTEREO->m_uPixelWidth, 17 ) ).setTo( CColorCodeBGR( 0, 0, 0 ) );
    cv::putText( p_matDisplay, chBuffer , cv::Point2i( 2, 12 ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );
}
