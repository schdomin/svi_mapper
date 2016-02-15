#include "CTrackerGT.h"

#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>

#include "../gui/CConfigurationOpenCV.h"
#include "../exceptions/CExceptionPoseOptimization.h"
#include "../exceptions/CExceptionNoMatchFound.h"
#include "utility/CIMUInterpolator.h"

//#define LOOPCLOSING_BOW_BOW
//#define LOOPCLOSING_BOW_BTREE



CTrackerGT::CTrackerGT( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                                const EPlaybackMode& p_eMode,
                                const uint32_t& p_uWaitKeyTimeoutMS ): m_uWaitKeyTimeoutMS( p_uWaitKeyTimeoutMS ),
                                                                           m_pCameraLEFT( p_pCameraSTEREO->m_pCameraLEFT ),
                                                                           m_pCameraRIGHT( p_pCameraSTEREO->m_pCameraRIGHT ),
                                                                           m_pCameraSTEREO( p_pCameraSTEREO ),

                                                                           m_vecLandmarks( std::make_shared< std::vector< CLandmark* > >( ) ),
                                                                           m_vecKeyFrames( std::make_shared< std::vector< CKeyFrame* > >( ) ),

                                                                           m_matTransformationWORLDtoLEFTLAST( CIMUInterpolator::getTransformationInitialWORLDtoLEFT( ) ),
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

                                                                           m_eMode( p_eMode ),
                                                                           m_strVersionInfo( "CTrackerGT [" + std::to_string( m_pCameraSTEREO->m_uPixelWidth )
                                                                                                             + "|" + std::to_string( m_pCameraSTEREO->m_uPixelHeight ) + "]" )
#if defined USING_BOW
#define DBOW2_ID_LEVELS 2
                                                                          ,m_pBoWDatabase( std::make_shared< BriefDatabase >( BriefVocabulary( "brief_k10L6.voc.gz" ), true, DBOW2_ID_LEVELS ) )
#endif
#if defined USING_BTREE_INDEXED
                                                                          ,m_pBTree( std::make_shared< CBITree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( ) )
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
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) <OpenCV> available CPUs: %i\n", m_uFrameCount, cv::getNumberOfCPUs( ) );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) <OpenCV> available threads: %i\n", m_uFrameCount, cv::getNumThreads( ) );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) feature detector: %s\n", m_uFrameCount, m_pDetector->name( ).c_str( ) );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) descriptor extractor: %s\n", m_uFrameCount, m_pExtractor->name( ).c_str( ) );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) descriptor size: %i bytes\n", m_uFrameCount, m_pExtractor->descriptorSize( ) );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) <CIMUInterpolator> maximum timestamp delta: %f\n", m_uFrameCount, CIMUInterpolator::dMaximumDeltaTimeSeconds );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) <CIMUInterpolator> imprecision angular velocity: %f\n", m_uFrameCount, CIMUInterpolator::m_dImprecisionAngularVelocity );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) <CIMUInterpolator> imprecision linear acceleration: %f\n", m_uFrameCount, CIMUInterpolator::m_dImprecisionLinearAcceleration );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) <CIMUInterpolator> bias linear acceleration x/y/z: %4.2f/%4.2f/%4.2f\n", m_uFrameCount, CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[0],
                                                                                                                                                  CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[1],
                                                                                                                                                  CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[2] );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) <Landmark> cap iterations: %u\n", m_uFrameCount, CLandmark::uCapIterations );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) <Landmark> convergence delta: %f\n", m_uFrameCount, CLandmark::dConvergenceDelta );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) <Landmark> maximum error L2 inlier: %f\n", m_uFrameCount, CLandmark::dKernelMaximumErrorSquaredPixels );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) <Landmark> maximum error L2 average: %f\n", m_uFrameCount, CLandmark::dMaximumErrorSquaredAveragePixels );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) loop closing minimum relative matches: %f\n", m_uFrameCount, m_dMinimumRelativeMatchesLoopClosure );
    std::printf( "[0][%06lu]<CTrackerGT>(CTrackerGT) instance allocated\n", m_uFrameCount );
    CLogger::closeBox( );
}

CTrackerGT::~CTrackerGT( )
{
    //ds total data structure size
    uint64_t uSizeBytesLandmarks = 0;

    //ds free all landmarks
    for( const CLandmark* pLandmark: *m_vecLandmarks )
    {
        //ds accumulate size information
        uSizeBytesLandmarks += pLandmark->getSizeBytes( );

        assert( 0 != pLandmark );
        delete pLandmark;
    }
    std::printf( "[0][%06lu]<CTrackerGT>(~CTrackerGT) deallocated landmarks: %lu (%.0fMB)\n", m_uFrameCount, m_vecLandmarks->size( ), uSizeBytesLandmarks/1e6 );

    //ds total data structure size
    uint64_t uSizeBytesKeyFrames = 0;

    //ds free keyframes
    for( const CKeyFrame* pKeyFrame: *m_vecKeyFrames )
    {
        uSizeBytesKeyFrames += pKeyFrame->getSizeBytes( );
        assert( 0 != pKeyFrame );
        delete pKeyFrame;
    }
    std::printf( "[0][%06lu]<CTrackerGT>(~CTrackerGT) deallocated key frames: %lu (%.0fMB)\n", m_uFrameCount, m_vecKeyFrames->size( ), uSizeBytesKeyFrames/1e6 );
    std::printf( "[0][%06lu]<CTrackerGT>(~CTrackerGT) instance deallocated\n", m_uFrameCount );
}

void CTrackerGT::process( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
                          const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageRIGHT,
                          const Eigen::Isometry3d& p_matTransformationLEFTLASTtoLEFTNOW )
{
    //ds preprocessed images
    cv::Mat matPreprocessedLEFT( p_pImageLEFT->image( ) );
    cv::Mat matPreprocessedRIGHT( p_pImageRIGHT->image( ) );

    //ds current timestamp
    const double dTimestampSeconds      = p_pImageLEFT->timestamp( );
    const double dDeltaTimestampSeconds = dTimestampSeconds - m_dTimestampLASTSeconds;

    assert( 0.0 <= dDeltaTimestampSeconds );
    if( CIMUInterpolator::dMaximumDeltaTimeSeconds < dDeltaTimestampSeconds )
    {
        std::printf( "[0][%06lu]<CTrackerGT>(process) received large timestamp delta: %fs\n", m_uFrameCount, dDeltaTimestampSeconds );
    }

    //ds update change
    m_matTransformationLEFTLASTtoLEFTNOW = p_matTransformationLEFTLASTtoLEFTNOW;

    //ds changes from last (based on constant velocity motion model)
    const Eigen::Vector3d vecRotationTotal( CMiniVisionToolbox::toOrientationRodrigues( m_matTransformationLEFTLASTtoLEFTNOW.linear( ) ) );
    const Eigen::Vector3d vecTranslationTotal( m_matTransformationLEFTLASTtoLEFTNOW.translation( ) );

    //ds process images (fed with IMU prior pose)
    _trackLandmarks( matPreprocessedLEFT,
                     matPreprocessedRIGHT,
                     m_matTransformationLEFTLASTtoLEFTNOW*m_matTransformationWORLDtoLEFTLAST,
                     vecRotationTotal,
                     vecTranslationTotal,
                     dDeltaTimestampSeconds );

    //ds update timestamp
    m_dTimestampLASTSeconds = dTimestampSeconds;
}

void CTrackerGT::finalize( )
{
    //ds nothing to do
    std::printf( "[0][%06lu]<CTrackerGT>(finalize) terminating tracker\n", m_uFrameCount );

    //ds trigger shutdown
    m_bIsShutdownRequested = true;
}

void CTrackerGT::_trackLandmarks( const cv::Mat& p_matImageLEFT,
                                                 const cv::Mat& p_matImageRIGHT,
                                                 const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
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
    const double dMotionScaling = std::min( 1.0+( 10.0*p_vecRotationTotal.norm( )+0.5*p_vecTranslationTotal.norm( ) ), 5.0 );

    //ds refresh landmark states
    m_cMatcher.resetVisibilityActiveLandmarks( );

    //ds buffer poses
    const Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationWORLDtoLEFT );
    const Eigen::Isometry3d matTransformationLEFTtoWORLD( matTransformationWORLDtoLEFT.inverse( ) );

    //ds get current measurements (including landmarks already detected in the pose optimization)
    m_cMatcher.trackManual( m_uFrameCount,
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
        std::printf( "[0][%06lu]<CTrackerGT>(_trackLandmarks) lost track (landmarks visible: %3i lost: %3i), total delta: %f (%f %f %f), motion scaling: %f\n",
                     m_uFrameCount, uNumberOfVisibleLandmarks, static_cast< int32_t >( iLandmarksLost ), m_matTransformationLEFTLASTtoLEFTNOW.translation( ).squaredNorm( ), p_vecRotationTotal.x( ), p_vecRotationTotal.y( ), p_vecRotationTotal.z( ), dMotionScaling );
        m_uWaitKeyTimeoutMS = 0;
    }

    //ds current translation
    m_vecPositionLAST    = m_vecPositionCurrent;
    m_vecPositionCurrent = matTransformationLEFTtoWORLD.translation( );

    //ds update reference
    m_uNumberofVisibleLandmarksLAST = uNumberOfVisibleLandmarks;

    //ds optimize landmarks in every frame
    m_cMatcher.optimizeActiveLandmarks( m_uFrameCount );

    //ds display measurements (blocks)
    m_cMatcher.drawVisibleLandmarks( matDisplayLEFT, matDisplayRIGHT, matTransformationWORLDtoLEFT );

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
        //ds compute cloud for current keyframe
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
                                                  CLinearAccelerationIMU::Zero( ),
                                                  m_cMatcher.getMeasurementsForVisibleLandmarks( ),
                                                  vecCloud,
                                                  0,
                                                  dMotionScaling );

#if defined LOOPCLOSING_BOW_BTREE
            assert( 0 != m_pBoWDatabase );
            assert( 0 < pKeyFrameNEW->vecDescriptorPoolBoW.size( ) );
            m_pBoWDatabase->getVocabulary( )->transform( pKeyFrameNEW->vecDescriptorPoolBoW, pKeyFrameNEW->vecDescriptorPoolB, pKeyFrameNEW->vecDescriptorPoolF, DBOW2_ID_LEVELS  );
#elif defined USING_BOW
            assert( 0 != m_pBoWDatabase );
            assert( 0 < pKeyFrameNEW->vecDescriptorPool.size( ) );
            m_pBoWDatabase->getVocabulary( )->transform( pKeyFrameNEW->vecDescriptorPool, pKeyFrameNEW->vecDescriptorPoolB, pKeyFrameNEW->vecDescriptorPoolF, DBOW2_ID_LEVELS  );
#endif

            //ds set loop closures
            pKeyFrameNEW->vecLoopClosures = _getLoopClosuresForKeyFrame( pKeyFrameNEW, matTransformationLEFTtoWORLD, m_dLoopClosingRadiusSquaredMetersL2, m_dMinimumRelativeMatchesLoopClosure );

            //ds if we found closures
            if( 0 < pKeyFrameNEW->vecLoopClosures.size( ) )
            {
                //ds register closed key frame (ignore actual number of closures for this frame)
                ++m_uLoopClosingKeyFramesInQueue;
            }

            //ds add the new key frame to our stack
            m_vecKeyFrames->push_back( pKeyFrameNEW );

            //ds current key frame id
            const UIDKeyFrame uIDKeyFrameCurrent = m_vecKeyFrames->back( )->uID;

            //ds check if optimization is required (based on key frame id or loop closing) TODO beautify this case
            if( m_uIDDeltaKeyFrameForOptimization < uIDKeyFrameCurrent-m_uIDOptimizedKeyFrameLAST                                                                              ||
               ( m_uLoopClosingKeyFrameWaitingQueue < m_uLoopClosingKeyFramesInQueue && m_uIDDeltaKeyFrameForOptimization < uIDKeyFrameCurrent-m_uIDLoopClosureOptimizedLAST ) )
            {
                //ds shallow optimization
                m_cOptimizer.saveGraph( m_uFrameCount, m_uIDOptimizedKeyFrameLAST, m_uLoopClosingKeyFramesInQueue, m_vecTranslationToG2o );

                //ds if the optimization contained loop closures
                if( 0 < m_uLoopClosingKeyFramesInQueue )
                {
                    m_uIDLoopClosureOptimizedLAST = uIDKeyFrameCurrent;
                }

                //ds update counters
                m_uLoopClosingKeyFramesInQueue = 0;
                m_uIDOptimizedKeyFrameLAST     = m_vecKeyFrames->back( )->uID+1;

                //ds integrate optimization
                m_vecPositionCurrent = matTransformationLEFTtoWORLD.translation( );
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
                    std::printf( "[0][%06lu]<CTrackerGT>(_trackLandmarks) switched to stepwise mode\n", m_uFrameCount );
                }
                else
                {
                    //ds switch to benchmark mode
                    m_uWaitKeyTimeoutMS = 1;
                    m_eMode = ePlaybackBenchmark;
                    std::printf( "[0][%06lu]<CTrackerGT>(_trackLandmarks) switched back to benchmark mode\n", m_uFrameCount );
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
    m_matTransformationWORLDtoLEFTLAST = matTransformationWORLDtoLEFT;
    m_dDistanceTraveledMeters          += m_vecTranslationDeltas.back( ).norm( );
    m_dMotionScalingLAST               = dMotionScaling;
}

void CTrackerGT::_addNewLandmarks( const cv::Mat& p_matImageLEFT,
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

            //ds KITTI default
            pLandmarkNEW->bIsOptimal = true;

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
            //std::printf( "[0][%06lu]<CTrackerGT>(_addNewLandmarks) could not find match for keypoint: '%s'\n", m_uFrameCount, p_cException.what( ) );
        }
    }

    //ds if we couldn't find new landmarks
    if( 0 == vecNewLandmarks->size( ) )
    {
        std::printf( "[0][%06lu]<CTrackerGT>(_getNewLandmarks) unable to detect new landmarks\n", m_uFrameCount );
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

        //std::printf( "<CTrackerGT>(_getNewLandmarks) added new landmarks: %lu/%lu\n", vecNewLandmarks->size( ), vecKeyPoints.size( ) );
    }
}

//ds locked key frames from upper scope
const std::vector< const CKeyFrame::CMatchICP* > CTrackerGT::_getLoopClosuresForKeyFrame( const CKeyFrame* p_pKeyFrameQUERY,
                                                                                          const Eigen::Isometry3d& p_matTransformationLEFTtoWORLDQUERY,
                                                                                          const double& p_dSearchRadiusMetersL2,
                                                                                          const double& p_dMinimumRelativeMatchesLoopClosure )
{
    //ds overall timing
    const double dTimeStartSeconds = CTimer::getTimeSeconds( );

    //ds potential closures list
    std::vector< std::map< UIDLandmark, std::vector< CMatchCloud > > > vecPotentialClosures( m_vecKeyFrames->size( ) );

    //ds last key frame ID available for a closure
    const int64_t uIDKeyFramesAvailableToCloseCap = p_pKeyFrameQUERY->uID-m_uMinimumLoopClosingKeyFrameDistance;

    //ds total matching duration for this query cloud
    double dDurationMatchingSeconds = 0.0;

    //ds to string
    const std::string strMinimumRelativeMatches = std::to_string( static_cast< uint32_t >( m_dMinimumRelativeMatchesLoopClosure*100 ) );

#if defined LOOPCLOSING_BOW_BOW

    const std::string strOutFileTiming( "logs/matching_time_closures_dbow2_dbow2_"+strMinimumRelativeMatches+".txt" );
    const std::string strOutFileClosureMap( "logs/closure_map_dbow2_dbow2_"+strMinimumRelativeMatches+".txt" );

    //ds query descriptors
    const DBoW2::FeatureVector& vecDescriptorPoolFQUERY = p_pKeyFrameQUERY->vecDescriptorPoolF;

    //ds results
    DBoW2::QueryResults vecResultsQUERY;

    //ds get the query results
    const double dTimeStartQuerySeconds = CTimer::getTimeSeconds( );
    m_pBoWDatabase->query( p_pKeyFrameQUERY->vecDescriptorPoolB, vecResultsQUERY, m_vecKeyFrames->size( ), m_vecKeyFrames->size( ) );
    const double dDurationQuerySeconds = CTimer::getTimeSeconds( )-dTimeStartQuerySeconds;
    dDurationMatchingSeconds = dDurationQuerySeconds;

    //ds check results
    for( const DBoW2::Result& cResult: vecResultsQUERY )
    {
        //ds if available
        if( uIDKeyFramesAvailableToCloseCap > cResult.Id )
        {
            //ds if minimum matches are provided
            if( p_dMinimumRelativeMatchesLoopClosure/3.0 < cResult.Score )
            {
                const double dTimeStartGetCorrespondences = CTimer::getTimeSeconds( );
                const CKeyFrame* pKeyFrameREFERENCE = m_vecKeyFrames->at( cResult.Id );
                const DBoW2::FeatureVector& vecDescriptorPoolFREFERENCE = m_pBoWDatabase->retrieveFeatures( cResult.Id );

                std::vector<unsigned int> i_old, i_cur;

                DBoW2::FeatureVector::const_iterator old_it = vecDescriptorPoolFREFERENCE.begin();
                DBoW2::FeatureVector::const_iterator cur_it = vecDescriptorPoolFQUERY.begin( );
                const DBoW2::FeatureVector::const_iterator old_end = vecDescriptorPoolFREFERENCE.end( );
                const DBoW2::FeatureVector::const_iterator cur_end = vecDescriptorPoolFQUERY.end( );

                while(old_it != old_end && cur_it != cur_end)
                {
                    if(old_it->first == cur_it->first)
                    {
                        // compute matches between
                        // features old_it->second of m_image_keys[old_entry] and
                        // features cur_it->second of keys
                        std::vector<unsigned int> i_old_now, i_cur_now;

                        _getMatches_neighratio( pKeyFrameREFERENCE->vecDescriptorPool, old_it->second, p_pKeyFrameQUERY->vecDescriptorPool, cur_it->second, i_old_now, i_cur_now);

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

                dDurationMatchingSeconds += CTimer::getTimeSeconds( )-dTimeStartGetCorrespondences;

                assert( i_cur.size( ) == i_old.size( ) );

                //ds evaluate all matches for this reference cloud
                for( uint64_t u = 0; u < i_old.size( ); ++u )
                {
                    //ds if distance is acceptable (not available here)
                    {
                        //ds buffer query point
                        const CDescriptorVectorPoint3DWORLD* pPointQUERY = p_pKeyFrameQUERY->mapDescriptorToPoint.at( i_cur[u] );

                        try
                        {
                            vecPotentialClosures[pKeyFrameREFERENCE->uID].at( pPointQUERY->uID ).push_back( CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( i_old[u] ), MAXIMUM_DISTANCE_HAMMING ) );
                        }
                        catch( const std::out_of_range& p_cException )
                        {
                            vecPotentialClosures[pKeyFrameREFERENCE->uID].insert( std::make_pair( pPointQUERY->uID, std::vector< CMatchCloud >( 1, CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( i_old[u] ), MAXIMUM_DISTANCE_HAMMING ) ) ) );
                        }
                    }
                }
            }
        }
    }

#elif defined LOOPCLOSING_BOW_BTREE

    const std::string strOutFileTiming( "logs/matching_time_closures_dbow2_btree_"+strMinimumRelativeMatches+".txt" );
    const std::string strOutFileClosureMap( "logs/closure_map_dbow2_btree_"+strMinimumRelativeMatches+".txt" );

    //ds query descriptors for the combined methods
    const DBoW2::FeatureVector& vecDescriptorPoolFQUERY = p_pKeyFrameQUERY->vecDescriptorPoolF;
    const std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > vecDescriptorPoolQUERY = p_pKeyFrameQUERY->vecDescriptorPoolBTree;

    //ds results
    DBoW2::QueryResults vecResultsQUERY;

    //ds get the query results
    const double dTimeStartQuerySeconds = CTimer::getTimeSeconds( );
    m_pBoWDatabase->query( p_pKeyFrameQUERY->vecDescriptorPoolB, vecResultsQUERY, m_vecKeyFrames->size( ), m_vecKeyFrames->size( ) );
    const double dDurationQuerySeconds = CTimer::getTimeSeconds( )-dTimeStartQuerySeconds;
    dDurationMatchingSeconds = dDurationQuerySeconds;

    //ds check results
    for( const DBoW2::Result& cResult: vecResultsQUERY )
    {
        //ds if available
        if( uIDKeyFramesAvailableToCloseCap > cResult.Id )
        {
            //ds if minimum matches are provided
            if( p_dMinimumRelativeMatchesLoopClosure/3.0 < cResult.Score )
            {
                //ds buffer reference key frame
                const CKeyFrame* pKeyFrameREFERENCE = m_vecKeyFrames->at( cResult.Id );

                //ds matches within the current reference
                std::vector< cv::DMatch > vecMatches( 0 );

                //ds match
                const double dTimeStartMatchingSeconds = CTimer::getTimeSeconds( );
                pKeyFrameREFERENCE->m_pBTree->match( vecDescriptorPoolQUERY, vecMatches );
                dDurationMatchingSeconds += CTimer::getTimeSeconds( )-dTimeStartMatchingSeconds;

                //ds evaluate all matches for this reference cloud
                for( const cv::DMatch& cMatch: vecMatches )
                {
                    //ds if distance is acceptable (fixed for BTree)
                    {
                        //ds buffer query point
                        const CDescriptorVectorPoint3DWORLD* pPointQUERY = p_pKeyFrameQUERY->mapDescriptorToPoint.at( cMatch.queryIdx );

                        try
                        {
                            vecPotentialClosures[pKeyFrameREFERENCE->uID].at( pPointQUERY->uID ).push_back( CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) );
                        }
                        catch( const std::out_of_range& p_cException )
                        {
                            vecPotentialClosures[pKeyFrameREFERENCE->uID].insert( std::make_pair( pPointQUERY->uID, std::vector< CMatchCloud >( 1, CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) ) ) );
                        }
                    }
                }
            }
        }
    }

#elif defined USING_BTREE

    const std::string strOutFileTiming( "logs/matching_time_closures_btree_"+strMinimumRelativeMatches+".txt" );
    const std::string strOutFileClosureMap( "logs/closure_map_btree_"+strMinimumRelativeMatches+".txt" );

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
            //ds if distance is acceptable (fixed for BTree)
            {
                //ds buffer query point
                const CDescriptorVectorPoint3DWORLD* pPointQUERY = p_pKeyFrameQUERY->mapDescriptorToPoint.at( cMatch.queryIdx );

                try
                {
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].at( pPointQUERY->uID ).push_back( CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) );
                }
                catch( const std::out_of_range& p_cException )
                {
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].insert( std::make_pair( pPointQUERY->uID, std::vector< CMatchCloud >( 1, CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) ) ) );
                }
            }
        }
    }

#elif defined USING_BF

    const std::string strOutFileTiming( "logs/matching_time_closures_bf_"+strMinimumRelativeMatches+".txt" );
    const std::string strOutFileClosureMap( "logs/closure_map_bf_"+strMinimumRelativeMatches+".txt" );

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
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].insert( std::make_pair( pPointQUERY->uID, std::vector< CMatchCloud >( 1, CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) ) ) );
                }
            }
        }
    }

#elif defined USING_LSH

    const std::string strOutFileTiming( "logs/matching_time_closures_lsh_"+strMinimumRelativeMatches+".txt" );
    const std::string strOutFileClosureMap( "logs/closure_map_lsh_"+strMinimumRelativeMatches+".txt" );

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
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].insert( std::make_pair( pPointQUERY->uID, std::vector< CMatchCloud >( 1, CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( cMatch.trainIdx ), cMatch.distance ) ) ) );
                }
            }
        }
    }

#elif defined USING_BOW

    const std::string strOutFileTiming( "logs/matching_time_closures_dbow2_"+strMinimumRelativeMatches+".txt" );
    const std::string strOutFileClosureMap( "logs/closure_map_dbow2_"+strMinimumRelativeMatches+".txt" );

    //ds query descriptors
    const DBoW2::FeatureVector& vecDescriptorPoolFQUERY = p_pKeyFrameQUERY->vecDescriptorPoolF;

    //ds results
    DBoW2::QueryResults vecResultsQUERY;

    //ds get the query results
    const double dTimeStartQuerySeconds = CTimer::getTimeSeconds( );
    m_pBoWDatabase->query( p_pKeyFrameQUERY->vecDescriptorPoolB, vecResultsQUERY, m_vecKeyFrames->size( ), m_vecKeyFrames->size( ) );
    const double dDurationQuerySeconds = CTimer::getTimeSeconds( )-dTimeStartQuerySeconds;
    dDurationMatchingSeconds = dDurationQuerySeconds;

    //ds closure count based on NC: no correspondence
    uint64_t uNumberOfClosuresNC  = 0;
    double dRelativeMatchesBestNC = 0.0;

    //ds check results
    for( const DBoW2::Result& cResult: vecResultsQUERY )
    {
        //ds if available
        if( uIDKeyFramesAvailableToCloseCap > cResult.Id )
        {
            //ds if minimum matches are provided
            if( p_dMinimumRelativeMatchesLoopClosure/3.0 < cResult.Score )
            {
                //ds register match
                ++uNumberOfClosuresNC;

                //ds update best match if set
                if( dRelativeMatchesBestNC < cResult.Score )
                {
                    dRelativeMatchesBestNC = cResult.Score;
                }
            }
        }
    }

    //ds log DBoW2 performance without correspondence expenses
    std::ofstream ofLogfileTimingNC( "logs/matching_time_closures_dbow2_nc_"+strMinimumRelativeMatches+".txt", std::ofstream::out | std::ofstream::app );
    ofLogfileTimingNC << p_pKeyFrameQUERY->uID << " " << dDurationQuerySeconds << " " << uNumberOfClosuresNC << " " << dRelativeMatchesBestNC << "\n";
    ofLogfileTimingNC.close( );

    //ds get actual matches from results
    for( const DBoW2::Result& cResult: vecResultsQUERY )
    {
        const double dTimeStartGetMatches = CTimer::getTimeSeconds( );
        const CKeyFrame* pKeyFrameREFERENCE = m_vecKeyFrames->at( cResult.Id );
        const DBoW2::FeatureVector& vecDescriptorPoolFREFERENCE = m_pBoWDatabase->retrieveFeatures( cResult.Id );

        std::vector<unsigned int> i_old, i_cur;

        DBoW2::FeatureVector::const_iterator old_it = vecDescriptorPoolFREFERENCE.begin();
        DBoW2::FeatureVector::const_iterator cur_it = vecDescriptorPoolFQUERY.begin( );
        const DBoW2::FeatureVector::const_iterator old_end = vecDescriptorPoolFREFERENCE.end( );
        const DBoW2::FeatureVector::const_iterator cur_end = vecDescriptorPoolFQUERY.end( );

        while(old_it != old_end && cur_it != cur_end)
        {
            if(old_it->first == cur_it->first)
            {
                // compute matches between
                // features old_it->second of m_image_keys[old_entry] and
                // features cur_it->second of keys
                std::vector<unsigned int> i_old_now, i_cur_now;

                _getMatches_neighratio( pKeyFrameREFERENCE->vecDescriptorPool, old_it->second, p_pKeyFrameQUERY->vecDescriptorPool, cur_it->second, i_old_now, i_cur_now);

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

        dDurationMatchingSeconds += CTimer::getTimeSeconds( )-dTimeStartGetMatches;

        assert( i_cur.size( ) == i_old.size( ) );

        //ds evaluate all matches for this reference cloud
        for( uint64_t u = 0; u < i_old.size( ); ++u )
        {
            //ds if distance is acceptable (not available here)
            {
                //ds buffer query point
                const CDescriptorVectorPoint3DWORLD* pPointQUERY = p_pKeyFrameQUERY->mapDescriptorToPoint.at( i_cur[u] );

                try
                {
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].at( pPointQUERY->uID ).push_back( CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( i_old[u] ), MAXIMUM_DISTANCE_HAMMING ) );
                }
                catch( const std::out_of_range& p_cException )
                {
                    vecPotentialClosures[pKeyFrameREFERENCE->uID].insert( std::make_pair( pPointQUERY->uID, std::vector< CMatchCloud >( 1, CMatchCloud( pPointQUERY, pKeyFrameREFERENCE->mapDescriptorToPoint.at( i_old[u] ), MAXIMUM_DISTANCE_HAMMING ) ) ) );
                }
            }
        }
    }

#elif defined USING_BTREE_INDEXED

    const std::string strOutFileTiming( "logs/matching_time_closures_bitree_"+strMinimumRelativeMatches+".txt" );
    const std::string strOutFileClosureMap( "logs/closure_map_bitree_"+strMinimumRelativeMatches+".txt" );

    //ds query descriptors
    const std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > vecDescriptorPoolQUERY = p_pKeyFrameQUERY->vecDescriptorPool;

    //ds cumulative matches vector
    std::vector< cv::DMatch > vecMatches( 0 );

    //ds match
    const double dTimeStartMatchingSeconds = CTimer::getTimeSeconds( );
    m_pBTree->match( vecDescriptorPoolQUERY, p_pKeyFrameQUERY->uID, vecMatches );
    dDurationMatchingSeconds = CTimer::getTimeSeconds( )-dTimeStartMatchingSeconds;

    //ds evaluate all matches
    for( const cv::DMatch& cMatch: vecMatches )
    {
        //ds if distance is acceptable (fixed for BTree)
        {
            //ds buffer points
            const CDescriptorVectorPoint3DWORLD* pPointQUERY     = p_pKeyFrameQUERY->mapDescriptorToPoint.at( cMatch.queryIdx );
            const CDescriptorVectorPoint3DWORLD* pPointREFERENCE = m_vecKeyFrames->at( cMatch.imgIdx )->mapDescriptorToPoint.at( cMatch.trainIdx );

            try
            {
                vecPotentialClosures[cMatch.imgIdx].at( pPointQUERY->uID ).push_back( CMatchCloud( pPointQUERY, pPointREFERENCE, cMatch.distance ) );
            }
            catch( const std::out_of_range& p_cException )
            {
                vecPotentialClosures[cMatch.imgIdx].insert( std::make_pair( pPointQUERY->uID, std::vector< CMatchCloud >( 1, CMatchCloud( pPointQUERY, pPointREFERENCE, cMatch.distance ) ) ) );
            }
        }
    }

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

    //ds closure vector to align
    std::vector< std::pair< const CKeyFrame*, const std::vector< CMatchCloud > > > vecClosuresToCompute;

    //ds check all keyframes for distance
    for( int64_t uIDREFERENCE = 0; uIDREFERENCE < uIDKeyFramesAvailableToCloseCap; ++uIDREFERENCE )
    {
        assert( 0 <= uIDREFERENCE );

        //ds compute relative matches
        const double dRelativeMatches = static_cast< double >( vecPotentialClosures[uIDREFERENCE].size( ) )/p_pKeyFrameQUERY->vecCloud->size( );

        if( dRelativeMatchesBest < dRelativeMatches )
        {
            dRelativeMatchesBest = dRelativeMatches;
        }

        //ds if we have a sufficient amount of matches
        if( p_dMinimumRelativeMatchesLoopClosure < dRelativeMatches )
        {
            //ds stats
            matClosureMapNew( m_vecKeyFrames->size( ), uIDREFERENCE ) = 1.0;
            matClosureMapNew( uIDREFERENCE, m_vecKeyFrames->size( ) ) = 1.0;
            ++uNumberOfClosures;
            std::printf( "[0][%06lu]<CTrackerSVI>(_getLoopClosuresForKeyFrame) found closure: [%06lu] > [%06lu] relative matches: %f (%lu/%lu)\n",
                         m_uFrameCount, p_pKeyFrameQUERY->uID, uIDREFERENCE, dRelativeMatches, vecPotentialClosures[uIDREFERENCE].size( ), p_pKeyFrameQUERY->vecCloud->size( ) );




            //ds spatial matches for ICP loop closure computation
            std::vector< CMatchCloud > vecMatchesForICP;
            vecMatchesForICP.reserve( vecPotentialClosures[uIDREFERENCE].size( ) );

            //ds filter actual spatial matches
            for( const std::pair< UIDLandmark, std::vector< CMatchCloud > >& prMatch: vecPotentialClosures[uIDREFERENCE] )
            {
                vecMatchesForICP.push_back( _getMatchNN( prMatch.second ) );
            }

            //ds add to compute
            vecClosuresToCompute.push_back( std::make_pair( m_vecKeyFrames->at( uIDREFERENCE ), vecMatchesForICP ) );
        }
    }

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

    //ds solution vector
    std::vector< const CKeyFrame::CMatchICP* > vecLoopClosures;

    //ds success rate count
    uint32_t uNumberOfClosedKeyFrames = 0;

    //ds loop over all closures to align
    for( const std::pair< const CKeyFrame*, const std::vector< CMatchCloud > >& prMatch: vecClosuresToCompute )
    {
        //ds buffer reference frame
        const CKeyFrame* pKeyFrameREFERENCE = prMatch.first;

        //ds buffer matches
        const std::vector< CMatchCloud >& vecMatches = prMatch.second;

        //ds transformation between this keyframe and the loop closure one (take current measurement as prior)
        Eigen::Isometry3d matTransformationToClosure( pKeyFrameREFERENCE->matTransformationLEFTtoWORLD.inverse( )*p_matTransformationLEFTtoWORLDQUERY );

        //ds and erase translation
        matTransformationToClosure.translation( ) = Eigen::Vector3d::Zero( );
        //const Eigen::Isometry3d matTransformationToClosureInitial( matTransformationToClosure );

        //ds 1mm for convergence
        const double dErrorDeltaForConvergence      = 1e-5;
        double dErrorSquaredTotalPrevious           = 0.0;
        const double dMaximumErrorForInlier         = 0.25; //0.25
        const double dMaximumErrorAverageForClosure = 0.2; //0.1
        const uint32_t uMaximumIterations           = 1000;

        //ds LS setup
        Eigen::Matrix< double, 6, 6 > matH;
        Eigen::Matrix< double, 6, 1 > vecB;
        Eigen::Matrix3d matOmega( Eigen::Matrix3d::Identity( ) );

        //ds run least-squares maximum 100 times
        for( uint32_t uLS = 0; uLS < uMaximumIterations; ++uLS )
        {
            //ds error
            double dErrorSquaredTotal = 0.0;
            uint32_t uInliers         = 0;

            //ds initialize setup
            matH.setZero( );
            vecB.setZero( );

            //ds for all the points
            for( const CMatchCloud& cMatch: vecMatches )
            {
                //ds compute projection into closure
                const CPoint3DCAMERA vecPointXYZQuery( matTransformationToClosure*cMatch.pPointQuery->vecPointXYZCAMERA );
                if( 0.0 < vecPointXYZQuery.z( ) )
                {
                    assert( 0.0 < cMatch.pPointReference->vecPointXYZCAMERA.z( ) );

                    //ds adjust omega to inverse depth value (the further away the point, the less weight)
                    matOmega(2,2) = 1.0/( cMatch.pPointReference->vecPointXYZCAMERA.z( ) );

                    //ds compute error
                    const Eigen::Vector3d vecError( vecPointXYZQuery-cMatch.pPointReference->vecPointXYZCAMERA );

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
                const double dErrorAverage = dErrorSquaredTotal/vecMatches.size( );

                //ds if the solution is acceptable
                if( dMaximumErrorAverageForClosure > dErrorAverage && 0 < uInliers )
                {
                    //std::printf( "<CTrackerSVI>(_getLoopClosuresForKeyFrame) found closure: [%06lu] > [%06lu] (matches: %3lu, iterations: %2u, average error: %5.3f, inliers: %2u)\n",
                    //             p_pKeyFrameQUERY->uID, pKeyFrameREFERENCE->uID, vecMatches.size( ), uLS, dErrorAverage, uInliers );
                    //vecLoopClosures.push_back( new CKeyFrame::CMatchICP( pKeyFrameREFERENCE, matTransformationToClosure, vecMatches ) );
                    ++uNumberOfClosedKeyFrames;
                    break;
                }
                else
                {
                    //std::printf( "%4.1f %4.1f %4.1f | ", matTransformationToClosure.translation( ).x( ), matTransformationToClosure.translation( ).y( ), matTransformationToClosure.translation( ).z( ) );
                    //std::printf( "<CTrackerSVI>(_getLoopClosuresForKeyFrame) system converged INVALID in %2u iterations, average error: %5.3f (inliers: %2u) - discarded\n", uLS, dErrorAverage, uInliers );
                    break;
                }
            }
            else
            {
                dErrorSquaredTotalPrevious = dErrorSquaredTotal;
            }

            //ds if not converged
            if( uMaximumIterations-1 == uLS )
            {
                //std::printf( "<CTrackerSVI>(_getLoopClosuresForKeyFrame) system did not converge\n" );
            }
        }
    }

    if( 0 < vecClosuresToCompute.size( ) )
    {
        const double dSuccessRateICP = static_cast< double >( uNumberOfClosedKeyFrames )/vecClosuresToCompute.size( );
        std::printf( "[0][%06lu]<CTrackerSVI>(_getLoopClosuresForKeyFrame) [%06lu] ICP success rate: %f\n", m_uFrameCount, p_pKeyFrameQUERY->uID, dSuccessRateICP );

        //ds log stats
        std::ofstream ofLogfileTiming( strOutFileTiming, std::ofstream::out | std::ofstream::app );
        ofLogfileTiming << p_pKeyFrameQUERY->uID << " " << dDurationMatchingSeconds << " " << uNumberOfClosures << " " << dRelativeMatchesBest << " " << dSuccessRateICP << "\n";
        ofLogfileTiming.close( );
    }
    else
    {
        //ds log stats (no closures to compute, default success rate)
        std::ofstream ofLogfileTiming( strOutFileTiming, std::ofstream::out | std::ofstream::app );
        ofLogfileTiming << p_pKeyFrameQUERY->uID << " " << dDurationMatchingSeconds << " " << uNumberOfClosures << " " << dRelativeMatchesBest << " " << 1 << "\n";
        ofLogfileTiming.close( );
    }

#if defined USING_BOW
    m_pBoWDatabase->add( p_pKeyFrameQUERY->vecDescriptorPoolB, vecDescriptorPoolFQUERY );
#endif

#if defined USING_BTREE_INDEXED
    m_pBTree->add( vecDescriptorPoolQUERY );
#endif


    //ds info
    m_dDurationTotalSecondsLoopClosing += CTimer::getTimeSeconds( )-dTimeStartSeconds;

    //ds return found closures
    return vecLoopClosures;
}

//ds TODO make this efficient
const CMatchCloud CTrackerGT::_getMatchNN( const std::vector< CMatchCloud >& p_vecMatches ) const
{
    assert( 0 < p_vecMatches.size( ) );

    //ds point counts
    std::set< UIDLandmark > setCounts;

    //ds best match and count so far
    const CMatchCloud* pMatchBest = 0;
    uint32_t uCountBest           = 0;

    //ds loop over the list and count entries
    for( const CMatchCloud& cMatch: p_vecMatches )
    {
        //ds update count
        setCounts.insert( cMatch.pPointReference->uID );
        const uint32_t uCountCurrent = setCounts.count( cMatch.pPointReference->uID );

        //ds if we get a better count
        if( uCountBest < uCountCurrent )
        {
            uCountBest = uCountCurrent;
            pMatchBest = &cMatch;
        }
    }

    assert( 0 != pMatchBest );

    //ds return with best match
    return CMatchCloud( pMatchBest->pPointQuery, pMatchBest->pPointReference, pMatchBest->dMatchingDistance );
}

void CTrackerGT::_initializeTranslationWindow( )
{
    //ds reinitialize translation window
    m_vecTranslationDeltas.clear( );
    for( std::vector< Eigen::Vector3d >::size_type u = 0; u < m_uIMULogbackSize; ++u )
    {
        m_vecTranslationDeltas.push_back( Eigen::Vector3d::Zero( ) );
    }
    m_vecGradientXYZ = Eigen::Vector3d::Zero( );
}

void CTrackerGT::_shutDown( )
{
    m_bIsShutdownRequested = true;
    std::printf( "[0][%06lu]<CTrackerGT>(_shutDown) termination requested, <CTrackerGT> disabled\n", m_uFrameCount );
}

void CTrackerGT::_updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange )
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

void CTrackerGT::_drawInfoBox( cv::Mat& p_matDisplay, const double& p_dMotionScaling ) const
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
            std::printf( "[0][%06lu]<CTrackerGT>(_drawInfoBox) unsupported playback mode, no info box displayed\n", m_uFrameCount );
            break;
        }
    }


    p_matDisplay( cv::Rect( 0, 0, 2*m_pCameraSTEREO->m_uPixelWidth, 17 ) ).setTo( CColorCodeBGR( 0, 0, 0 ) );
    cv::putText( p_matDisplay, chBuffer , cv::Point2i( 2, 12 ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );
}

#if defined USING_BOW

//ds snippet: https://github.com/dorian3d/DLoopDetector/blob/master/include/DLoopDetector/TemplatedLoopDetector.h
void CTrackerGT::_getMatches_neighratio( const std::vector< boost::dynamic_bitset<>> &A,
                             const std::vector<unsigned int> &i_A,
                             const std::vector<boost::dynamic_bitset<>> &B,
                             const std::vector<unsigned int> &i_B,
                             std::vector<unsigned int> &i_match_A,
                             std::vector<unsigned int> &i_match_B ) const
{
    i_match_A.resize(0);
    i_match_B.resize(0);
    i_match_A.reserve( min(i_A.size(), i_B.size()) );
    i_match_B.reserve( min(i_A.size(), i_B.size()) );

    vector<unsigned int>::const_iterator ait, bit;
    unsigned int i, j;
    i = 0;
    for(ait = i_A.begin(); ait != i_A.end(); ++ait, ++i)
    {
        int best_j_now = -1;
        double best_dist_1 = 1e9;
        double best_dist_2 = 1e9;

        j = 0;
        for(bit = i_B.begin(); bit != i_B.end(); ++bit, ++j)
        {
            double d = DBoW2::FBrief::distance(A[*ait], B[*bit]);

            // in i
            if(d < best_dist_1)
            {
                best_j_now = j;
                best_dist_2 = best_dist_1;
                best_dist_1 = d;
            }
            else if(d < best_dist_2)
            {
                best_dist_2 = d;
            }
        }

        //ds if the best distance is a portion of the second best
        if(best_dist_1 / best_dist_2 <= 0.6)
        {
            unsigned int idx_B = i_B[best_j_now];
            bit = find(i_match_B.begin(), i_match_B.end(), idx_B);

            if(bit == i_match_B.end())
            {
                //ds if matching distance is satisfactory
                if( MAXIMUM_DISTANCE_HAMMING > best_dist_1 )
                {
                    i_match_B.push_back(idx_B);
                    i_match_A.push_back(*ait);
                }
            }
            else
            {
                unsigned int idx_A = i_match_A[ bit - i_match_B.begin() ];
                double d = DBoW2::FBrief::distance(A[idx_A], B[idx_B]);
                if(best_dist_1 < d)
                {
                    i_match_A[ bit - i_match_B.begin() ] = *ait;
                }
            }
        }
    }
}

#endif

