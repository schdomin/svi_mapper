#include "CTrackerGT.h"

#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>

#include "../gui/CConfigurationOpenCV.h"
#include "../exceptions/CExceptionPoseOptimization.h"
#include "../exceptions/CExceptionNoMatchFound.h"
#include "utility/CIMUInterpolator.h"



CTrackerGT::CTrackerGT( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                                const EPlaybackMode& p_eMode,
                                const double& p_dMinimumRelativeMatchesLoopClosure,
                                const uint32_t& p_uWaitKeyTimeoutMS ): m_uWaitKeyTimeoutMS( p_uWaitKeyTimeoutMS ),
                                                                           m_pCameraLEFT( p_pCameraSTEREO->m_pCameraLEFT ),
                                                                           m_pCameraRIGHT( p_pCameraSTEREO->m_pCameraRIGHT ),
                                                                           m_pCameraSTEREO( p_pCameraSTEREO ),

                                                                           m_vecKeyFrames( std::make_shared< std::vector< CKeyFrame* > >( ) ),

                                                                           m_matTransformationWORLDtoLEFTLAST( CIMUInterpolator::getTransformationInitialWORLDtoLEFT( ) ),
                                                                           m_matTransformationLEFTLASTtoLEFTNOW( Eigen::Matrix4d::Identity( ) ),
                                                                           m_vecPositionKeyFrameLAST( m_matTransformationWORLDtoLEFTLAST.inverse( ).translation( ) ),
                                                                           m_vecPositionCurrent( m_vecPositionKeyFrameLAST ),
                                                                           m_vecPositionLAST( m_vecPositionCurrent ),

                                                                           m_uVisibleLandmarksMinimum( 100 ),

                                                                           m_cMatcher( m_pCameraSTEREO ),
                                                                           m_cOptimizer( m_pCameraSTEREO, m_vecKeyFrames, m_matTransformationWORLDtoLEFTLAST.inverse( ) ),

                                                                           m_dMinimumRelativeMatchesLoopClosure( p_dMinimumRelativeMatchesLoopClosure ),
                                                                           m_eMode( p_eMode ),
                                                                           m_strVersionInfo( "CTrackerGT [" + std::to_string( m_pCameraSTEREO->m_uPixelWidth )
                                                                                                             + "|" + std::to_string( m_pCameraSTEREO->m_uPixelHeight ) + "]" )
#define DBOW2_ID_LEVELS 2
                                                                          ,m_pBoWDatabase( std::make_shared< BriefDatabase >( BriefVocabulary( "brief_k10L6.voc.gz" ), true, DBOW2_ID_LEVELS ) )
{
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

            assert( 0 != m_pBoWDatabase );
            assert( 0 < pKeyFrameNEW->vecDescriptorPoolBoW.size( ) );
            m_pBoWDatabase->getVocabulary( )->transform( pKeyFrameNEW->vecDescriptorPoolBoW, pKeyFrameNEW->vecDescriptorPoolB, pKeyFrameNEW->vecDescriptorPoolF, DBOW2_ID_LEVELS  );

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
                //ds load new landmarks to graph
                m_cMatcher.addLandmarksToGraph( m_cOptimizer, m_vecTranslationToG2o, m_uFrameCount );

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

        //ds add new landmarks (blocks partially)
        m_uNumberofVisibleLandmarksLAST = m_cMatcher.addNewLandmarks( p_matImageLEFT, p_matImageRIGHT, matTransformationWORLDtoLEFT, matTransformationLEFTtoWORLD, m_uFrameCount, m_matDisplayLowerReference );

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
    const std::string strOutFileTiming( "logs/matching_time_closures_dbow2_vbst.txt" );
    const std::string strOutFileClosureMap( "logs/closure_map_dbow2_vbst.txt" );

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
            if( p_dMinimumRelativeMatchesLoopClosure/4.0 < cResult.Score )
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

    //ds validate design
    assert( p_pKeyFrameQUERY->uID == vecPotentialClosures.size( ) );

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
            dRelativeMatchesBest   = dRelativeMatches;
        }

        //ds if we have a sufficient amount of matches
        if( p_dMinimumRelativeMatchesLoopClosure < dRelativeMatches )
        {
            /*std::printf( "[0][%06lu]<CTrackerSVI>(_getLoopClosuresForKeyFrame) found closure: [%06lu] > [%06lu] relative matches: %f (%lu/%lu)\n",
                         m_uFrameCount, p_pKeyFrameQUERY->uID, uIDREFERENCE, dRelativeMatches, vecPotentialClosures[uIDREFERENCE].size( ), p_pKeyFrameQUERY->vecCloud->size( ) );*/

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
        const double dMaximumErrorForInlier         = 1.0; //0.25
        const double dMaximumErrorAverageForClosure = 0.9; //0.1
        const uint32_t uMaximumIterations           = 1000;
        const uint32_t uMinimumInliers              = 25;

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
                if( dMaximumErrorAverageForClosure > dErrorAverage && uMinimumInliers < uInliers )
                {
                    std::printf( "<CTrackerSVI>(_getLoopClosuresForKeyFrame) found closure: [%06lu] > [%06lu] (matches: %3lu, iterations: %2u, average error: %5.3f, inliers: %2u)\n",
                                 p_pKeyFrameQUERY->uID, pKeyFrameREFERENCE->uID, vecMatches.size( ), uLS, dErrorAverage, uInliers );
                    //vecLoopClosures.push_back( new CKeyFrame::CMatchICP( pKeyFrameREFERENCE, matTransformationToClosure, vecMatches ) );
                    ++uNumberOfClosedKeyFrames;
                    break;
                }
                else
                {
                    std::printf( "<CTrackerSVI>(_getLoopClosuresForKeyFrame) system converged INVALID in %2u iterations, average error: %5.3f (inliers: %2u) - discarded\n", uLS, dErrorAverage, uInliers );
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

    /*if( 0 < vecClosuresToCompute.size( ) )
    {
        const double dSuccessRateICP = static_cast< double >( uNumberOfClosedKeyFrames )/vecClosuresToCompute.size( );
        std::printf( "[0][%06lu]<CTrackerSVI>(_getLoopClosuresForKeyFrame) [%06lu] ICP success rate: %f\n", m_uFrameCount, p_pKeyFrameQUERY->uID, dSuccessRateICP );
    }*/

    m_uTotalNumberOfVerifiedClosures += uNumberOfClosedKeyFrames;

    m_pBoWDatabase->add( p_pKeyFrameQUERY->vecDescriptorPoolB, vecDescriptorPoolFQUERY );

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
        //ds fall-through intended
        case ePlaybackStepwise:
        case ePlaybackBenchmark:
        {
            std::snprintf( chBuffer, 1024, "[%13.2f"
                                           "|%06lu] FPS: %4.1f "
                                           "| X: %5.1f Y: %5.1f Z: %5.1f DELTA L2: %5.2f SCALING: %4.2f "
                                           "| LANDMARKs V: %3i (%3lu,%3lu,%3lu,%3lu) F: %5lu I: %5lu T: %5lu(%5lu) "
                                           "| DETECTIONs: %2lu(%3lu) "
                                           "| KFs: %3lu "
                                           "| OPTIMIZATIONs: %3lu PXYZ: %5lu",
                           m_dTimestampLASTSeconds, m_uFrameCount, m_dPreviousFrameRate,
                           m_vecPositionCurrent.x( ), m_vecPositionCurrent.y( ), m_vecPositionCurrent.z( ), m_dTranslationDeltaSquaredNormCurrent, p_dMotionScaling,
                           m_uNumberofVisibleLandmarksLAST, m_cMatcher.getNumberOfTracksStage1( ), m_cMatcher.getNumberOfTracksStage2_1( ), m_cMatcher.getNumberOfTracksStage3( ), m_cMatcher.getNumberOfTracksStage2_2( ), m_cMatcher.getNumberOfFailedLandmarkOptimizations( ), m_cMatcher.getNumberOfInvalidLandmarksTotal( ), m_cMatcher.getNumberOfLandmarksInWINDOW( ), m_cMatcher.getNumberOfLandmarksTotal( ),
                           m_cMatcher.getNumberOfDetectionPointsActive( ), m_cMatcher.getNumberOfDetectionPointsTotal( ),
                           m_vecKeyFrames->size( ),
                           m_cOptimizer.getNumberOfOptimizations( ), m_cMatcher.getNumberOfLandmarksInGRAPH( ) );
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

