#include "CFundamentalMatcher.h"

#include "exceptions/CExceptionNoMatchFound.h"
#include "exceptions/CExceptionNoMatchFoundInternal.h"
#include "exceptions/CExceptionPoseOptimization.h"
#include "vision/CMiniVisionToolbox.h"
#include "utility/CLogger.h"
#include "exceptions/CExceptionEpipolarLine.h"



CFundamentalMatcher::CFundamentalMatcher( const std::shared_ptr< CTriangulator > p_pTriangulator,
                                    const std::shared_ptr< cv::FeatureDetector > p_pDetectorSingle ): m_pTriangulator( p_pTriangulator ),
                                                                              m_pCameraLEFT( m_pTriangulator->m_pCameraSTEREO->m_pCameraLEFT ),
                                                                              m_pCameraRIGHT( m_pTriangulator->m_pCameraSTEREO->m_pCameraRIGHT ),
                                                                              m_pCameraSTEREO( m_pTriangulator->m_pCameraSTEREO ),
                                                                              m_pDetector( p_pDetectorSingle ),
                                                                              m_pExtractor( m_pTriangulator->m_pExtractor ),
                                                                              m_pMatcher( m_pTriangulator->m_pMatcher ),
                                                                              m_dMinimumDepthMeters( m_pTriangulator->dDepthMinimumMeters ),
                                                                              m_dMaximumDepthMeters( m_pTriangulator->dDepthMaximumMeters ),
                                                                              m_dMatchingDistanceCutoffTrackingStage1( 25.0 ),
                                                                              m_dMatchingDistanceCutoffTrackingStage2( 50.0 ),
                                                                              m_dMatchingDistanceCutoffTrackingStage3( 50.0 ),
                                                                              m_dMatchingDistanceCutoffOriginal( 2*m_dMatchingDistanceCutoffTrackingStage3 ),
                                                                              m_uAvailableDetectionPointID( 0 ),
                                                                              m_cSolverSterePosit( m_pCameraLEFT->m_matProjection, m_pCameraRIGHT->m_matProjection )
{
    m_vecDetectionPointsActive.clear( );
    m_vecVisibleLandmarks.clear( );
    m_vecMeasurementsVisible.clear( );

    CLogger::openBox( );
    std::printf( "[0]<CFundamentalMatcher>(CFundamentalMatcher) descriptor extractor: %s\n", m_pExtractor->name( ).c_str( ) );
    std::printf( "[0]<CFundamentalMatcher>(CFundamentalMatcher) descriptor matcher: %s\n", m_pMatcher->name( ).c_str( ) );
    std::printf( "[0]<CFundamentalMatcher>(CFundamentalMatcher) minimum depth cutoff: %f\n", m_dMinimumDepthMeters );
    std::printf( "[0]<CFundamentalMatcher>(CFundamentalMatcher) maximum depth cutoff: %f\n", m_dMinimumDepthMeters );
    std::printf( "[0]<CFundamentalMatcher>(CFundamentalMatcher) matching distance cutoff stage 1: %f\n", m_dMatchingDistanceCutoffTrackingStage1 );
    std::printf( "[0]<CFundamentalMatcher>(CFundamentalMatcher) matching distance cutoff stage 2: %f\n", m_dMatchingDistanceCutoffTrackingStage2 );
    std::printf( "[0]<CFundamentalMatcher>(CFundamentalMatcher) matching distance cutoff stage 3: %f\n", m_dMatchingDistanceCutoffTrackingStage3 );
    std::printf( "[0]<CFundamentalMatcher>(CFundamentalMatcher) maximum number of non-detections before dropping landmark: %u\n", m_uMaximumFailedSubsequentTrackingsPerLandmark );
    std::printf( "[0]<CFundamentalMatcher>(CFundamentalMatcher) instance allocated\n" );
    CLogger::closeBox( );
}

CFundamentalMatcher::~CFundamentalMatcher( )
{
    //CLogger::CLogDetectionEpipolar::close( );
    //CLogger::CLogOptimizationOdometry::close( );
    std::printf( "[0]<CFundamentalMatcher>(~CFundamentalMatcher) instance deallocated\n" );
}

//ds landmark locked in upper scope
void CFundamentalMatcher::addDetectionPoint( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks )
{
    assert( 0 < p_vecLandmarks->size( ) );
    m_vecDetectionPointsActive.push_back( CDetectionPoint( m_uAvailableDetectionPointID, p_matTransformationLEFTtoWORLD, p_vecLandmarks ) );
    ++m_uAvailableDetectionPointID;
}

//ds routine that resets the visibility of all active landmarks
void CFundamentalMatcher::resetVisibilityActiveLandmarks( )
{
    //ds loop over the currently visible landmarks
    for( CLandmark* pLandmark: m_vecVisibleLandmarks )
    {
        pLandmark->bIsCurrentlyVisible = false;
    }

    //ds clear reference vector
    m_vecVisibleLandmarks.clear( );
}

void CFundamentalMatcher::setKeyFrameToVisibleLandmarks( )
{
    //ds loop over the currently visible landmarks
    for( CLandmark* pLandmark: m_vecVisibleLandmarks )
    {
        ++pLandmark->uNumberOfKeyFramePresences;
    }
}

void CFundamentalMatcher::optimizeActiveLandmarks( const UIDFrame& p_uFrame ) const
{
    //ds active measurements
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds optimize contained landmarks
            pLandmark->optimize( p_uFrame );
        }
    }
}

const std::shared_ptr< const std::vector< CLandmark* > > CFundamentalMatcher::getVisibleOptimizedLandmarks( ) const
{
    //ds return vector
    std::shared_ptr< std::vector< CLandmark* > > vecVisibleLandmarks( std::make_shared< std::vector< CLandmark* > >( ) );

    for( CLandmark* pLandmark: m_vecVisibleLandmarks )
    {
        if( pLandmark->bIsOptimal )
        {
            vecVisibleLandmarks->push_back( pLandmark );
        }
    }

    return vecVisibleLandmarks;
}

const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > CFundamentalMatcher::getCloudForVisibleOptimizedLandmarks( const UIDFrame& p_uFrame ) const
{
    //ds return vector
    std::shared_ptr< std::vector< CDescriptorVectorPoint3DWORLD* > > vecCloud( std::make_shared< std::vector< CDescriptorVectorPoint3DWORLD* > >( ) );

    for( CLandmark* pLandmark: m_vecVisibleLandmarks )
    {
        //ds trigger optimization manually
        pLandmark->optimize( p_uFrame );

        //ds check if optimal
        if( pLandmark->bIsOptimal )
        {
            vecCloud->push_back( new CDescriptorVectorPoint3DWORLD( pLandmark->uID,
                                                                    pLandmark->vecPointXYZOptimized,
                                                                    pLandmark->getLastPointXYZLEFT( ),
                                                                    pLandmark->getLastDetectionLEFT( ),
                                                                    pLandmark->getLastDetectionRIGHT( ),
                                                                    pLandmark->vecDescriptorsLEFT ) );
        }
    }

    return vecCloud;
}

//ds compute 3D pose using stereo posit - EXCEPTION - locked from upper scope
const Eigen::Isometry3d CFundamentalMatcher::getPoseStereoPosit( const UIDFrame p_uFrame,
                                                                  cv::Mat& p_matDisplayLEFT,
                                                                  cv::Mat& p_matDisplayRIGHT,
                                                                  const cv::Mat& p_matImageLEFT,
                                                                  const cv::Mat& p_matImageRIGHT,
                                                                  const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                                                                  const Eigen::Isometry3d& p_matTransformationWORLDtoLEFTLAST,
                                                                  const Eigen::Vector3d& p_vecRotationIMU,
                                                                  const Eigen::Vector3d& p_vecTranslationIMU,
                                                                  const double& p_dMotionScaling )
{
    assert( 1.0 <= p_dMotionScaling );

    //ds found landmarks in this frame
    std::vector< CSolverStereoPosit::CMatch > vecMeasurementsForStereoPosit;

    //ds single keypoint buffer
    std::vector< cv::KeyPoint > vecKeyPointBufferSingle( 1 );

    //ds timing
    const double dTimeStartSeconds = CTimer::getTimeSeconds( );

    //ds info
    m_uNumberOfTracksStage1   = 0;
    m_uNumberOfTracksStage2_1 = 0;

    //ds triangulation search length scaling
    const float fTriangulationScale = 1.0+p_dMotionScaling;

    //ds 2 STAGE tracking algorithm
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan (use all points not only optimized and visible ones)
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds only handle optimal landmarks
            if( pLandmark->bIsOptimal )
            {
                //ds compute current reprojection point
                const CPoint3DCAMERA vecPointXYZLEFT( p_matTransformationEstimateWORLDtoLEFT*pLandmark->vecPointXYZOptimized );
                const cv::Point2f ptUVEstimateLEFT( m_pCameraLEFT->getProjectionRounded( vecPointXYZLEFT ) );
                const cv::Point2f ptUVEstimateRIGHT( m_pCameraRIGHT->getProjectionRounded( vecPointXYZLEFT ) );
                assert( ptUVEstimateLEFT.y == ptUVEstimateRIGHT.y );

                const float fKeyPointSizePixels       = pLandmark->dKeyPointSize;
                const float fKeyPointSizePixelsHalf   = 4*fKeyPointSizePixels;
                const float fKeyPointSizePixelsLength = 8*fKeyPointSizePixels+1;
                const cv::Point2f ptOffsetKeyPointHalf( fKeyPointSizePixelsHalf, fKeyPointSizePixelsHalf );
                const float fSearchRange = fTriangulationScale*pLandmark->getLastDisparity( );

                //ds check if we are in tracking range
                if( m_pCameraLEFT->m_cFieldOfView.contains( ptUVEstimateLEFT ) && m_pCameraRIGHT->m_cFieldOfView.contains( ptUVEstimateRIGHT ) )
                {
                    //ds STAGE 1: check LEFT
                    try
                    {
                        //ds LEFT roi
                        const cv::Point2f ptUVEstimateLEFTROI( ptUVEstimateLEFT-ptOffsetKeyPointHalf );
                        const cv::Rect matROI( ptUVEstimateLEFTROI.x, ptUVEstimateLEFTROI.y, fKeyPointSizePixelsLength, fKeyPointSizePixelsLength );

                        //ds compute descriptor at this point
                        vecKeyPointBufferSingle[0] = cv::KeyPoint( ptOffsetKeyPointHalf, pLandmark->dKeyPointSize );
                        cv::Mat matDescriptorLEFT;
                        m_pExtractor->compute( p_matImageLEFT( matROI ), vecKeyPointBufferSingle, matDescriptorLEFT );

                        //ds if acceptable
                        if( 1 == matDescriptorLEFT.rows && m_dMatchingDistanceCutoffTrackingStage1 > cv::norm( pLandmark->getLastDescriptorLEFT( ), matDescriptorLEFT, cv::NORM_HAMMING ) )
                        {
                            //ds triangulate the point directly
                            const CMatchTriangulation cMatchRIGHT( m_pTriangulator->getPointTriangulatedInRIGHT( p_matImageRIGHT,
                                                                                                                 std::max( 0.0f, ptUVEstimateLEFTROI.x-fSearchRange ),
                                                                                                                 ptUVEstimateLEFTROI.y,
                                                                                                                 fKeyPointSizePixels,
                                                                                                                 ptUVEstimateLEFTROI+vecKeyPointBufferSingle[0].pt,
                                                                                                                 matDescriptorLEFT ) );
                            const CDescriptor matDescriptorRIGHT( cMatchRIGHT.matDescriptorCAMERA );

                            //ds check depth
                            const double dDepthMeters = cMatchRIGHT.vecPointXYZCAMERA.z( );
                            if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                            {
                                throw CExceptionNoMatchFound( "invalid depth" );
                            }

                            //ds check if the descriptor match on the right side is out of range
                            if( m_dMatchingDistanceCutoffTrackingStage1 < cv::norm( pLandmark->getLastDescriptorRIGHT( ), matDescriptorRIGHT, cv::NORM_HAMMING ) )
                            {
                                throw CExceptionNoMatchFound( "triangulation descriptor mismatch" );
                            }

                            //ds latter landmark update (cannot be done before pose is optimized)
                            vecMeasurementsForStereoPosit.push_back( CSolverStereoPosit::CMatch( pLandmark, pLandmark->vecPointXYZOptimized, cMatchRIGHT.vecPointXYZCAMERA, ptUVEstimateLEFT, cMatchRIGHT.ptUVCAMERA, matDescriptorLEFT, matDescriptorRIGHT ) );
                            cv::circle( p_matDisplayLEFT, ptUVEstimateLEFT, 4, CColorCodeBGR( 0, 255, 0 ), 1 );
                            ++m_uNumberOfTracksStage1;
                        }
                        else
                        {
                            throw CExceptionNoMatchFound( "insufficient matching distance" );
                        }
                    }
                    catch( const CExceptionNoMatchFound& p_cExceptionStage1LEFT )
                    {
                        //ds STAGE 1: check RIGHT
                        try
                        {
                            //ds RIGHT roi
                            const cv::Point2f ptUVEstimateRIGHTROI( ptUVEstimateRIGHT-ptOffsetKeyPointHalf );
                            const cv::Rect matROI( ptUVEstimateRIGHTROI.x, ptUVEstimateRIGHTROI.y, fKeyPointSizePixelsLength, fKeyPointSizePixelsLength );

                            //ds compute descriptor at this point
                            vecKeyPointBufferSingle[0] = cv::KeyPoint( ptOffsetKeyPointHalf, pLandmark->dKeyPointSize );
                            cv::Mat matDescriptorRIGHT;
                            m_pExtractor->compute( p_matImageRIGHT( matROI ), vecKeyPointBufferSingle, matDescriptorRIGHT );

                            //ds if acceptable
                            if( 1 == matDescriptorRIGHT.rows && m_dMatchingDistanceCutoffTrackingStage1 > cv::norm( pLandmark->getLastDescriptorRIGHT( ), matDescriptorRIGHT, cv::NORM_HAMMING ) )
                            {
                                //ds triangulate the point directly
                                const CMatchTriangulation cMatchLEFT( m_pTriangulator->getPointTriangulatedInLEFT( p_matImageLEFT,
                                                                                                                   fSearchRange,
                                                                                                                   ptUVEstimateRIGHTROI.x,
                                                                                                                   ptUVEstimateRIGHTROI.y,
                                                                                                                   fKeyPointSizePixels,
                                                                                                                   ptUVEstimateRIGHTROI+vecKeyPointBufferSingle[0].pt,
                                                                                                                   matDescriptorRIGHT ) );
                                const CDescriptor matDescriptorLEFT( cMatchLEFT.matDescriptorCAMERA );

                                //ds check depth
                                const double dDepthMeters = cMatchLEFT.vecPointXYZCAMERA.z( );
                                if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                                {
                                    throw CExceptionNoMatchFound( "invalid depth" );
                                }

                                //ds check if the descriptor match on the right side is out of range
                                if( m_dMatchingDistanceCutoffTrackingStage1 < cv::norm( pLandmark->getLastDescriptorLEFT( ), matDescriptorLEFT, cv::NORM_HAMMING ) )
                                {
                                    throw CExceptionNoMatchFound( "triangulation descriptor mismatch" );
                                }

                                //ds latter landmark update (cannot be done before pose is optimized)
                                vecMeasurementsForStereoPosit.push_back( CSolverStereoPosit::CMatch( pLandmark, pLandmark->vecPointXYZOptimized, cMatchLEFT.vecPointXYZCAMERA, cMatchLEFT.ptUVCAMERA, ptUVEstimateRIGHT, matDescriptorLEFT, matDescriptorRIGHT ) );
                                cv::circle( p_matDisplayRIGHT, ptUVEstimateRIGHT, 4, CColorCodeBGR( 0, 255, 0 ), 1 );
                                ++m_uNumberOfTracksStage1;
                            }
                            else
                            {
                                throw CExceptionNoMatchFound( "insufficient matching distance" );
                            }
                        }
                        catch( const CExceptionNoMatchFound& p_cExceptionStage1RIGHT )
                        {
                            //std::printf( "[%06lu]<CFundamentalMatcher>(getPoseStereoPosit) landmark [%06lu] Tracking stage 1 failed: '%s' '%s'\n", p_uFrame, pLandmark->uID, p_cExceptionStage1LEFT.what( ), p_cExceptionStage1RIGHT.what( ) );

                            //ds buffer world position
                            const CPoint3DWORLD vecPointXYZ( pLandmark->vecPointXYZOptimized );

                            //ds STAGE 2: check region on LEFT
                            try
                            {
                                //ds compute search ranges (no subpixel accuracy)
                                const double dScaleSearchULEFT      = std::round( m_pCameraLEFT->getPrincipalWeightU( ptUVEstimateLEFT ) + p_dMotionScaling );
                                const double dScaleSearchVLEFT      = std::round( m_pCameraLEFT->getPrincipalWeightV( ptUVEstimateLEFT ) + p_dMotionScaling );
                                const double dSearchHalfWidthLEFT   = std::round( dScaleSearchULEFT*m_uSearchBlockSizePoseOptimization );
                                const double dSearchHalfHeightLEFT  = std::round( dScaleSearchVLEFT*m_uSearchBlockSizePoseOptimization );

                                //ds corners
                                cv::Point2f ptUpperLeftLEFT( std::max( ptUVEstimateLEFT.x-dSearchHalfWidthLEFT, 0.0 ), std::max( ptUVEstimateLEFT.y-dSearchHalfHeightLEFT, 0.0 ) );
                                cv::Point2f ptLowerRightLEFT( std::min( ptUVEstimateLEFT.x+dSearchHalfWidthLEFT, m_pCameraLEFT->m_dWidthPixels ), std::min( ptUVEstimateLEFT.y+dSearchHalfHeightLEFT, m_pCameraLEFT->m_dHeightPixels ) );

                                //ds search rectangle
                                const cv::Rect cSearchROILEFT( ptUpperLeftLEFT, ptLowerRightLEFT );
                                //cv::rectangle( p_matDisplayLEFT, cSearchROILEFT, CColorCodeBGR( 255, 0, 0 ) );

                                //ds run detection on current frame
                                std::vector< cv::KeyPoint > vecKeyPointsLEFT;
                                m_pDetector->detect( p_matImageLEFT( cSearchROILEFT ), vecKeyPointsLEFT );

                                //for( const cv::KeyPoint& cKeyPoint: vecKeyPointsLEFT )
                                //{
                                //    cv::circle( p_matDisplayLEFT, cKeyPoint.pt+ptUpperLeftLEFT, 3, CColorCodeBGR( 255, 255, 255 ), -1 );
                                //}

                                //ds if we found some features in the LEFT frame
                                if( 0 < vecKeyPointsLEFT.size( ) )
                                {
                                    //ds compute descriptors in extended ROI
                                    const float fUpperLeftU  = std::max( ptUpperLeftLEFT.x-fKeyPointSizePixelsHalf, 0.0f );
                                    const float fUpperLeftV  = std::max( ptUpperLeftLEFT.y-fKeyPointSizePixelsHalf, 0.0f );
                                    const float fLowerRightU = std::min( ptLowerRightLEFT.x+fKeyPointSizePixelsHalf, m_pCameraLEFT->m_fWidthPixels );
                                    const float fLowerRightV = std::min( ptLowerRightLEFT.y+fKeyPointSizePixelsHalf, m_pCameraLEFT->m_fHeightPixels );
                                    cv::Mat matDescriptorsLEFT;

                                    //ds adjust keypoint offsets before computing descriptors
                                    std::for_each( vecKeyPointsLEFT.begin( ), vecKeyPointsLEFT.end( ), [ &ptOffsetKeyPointHalf ]( cv::KeyPoint& cKeyPoint ){ cKeyPoint.pt += ptOffsetKeyPointHalf; } );
                                    m_pExtractor->compute( p_matImageLEFT( cv::Rect( cv::Point2f( fUpperLeftU, fUpperLeftV ), cv::Point2f( fLowerRightU, fLowerRightV ) ) ), vecKeyPointsLEFT, matDescriptorsLEFT );

                                    //ds check descriptor matches for this landmark
                                    std::vector< cv::DMatch > vecMatchesLEFT;
                                    m_pMatcher->match( pLandmark->getLastDescriptorLEFT( ), matDescriptorsLEFT, vecMatchesLEFT );

                                    //ds if we got a match and the matching distance is within the range
                                    if( 0 < vecMatchesLEFT.size( ) )
                                    {
                                        if( m_dMatchingDistanceCutoffTrackingStage2 > vecMatchesLEFT[0].distance )
                                        {
                                            const cv::Point2f ptBestMatchLEFT( vecKeyPointsLEFT[vecMatchesLEFT[0].trainIdx].pt );
                                            const cv::Point2f ptBestMatchLEFTInCamera( ptUpperLeftLEFT+ptBestMatchLEFT-ptOffsetKeyPointHalf );
                                            const CDescriptor matDescriptorLEFT( matDescriptorsLEFT.row(vecMatchesLEFT[0].trainIdx) );

                                            //ds center V
                                            const float fVReference = ptBestMatchLEFTInCamera.y-fKeyPointSizePixelsHalf;

                                            //ds if in range triangulate the point directly
                                            if( 0.0 <= fVReference )
                                            {
                                                const CMatchTriangulation cMatchRIGHT( m_pTriangulator->getPointTriangulatedInRIGHT( p_matImageRIGHT,
                                                                                                                                     std::max( 0.0f, ptBestMatchLEFTInCamera.x-fSearchRange-fKeyPointSizePixelsHalf ),
                                                                                                                                     fVReference,
                                                                                                                                     fKeyPointSizePixels,
                                                                                                                                     ptBestMatchLEFTInCamera,
                                                                                                                                     matDescriptorLEFT ) );
                                                const CDescriptor matDescriptorRIGHT( cMatchRIGHT.matDescriptorCAMERA );

                                                //ds check depth
                                                const double dDepthMeters = cMatchRIGHT.vecPointXYZCAMERA.z( );
                                                if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                                                {
                                                    throw CExceptionNoMatchFound( "invalid depth" );
                                                }

                                                //ds check if the descriptor match is acceptable
                                                if( m_dMatchingDistanceCutoffTrackingStage2 > cv::norm( pLandmark->getLastDescriptorRIGHT( ), matDescriptorRIGHT, cv::NORM_HAMMING ) )
                                                {
                                                    //ds latter landmark update (cannot be done before pose is optimized)
                                                    vecMeasurementsForStereoPosit.push_back( CSolverStereoPosit::CMatch( pLandmark, vecPointXYZ, cMatchRIGHT.vecPointXYZCAMERA, ptBestMatchLEFTInCamera, cMatchRIGHT.ptUVCAMERA, matDescriptorLEFT, matDescriptorRIGHT ) );
                                                    cv::circle( p_matDisplayLEFT, ptBestMatchLEFTInCamera, 4, CColorCodeBGR( 255, 255, 255 ), 1 );
                                                    ++m_uNumberOfTracksStage2_1;
                                                }
                                                else
                                                {
                                                    //ds try the RIGHT frame
                                                    throw CExceptionNoMatchFound( "triangulation descriptor mismatch" );
                                                }
                                            }
                                            else
                                            {
                                                //ds try the RIGHT frame
                                                throw CExceptionNoMatchFound( "out of tracking range" );
                                            }
                                        }
                                        else
                                        {
                                            //ds try the RIGHT frame
                                            throw CExceptionNoMatchFound( "descriptor mismatch" );
                                        }
                                    }
                                    else
                                    {
                                        //ds try the RIGHT frame
                                        throw CExceptionNoMatchFound( "no matches found" );
                                    }
                                }
                                else
                                {
                                    //ds try the RIGHT frame
                                    throw CExceptionNoMatchFound( "no features detected" );
                                }
                            }
                            catch( const CExceptionNoMatchFound& p_cExceptionStage2LEFT )
                            {
                                //ds try to find the landmarks on RIGHT
                                try
                                {
                                    //ds compute search ranges (no subpixel accuracy)
                                    const double dScaleSearchURIGHT      = std::round( m_pCameraRIGHT->getPrincipalWeightU( ptUVEstimateRIGHT ) + p_dMotionScaling );
                                    const double dScaleSearchVRIGHT      = std::round( m_pCameraRIGHT->getPrincipalWeightV( ptUVEstimateRIGHT ) + p_dMotionScaling );
                                    const double dSearchHalfWidthRIGHT   = std::round( dScaleSearchURIGHT*m_uSearchBlockSizePoseOptimization );
                                    const double dSearchHalfHeightRIGHT  = std::round( dScaleSearchVRIGHT*m_uSearchBlockSizePoseOptimization );

                                    //ds corners
                                    cv::Point2f ptUpperLeftRIGHT( std::max( ptUVEstimateRIGHT.x-dSearchHalfWidthRIGHT, 0.0 ), std::max( ptUVEstimateRIGHT.y-dSearchHalfHeightRIGHT, 0.0 ) );
                                    cv::Point2f ptLowerRightRIGHT( std::min( ptUVEstimateRIGHT.x+dSearchHalfWidthRIGHT, m_pCameraRIGHT->m_dWidthPixels ), std::min( ptUVEstimateRIGHT.y+dSearchHalfHeightRIGHT, m_pCameraRIGHT->m_dHeightPixels ) );

                                    //ds search rectangle
                                    const cv::Rect cSearchROIRIGHT( ptUpperLeftRIGHT, ptLowerRightRIGHT );
                                    //cv::rectangle( p_matDisplayRIGHT, cSearchROIRIGHT, CColorCodeBGR( 255, 0, 0 ) );

                                    //ds run detection on current frame
                                    std::vector< cv::KeyPoint > vecKeyPointsRIGHT;
                                    m_pDetector->detect( p_matImageRIGHT( cSearchROIRIGHT ), vecKeyPointsRIGHT );

                                    //for( const cv::KeyPoint& cKeyPoint: vecKeyPointsRIGHT )
                                    //{
                                    //    cv::circle( p_matDisplayLEFT, cKeyPoint.pt+ptUpperLeftRIGHT, 3, CColorCodeBGR( 255, 255, 255 ), -1 );
                                    //}

                                    //ds if we found some features in the RIGHT frame
                                    if( 0 < vecKeyPointsRIGHT.size( ) )
                                    {
                                        //ds compute descriptors in extended ROI
                                        const float fUpperLeftU  = std::max( ptUpperLeftRIGHT.x-fKeyPointSizePixelsHalf, 0.0f );
                                        const float fUpperLeftV  = std::max( ptUpperLeftRIGHT.y-fKeyPointSizePixelsHalf, 0.0f );
                                        const float fLowerRightU = std::min( ptLowerRightRIGHT.x+fKeyPointSizePixelsHalf, m_pCameraRIGHT->m_fWidthPixels );
                                        const float fLowerRightV = std::min( ptLowerRightRIGHT.y+fKeyPointSizePixelsHalf, m_pCameraRIGHT->m_fHeightPixels );
                                        cv::Mat matDescriptorsRIGHT;

                                        //ds adjust keypoint offsets before computing descriptors
                                        std::for_each( vecKeyPointsRIGHT.begin( ), vecKeyPointsRIGHT.end( ), [ &ptOffsetKeyPointHalf ]( cv::KeyPoint& cKeyPoint ){ cKeyPoint.pt += ptOffsetKeyPointHalf; } );
                                        m_pExtractor->compute( p_matImageRIGHT( cv::Rect( cv::Point2f( fUpperLeftU, fUpperLeftV ), cv::Point2f( fLowerRightU, fLowerRightV ) ) ), vecKeyPointsRIGHT, matDescriptorsRIGHT );

                                        //ds check descriptor matches for this landmark
                                        std::vector< cv::DMatch > vecMatchesRIGHT;
                                        m_pMatcher->match( pLandmark->getLastDescriptorRIGHT( ), matDescriptorsRIGHT, vecMatchesRIGHT );

                                        //ds if we got a match and the matching distance is within the range
                                        if( 0 < vecMatchesRIGHT.size( ) )
                                        {
                                            if( m_dMatchingDistanceCutoffTrackingStage2 > vecMatchesRIGHT[0].distance )
                                            {
                                                const cv::Point2f ptBestMatchRIGHT( vecKeyPointsRIGHT[vecMatchesRIGHT[0].trainIdx].pt );
                                                const cv::Point2f ptBestMatchRIGHTInCamera( ptUpperLeftRIGHT+ptBestMatchRIGHT-ptOffsetKeyPointHalf );
                                                const CDescriptor matDescriptorRIGHT( matDescriptorsRIGHT.row(vecMatchesRIGHT[0].trainIdx) );

                                                //ds center V
                                                const float fVReference = ptBestMatchRIGHTInCamera.y-fKeyPointSizePixelsHalf;

                                                //ds if in range triangulate the point directly
                                                if( 0.0 <= fVReference )
                                                {
                                                    const CMatchTriangulation cMatchLEFT( m_pTriangulator->getPointTriangulatedInLEFT( p_matImageLEFT,
                                                                                                                                       fSearchRange,
                                                                                                                                       std::max( 0.0f, ptBestMatchRIGHTInCamera.x-fKeyPointSizePixelsHalf ),
                                                                                                                                       fVReference,
                                                                                                                                       fKeyPointSizePixels,
                                                                                                                                       ptBestMatchRIGHTInCamera,
                                                                                                                                       matDescriptorRIGHT ) );
                                                    const CDescriptor matDescriptorLEFT( cMatchLEFT.matDescriptorCAMERA );

                                                    //ds check depth
                                                    const double dDepthMeters = cMatchLEFT.vecPointXYZCAMERA.z( );
                                                    if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                                                    {
                                                        throw CExceptionNoMatchFound( "invalid depth" );
                                                    }

                                                    //ds check if the descriptor match is acceptable
                                                    if( m_dMatchingDistanceCutoffTrackingStage2 > cv::norm( pLandmark->getLastDescriptorLEFT( ), matDescriptorLEFT, cv::NORM_HAMMING ) )
                                                    {
                                                        //ds latter landmark update (cannot be done before pose is optimized)
                                                        vecMeasurementsForStereoPosit.push_back( CSolverStereoPosit::CMatch( pLandmark, vecPointXYZ, cMatchLEFT.vecPointXYZCAMERA, cMatchLEFT.ptUVCAMERA, ptBestMatchRIGHTInCamera, matDescriptorLEFT, matDescriptorRIGHT ) );
                                                        cv::circle( p_matDisplayRIGHT, ptBestMatchRIGHTInCamera, 4, CColorCodeBGR( 255, 255, 255 ), 1 );
                                                        ++m_uNumberOfTracksStage2_1;
                                                    }
                                                    else
                                                    {
                                                        throw CExceptionNoMatchFound( "triangulation descriptor mismatch" );
                                                    }
                                                }
                                                else
                                                {
                                                    throw CExceptionNoMatchFound( "out of tracking range" );
                                                }
                                            }
                                            else
                                            {
                                                throw CExceptionNoMatchFound( "descriptor mismatch" );
                                            }
                                        }
                                        else
                                        {
                                            throw CExceptionNoMatchFound( "no matches found" );
                                        }
                                    }
                                    else
                                    {
                                        throw CExceptionNoMatchFound( "no features detected" );
                                    }
                                }
                                catch( const CExceptionNoMatchFound& p_cExceptionStage2RIGHT )
                                {
                                    //std::printf( "[%06lu]<CFundamentalMatcher>(getPoseStereoPosit) landmark [%06lu] Tracking stage 2 failed: '%s' '%s'\n", p_uFrame, pLandmark->uID, p_cExceptionStage2LEFT.what( ), p_cExceptionStage2RIGHT.what( ) );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    //ds timing
    m_dDurationTotalSecondsRegionalTracking += CTimer::getTimeSeconds( )-dTimeStartSeconds;

    //ds call optimizer - exception on failure
    const Eigen::Isometry3d matTransformationWORLDtoLEFT = m_cSolverSterePosit.getTransformationWORLDtoLEFT( p_matTransformationWORLDtoLEFTLAST,
                                                                                                             p_vecTranslationIMU,
                                                                                                             p_matTransformationEstimateWORLDtoLEFT,
                                                                                                             vecMeasurementsForStereoPosit );

    //ds precompute intrinsics
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*matTransformationWORLDtoLEFT.matrix( ) );
    const MatrixProjection matProjectionWORLDtoRIGHT( m_pCameraRIGHT->m_matProjection*matTransformationWORLDtoLEFT.matrix( ) );
    const Eigen::Isometry3d matTransformationLEFTtoWORLD( matTransformationWORLDtoLEFT.inverse( ) );

    //ds update all visible landmarks
    for( const CSolverStereoPosit::CMatch& cMatchSTEREO: vecMeasurementsForStereoPosit )
    {
        _addMeasurementToLandmarkSTEREO( p_uFrame, cMatchSTEREO, matTransformationLEFTtoWORLD, matTransformationWORLDtoLEFT, matProjectionWORLDtoLEFT, matProjectionWORLDtoRIGHT );
    }

    //ds return with pose
    return matTransformationWORLDtoLEFT;
}

const Eigen::Isometry3d CFundamentalMatcher::getPoseStereoPositfromLAST( const UIDFrame p_uFrame,
                                                                         cv::Mat& p_matDisplayLEFT,
                                                                         cv::Mat& p_matDisplayRIGHT,
                                                                         const cv::Mat& p_matImageLEFT,
                                                                         const cv::Mat& p_matImageRIGHT,
                                                                         const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                                                                         const Eigen::Isometry3d& p_matTransformationWORLDtoLEFTLAST,
                                                                         const Eigen::Vector3d& p_vecRotationIMU,
                                                                         const Eigen::Vector3d& p_vecTranslationIMU,
                                                                         const double& p_dMotionScaling )
{
    assert( 1.0 <= p_dMotionScaling );

    //ds found landmarks in this frame
    std::vector< CSolverStereoPosit::CMatch > vecMeasurementsForStereoPosit;

    //ds single keypoint buffer
    std::vector< cv::KeyPoint > vecKeyPointBufferSingle( 1 );

    //ds timing
    const double dTimeStartSeconds = CTimer::getTimeSeconds( );

    //ds info
    m_uNumberOfTracksStage1   = 0;
    m_uNumberOfTracksStage2_1 = 0;

    //ds triangulation search length scaling
    const float fTriangulationScale = 1.0+p_dMotionScaling;

    //ds 2 STAGE tracking algorithm
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan (use all points not only optimized and visible ones)
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds compute current reprojection point
            const cv::Point2f ptUVEstimateLEFT( pLandmark->getLastDetectionLEFT( ) );
            const cv::Point2f ptUVEstimateRIGHT( pLandmark->getLastDetectionRIGHT( ) );
            assert( ptUVEstimateLEFT.y == ptUVEstimateRIGHT.y );

            const float fKeyPointSizePixels       = pLandmark->dKeyPointSize;
            const float fKeyPointSizePixelsHalf   = 4*fKeyPointSizePixels;
            const float fKeyPointSizePixelsLength = 8*fKeyPointSizePixels+1;
            const cv::Point2f ptOffsetKeyPointHalf( fKeyPointSizePixelsHalf, fKeyPointSizePixelsHalf );
            const float fSearchRange = fTriangulationScale*pLandmark->getLastDisparity( );

            //ds check if we are in tracking range
            if( m_pCameraLEFT->m_cFieldOfView.contains( ptUVEstimateLEFT ) && m_pCameraRIGHT->m_cFieldOfView.contains( ptUVEstimateRIGHT ) )
            {
                //ds STAGE 1: check LEFT
                try
                {
                    //ds LEFT roi
                    const cv::Point2f ptUVEstimateLEFTROI( ptUVEstimateLEFT-ptOffsetKeyPointHalf );
                    const cv::Rect matROI( ptUVEstimateLEFTROI.x, ptUVEstimateLEFTROI.y, fKeyPointSizePixelsLength, fKeyPointSizePixelsLength );

                    //ds compute descriptor at this point
                    vecKeyPointBufferSingle[0] = cv::KeyPoint( ptOffsetKeyPointHalf, pLandmark->dKeyPointSize );
                    cv::Mat matDescriptorLEFT;
                    m_pExtractor->compute( p_matImageLEFT( matROI ), vecKeyPointBufferSingle, matDescriptorLEFT );

                    //ds if acceptable
                    if( 1 == matDescriptorLEFT.rows && m_dMatchingDistanceCutoffTrackingStage1 > cv::norm( pLandmark->getLastDescriptorLEFT( ), matDescriptorLEFT, cv::NORM_HAMMING ) )
                    {
                        //ds triangulate the point directly
                        const CMatchTriangulation cMatchRIGHT( m_pTriangulator->getPointTriangulatedInRIGHT( p_matImageRIGHT,
                                                                                                             std::max( 0.0f, ptUVEstimateLEFTROI.x-fSearchRange ),
                                                                                                             ptUVEstimateLEFTROI.y,
                                                                                                             fKeyPointSizePixels,
                                                                                                             ptUVEstimateLEFTROI+vecKeyPointBufferSingle[0].pt,
                                                                                                             matDescriptorLEFT ) );
                        const CDescriptor matDescriptorRIGHT( cMatchRIGHT.matDescriptorCAMERA );

                        //ds check depth
                        const double dDepthMeters = cMatchRIGHT.vecPointXYZCAMERA.z( );
                        if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                        {
                            throw CExceptionNoMatchFound( "invalid depth" );
                        }

                        //ds check if the descriptor match on the right side is out of range
                        if( m_dMatchingDistanceCutoffTrackingStage1 < cv::norm( pLandmark->getLastDescriptorRIGHT( ), matDescriptorRIGHT, cv::NORM_HAMMING ) )
                        {
                            throw CExceptionNoMatchFound( "triangulation descriptor mismatch" );
                        }

                        //ds latter landmark update (cannot be done before pose is optimized)
                        vecMeasurementsForStereoPosit.push_back( CSolverStereoPosit::CMatch( pLandmark, pLandmark->vecPointXYZOptimized, cMatchRIGHT.vecPointXYZCAMERA, ptUVEstimateLEFT, cMatchRIGHT.ptUVCAMERA, matDescriptorLEFT, matDescriptorRIGHT ) );
                        cv::circle( p_matDisplayLEFT, ptUVEstimateLEFT, 4, CColorCodeBGR( 0, 255, 0 ), 1 );
                        ++m_uNumberOfTracksStage1;
                    }
                    else
                    {
                        throw CExceptionNoMatchFound( "insufficient matching distance" );
                    }
                }
                catch( const CExceptionNoMatchFound& p_cExceptionStage1LEFT )
                {
                    //ds STAGE 1: check RIGHT
                    try
                    {
                        //ds RIGHT roi
                        const cv::Point2f ptUVEstimateRIGHTROI( ptUVEstimateRIGHT-ptOffsetKeyPointHalf );
                        const cv::Rect matROI( ptUVEstimateRIGHTROI.x, ptUVEstimateRIGHTROI.y, fKeyPointSizePixelsLength, fKeyPointSizePixelsLength );

                        //ds compute descriptor at this point
                        vecKeyPointBufferSingle[0] = cv::KeyPoint( ptOffsetKeyPointHalf, pLandmark->dKeyPointSize );
                        cv::Mat matDescriptorRIGHT;
                        m_pExtractor->compute( p_matImageRIGHT( matROI ), vecKeyPointBufferSingle, matDescriptorRIGHT );

                        //ds if acceptable
                        if( 1 == matDescriptorRIGHT.rows && m_dMatchingDistanceCutoffTrackingStage1 > cv::norm( pLandmark->getLastDescriptorRIGHT( ), matDescriptorRIGHT, cv::NORM_HAMMING ) )
                        {
                            //ds triangulate the point directly
                            const CMatchTriangulation cMatchLEFT( m_pTriangulator->getPointTriangulatedInLEFT( p_matImageLEFT,
                                                                                                               fSearchRange,
                                                                                                               ptUVEstimateRIGHTROI.x,
                                                                                                               ptUVEstimateRIGHTROI.y,
                                                                                                               fKeyPointSizePixels,
                                                                                                               ptUVEstimateRIGHTROI+vecKeyPointBufferSingle[0].pt,
                                                                                                               matDescriptorRIGHT ) );
                            const CDescriptor matDescriptorLEFT( cMatchLEFT.matDescriptorCAMERA );

                            //ds check depth
                            const double dDepthMeters = cMatchLEFT.vecPointXYZCAMERA.z( );
                            if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                            {
                                throw CExceptionNoMatchFound( "invalid depth" );
                            }

                            //ds check if the descriptor match on the right side is out of range
                            if( m_dMatchingDistanceCutoffTrackingStage1 < cv::norm( pLandmark->getLastDescriptorLEFT( ), matDescriptorLEFT, cv::NORM_HAMMING ) )
                            {
                                throw CExceptionNoMatchFound( "triangulation descriptor mismatch" );
                            }

                            //ds latter landmark update (cannot be done before pose is optimized)
                            vecMeasurementsForStereoPosit.push_back( CSolverStereoPosit::CMatch( pLandmark, pLandmark->vecPointXYZOptimized, cMatchLEFT.vecPointXYZCAMERA, cMatchLEFT.ptUVCAMERA, ptUVEstimateRIGHT, matDescriptorLEFT, matDescriptorRIGHT ) );
                            cv::circle( p_matDisplayRIGHT, ptUVEstimateRIGHT, 4, CColorCodeBGR( 0, 255, 0 ), 1 );
                            ++m_uNumberOfTracksStage1;
                        }
                        else
                        {
                            throw CExceptionNoMatchFound( "insufficient matching distance" );
                        }
                    }
                    catch( const CExceptionNoMatchFound& p_cExceptionStage1RIGHT )
                    {
                        //std::printf( "[%06lu]<CFundamentalMatcher>(getPoseStereoPosit) landmark [%06lu] Tracking stage 1 failed: '%s' '%s'\n", p_uFrame, pLandmark->uID, p_cExceptionStage1LEFT.what( ), p_cExceptionStage1RIGHT.what( ) );

                        //ds buffer world position
                        const CPoint3DWORLD vecPointXYZ( pLandmark->vecPointXYZOptimized );

                        //ds STAGE 2: check region on LEFT
                        try
                        {
                            //ds compute search ranges (no subpixel accuracy)
                            const double dScaleSearchULEFT      = std::round( m_pCameraLEFT->getPrincipalWeightU( ptUVEstimateLEFT ) + p_dMotionScaling );
                            const double dScaleSearchVLEFT      = std::round( m_pCameraLEFT->getPrincipalWeightV( ptUVEstimateLEFT ) + p_dMotionScaling );
                            const double dSearchHalfWidthLEFT   = std::round( dScaleSearchULEFT*m_uSearchBlockSizePoseOptimization );
                            const double dSearchHalfHeightLEFT  = std::round( dScaleSearchVLEFT*m_uSearchBlockSizePoseOptimization );

                            //ds corners
                            cv::Point2f ptUpperLeftLEFT( std::max( ptUVEstimateLEFT.x-dSearchHalfWidthLEFT, 0.0 ), std::max( ptUVEstimateLEFT.y-dSearchHalfHeightLEFT, 0.0 ) );
                            cv::Point2f ptLowerRightLEFT( std::min( ptUVEstimateLEFT.x+dSearchHalfWidthLEFT, m_pCameraLEFT->m_dWidthPixels ), std::min( ptUVEstimateLEFT.y+dSearchHalfHeightLEFT, m_pCameraLEFT->m_dHeightPixels ) );

                            //ds search rectangle
                            const cv::Rect cSearchROILEFT( ptUpperLeftLEFT, ptLowerRightLEFT );
                            //cv::rectangle( p_matDisplayLEFT, cSearchROILEFT, CColorCodeBGR( 255, 0, 0 ) );

                            //ds run detection on current frame
                            std::vector< cv::KeyPoint > vecKeyPointsLEFT;
                            m_pDetector->detect( p_matImageLEFT( cSearchROILEFT ), vecKeyPointsLEFT );

                            //for( const cv::KeyPoint& cKeyPoint: vecKeyPointsLEFT )
                            //{
                            //    cv::circle( p_matDisplayLEFT, cKeyPoint.pt+ptUpperLeftLEFT, 3, CColorCodeBGR( 255, 255, 255 ), -1 );
                            //}

                            //ds if we found some features in the LEFT frame
                            if( 0 < vecKeyPointsLEFT.size( ) )
                            {
                                //ds compute descriptors in extended ROI
                                const float fUpperLeftU  = std::max( ptUpperLeftLEFT.x-fKeyPointSizePixelsHalf, 0.0f );
                                const float fUpperLeftV  = std::max( ptUpperLeftLEFT.y-fKeyPointSizePixelsHalf, 0.0f );
                                const float fLowerRightU = std::min( ptLowerRightLEFT.x+fKeyPointSizePixelsHalf, m_pCameraLEFT->m_fWidthPixels );
                                const float fLowerRightV = std::min( ptLowerRightLEFT.y+fKeyPointSizePixelsHalf, m_pCameraLEFT->m_fHeightPixels );
                                cv::Mat matDescriptorsLEFT;

                                //ds adjust keypoint offsets before computing descriptors
                                std::for_each( vecKeyPointsLEFT.begin( ), vecKeyPointsLEFT.end( ), [ &ptOffsetKeyPointHalf ]( cv::KeyPoint& cKeyPoint ){ cKeyPoint.pt += ptOffsetKeyPointHalf; } );
                                m_pExtractor->compute( p_matImageLEFT( cv::Rect( cv::Point2f( fUpperLeftU, fUpperLeftV ), cv::Point2f( fLowerRightU, fLowerRightV ) ) ), vecKeyPointsLEFT, matDescriptorsLEFT );

                                //ds check descriptor matches for this landmark
                                std::vector< cv::DMatch > vecMatchesLEFT;
                                m_pMatcher->match( pLandmark->getLastDescriptorLEFT( ), matDescriptorsLEFT, vecMatchesLEFT );

                                //ds if we got a match and the matching distance is within the range
                                if( 0 < vecMatchesLEFT.size( ) )
                                {
                                    if( m_dMatchingDistanceCutoffTrackingStage2 > vecMatchesLEFT[0].distance )
                                    {
                                        const cv::Point2f ptBestMatchLEFT( vecKeyPointsLEFT[vecMatchesLEFT[0].trainIdx].pt );
                                        const cv::Point2f ptBestMatchLEFTInCamera( ptUpperLeftLEFT+ptBestMatchLEFT-ptOffsetKeyPointHalf );
                                        const CDescriptor matDescriptorLEFT( matDescriptorsLEFT.row(vecMatchesLEFT[0].trainIdx) );

                                        //ds center V
                                        const float fVReference = ptBestMatchLEFTInCamera.y-fKeyPointSizePixelsHalf;

                                        //ds if in range triangulate the point directly
                                        if( 0.0 <= fVReference )
                                        {
                                            const CMatchTriangulation cMatchRIGHT( m_pTriangulator->getPointTriangulatedInRIGHT( p_matImageRIGHT,
                                                                                                                                 std::max( 0.0f, ptBestMatchLEFTInCamera.x-fSearchRange-fKeyPointSizePixelsHalf ),
                                                                                                                                 fVReference,
                                                                                                                                 fKeyPointSizePixels,
                                                                                                                                 ptBestMatchLEFTInCamera,
                                                                                                                                 matDescriptorLEFT ) );
                                            const CDescriptor matDescriptorRIGHT( cMatchRIGHT.matDescriptorCAMERA );

                                            //ds check depth
                                            const double dDepthMeters = cMatchRIGHT.vecPointXYZCAMERA.z( );
                                            if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                                            {
                                                throw CExceptionNoMatchFound( "invalid depth" );
                                            }

                                            //ds check if the descriptor match is acceptable
                                            if( m_dMatchingDistanceCutoffTrackingStage2 > cv::norm( pLandmark->getLastDescriptorRIGHT( ), matDescriptorRIGHT, cv::NORM_HAMMING ) )
                                            {
                                                //ds latter landmark update (cannot be done before pose is optimized)
                                                vecMeasurementsForStereoPosit.push_back( CSolverStereoPosit::CMatch( pLandmark, vecPointXYZ, cMatchRIGHT.vecPointXYZCAMERA, ptBestMatchLEFTInCamera, cMatchRIGHT.ptUVCAMERA, matDescriptorLEFT, matDescriptorRIGHT ) );
                                                cv::circle( p_matDisplayLEFT, ptBestMatchLEFTInCamera, 4, CColorCodeBGR( 255, 255, 255 ), 1 );
                                                ++m_uNumberOfTracksStage2_1;
                                            }
                                            else
                                            {
                                                //ds try the RIGHT frame
                                                throw CExceptionNoMatchFound( "triangulation descriptor mismatch" );
                                            }
                                        }
                                        else
                                        {
                                            //ds try the RIGHT frame
                                            throw CExceptionNoMatchFound( "out of tracking range" );
                                        }
                                    }
                                    else
                                    {
                                        //ds try the RIGHT frame
                                        throw CExceptionNoMatchFound( "descriptor mismatch" );
                                    }
                                }
                                else
                                {
                                    //ds try the RIGHT frame
                                    throw CExceptionNoMatchFound( "no matches found" );
                                }
                            }
                            else
                            {
                                //ds try the RIGHT frame
                                throw CExceptionNoMatchFound( "no features detected" );
                            }
                        }
                        catch( const CExceptionNoMatchFound& p_cExceptionStage2LEFT )
                        {
                            //ds try to find the landmarks on RIGHT
                            try
                            {
                                //ds compute search ranges (no subpixel accuracy)
                                const double dScaleSearchURIGHT      = std::round( m_pCameraRIGHT->getPrincipalWeightU( ptUVEstimateRIGHT ) + p_dMotionScaling );
                                const double dScaleSearchVRIGHT      = std::round( m_pCameraRIGHT->getPrincipalWeightV( ptUVEstimateRIGHT ) + p_dMotionScaling );
                                const double dSearchHalfWidthRIGHT   = std::round( dScaleSearchURIGHT*m_uSearchBlockSizePoseOptimization );
                                const double dSearchHalfHeightRIGHT  = std::round( dScaleSearchVRIGHT*m_uSearchBlockSizePoseOptimization );

                                //ds corners
                                cv::Point2f ptUpperLeftRIGHT( std::max( ptUVEstimateRIGHT.x-dSearchHalfWidthRIGHT, 0.0 ), std::max( ptUVEstimateRIGHT.y-dSearchHalfHeightRIGHT, 0.0 ) );
                                cv::Point2f ptLowerRightRIGHT( std::min( ptUVEstimateRIGHT.x+dSearchHalfWidthRIGHT, m_pCameraRIGHT->m_dWidthPixels ), std::min( ptUVEstimateRIGHT.y+dSearchHalfHeightRIGHT, m_pCameraRIGHT->m_dHeightPixels ) );

                                //ds search rectangle
                                const cv::Rect cSearchROIRIGHT( ptUpperLeftRIGHT, ptLowerRightRIGHT );
                                //cv::rectangle( p_matDisplayRIGHT, cSearchROIRIGHT, CColorCodeBGR( 255, 0, 0 ) );

                                //ds run detection on current frame
                                std::vector< cv::KeyPoint > vecKeyPointsRIGHT;
                                m_pDetector->detect( p_matImageRIGHT( cSearchROIRIGHT ), vecKeyPointsRIGHT );

                                //for( const cv::KeyPoint& cKeyPoint: vecKeyPointsRIGHT )
                                //{
                                //    cv::circle( p_matDisplayLEFT, cKeyPoint.pt+ptUpperLeftRIGHT, 3, CColorCodeBGR( 255, 255, 255 ), -1 );
                                //}

                                //ds if we found some features in the RIGHT frame
                                if( 0 < vecKeyPointsRIGHT.size( ) )
                                {
                                    //ds compute descriptors in extended ROI
                                    const float fUpperLeftU  = std::max( ptUpperLeftRIGHT.x-fKeyPointSizePixelsHalf, 0.0f );
                                    const float fUpperLeftV  = std::max( ptUpperLeftRIGHT.y-fKeyPointSizePixelsHalf, 0.0f );
                                    const float fLowerRightU = std::min( ptLowerRightRIGHT.x+fKeyPointSizePixelsHalf, m_pCameraRIGHT->m_fWidthPixels );
                                    const float fLowerRightV = std::min( ptLowerRightRIGHT.y+fKeyPointSizePixelsHalf, m_pCameraRIGHT->m_fHeightPixels );
                                    cv::Mat matDescriptorsRIGHT;

                                    //ds adjust keypoint offsets before computing descriptors
                                    std::for_each( vecKeyPointsRIGHT.begin( ), vecKeyPointsRIGHT.end( ), [ &ptOffsetKeyPointHalf ]( cv::KeyPoint& cKeyPoint ){ cKeyPoint.pt += ptOffsetKeyPointHalf; } );
                                    m_pExtractor->compute( p_matImageRIGHT( cv::Rect( cv::Point2f( fUpperLeftU, fUpperLeftV ), cv::Point2f( fLowerRightU, fLowerRightV ) ) ), vecKeyPointsRIGHT, matDescriptorsRIGHT );

                                    //ds check descriptor matches for this landmark
                                    std::vector< cv::DMatch > vecMatchesRIGHT;
                                    m_pMatcher->match( pLandmark->getLastDescriptorRIGHT( ), matDescriptorsRIGHT, vecMatchesRIGHT );

                                    //ds if we got a match and the matching distance is within the range
                                    if( 0 < vecMatchesRIGHT.size( ) )
                                    {
                                        if( m_dMatchingDistanceCutoffTrackingStage2 > vecMatchesRIGHT[0].distance )
                                        {
                                            const cv::Point2f ptBestMatchRIGHT( vecKeyPointsRIGHT[vecMatchesRIGHT[0].trainIdx].pt );
                                            const cv::Point2f ptBestMatchRIGHTInCamera( ptUpperLeftRIGHT+ptBestMatchRIGHT-ptOffsetKeyPointHalf );
                                            const CDescriptor matDescriptorRIGHT( matDescriptorsRIGHT.row(vecMatchesRIGHT[0].trainIdx) );

                                            //ds center V
                                            const float fVReference = ptBestMatchRIGHTInCamera.y-fKeyPointSizePixelsHalf;

                                            //ds if in range triangulate the point directly
                                            if( 0.0 <= fVReference )
                                            {
                                                const CMatchTriangulation cMatchLEFT( m_pTriangulator->getPointTriangulatedInLEFT( p_matImageLEFT,
                                                                                                                                   fSearchRange,
                                                                                                                                   std::max( 0.0f, ptBestMatchRIGHTInCamera.x-fKeyPointSizePixelsHalf ),
                                                                                                                                   fVReference,
                                                                                                                                   fKeyPointSizePixels,
                                                                                                                                   ptBestMatchRIGHTInCamera,
                                                                                                                                   matDescriptorRIGHT ) );
                                                const CDescriptor matDescriptorLEFT( cMatchLEFT.matDescriptorCAMERA );

                                                //ds check depth
                                                const double dDepthMeters = cMatchLEFT.vecPointXYZCAMERA.z( );
                                                if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                                                {
                                                    throw CExceptionNoMatchFound( "invalid depth" );
                                                }

                                                //ds check if the descriptor match is acceptable
                                                if( m_dMatchingDistanceCutoffTrackingStage2 > cv::norm( pLandmark->getLastDescriptorLEFT( ), matDescriptorLEFT, cv::NORM_HAMMING ) )
                                                {
                                                    //ds latter landmark update (cannot be done before pose is optimized)
                                                    vecMeasurementsForStereoPosit.push_back( CSolverStereoPosit::CMatch( pLandmark, vecPointXYZ, cMatchLEFT.vecPointXYZCAMERA, cMatchLEFT.ptUVCAMERA, ptBestMatchRIGHTInCamera, matDescriptorLEFT, matDescriptorRIGHT ) );
                                                    cv::circle( p_matDisplayRIGHT, ptBestMatchRIGHTInCamera, 4, CColorCodeBGR( 255, 255, 255 ), 1 );
                                                    ++m_uNumberOfTracksStage2_1;
                                                }
                                                else
                                                {
                                                    throw CExceptionNoMatchFound( "triangulation descriptor mismatch" );
                                                }
                                            }
                                            else
                                            {
                                                throw CExceptionNoMatchFound( "out of tracking range" );
                                            }
                                        }
                                        else
                                        {
                                            throw CExceptionNoMatchFound( "descriptor mismatch" );
                                        }
                                    }
                                    else
                                    {
                                        throw CExceptionNoMatchFound( "no matches found" );
                                    }
                                }
                                else
                                {
                                    throw CExceptionNoMatchFound( "no features detected" );
                                }
                            }
                            catch( const CExceptionNoMatchFound& p_cExceptionStage2RIGHT )
                            {
                                //std::printf( "[%06lu]<CFundamentalMatcher>(getPoseStereoPosit) landmark [%06lu] Tracking stage 2 failed: '%s' '%s'\n", p_uFrame, pLandmark->uID, p_cExceptionStage2LEFT.what( ), p_cExceptionStage2RIGHT.what( ) );
                            }
                        }
                    }
                }
            }
        }
    }

    //ds timing
    m_dDurationTotalSecondsRegionalTracking += CTimer::getTimeSeconds( )-dTimeStartSeconds;

    //ds call optimizer - exception on failure
    const Eigen::Isometry3d matTransformationWORLDtoLEFT = m_cSolverSterePosit.getTransformationWORLDtoLEFT( p_matTransformationWORLDtoLEFTLAST,
                                                                                                             p_vecTranslationIMU,
                                                                                                             p_matTransformationEstimateWORLDtoLEFT,
                                                                                                             vecMeasurementsForStereoPosit );

    //ds precompute intrinsics
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*matTransformationWORLDtoLEFT.matrix( ) );
    const MatrixProjection matProjectionWORLDtoRIGHT( m_pCameraRIGHT->m_matProjection*matTransformationWORLDtoLEFT.matrix( ) );
    const Eigen::Isometry3d matTransformationLEFTtoWORLD( matTransformationWORLDtoLEFT.inverse( ) );

    //ds update all visible landmarks
    for( const CSolverStereoPosit::CMatch& cMatchSTEREO: vecMeasurementsForStereoPosit )
    {
        _addMeasurementToLandmarkSTEREO( p_uFrame, cMatchSTEREO, matTransformationLEFTtoWORLD, matTransformationWORLDtoLEFT, matProjectionWORLDtoLEFT, matProjectionWORLDtoRIGHT );
    }

    //ds return with pose
    return matTransformationWORLDtoLEFT;
}

//ds locking landmarks in upper scope
void CFundamentalMatcher::trackEpipolar( const UIDFrame p_uFrame,
                                        const cv::Mat& p_matImageLEFT,
                                        const cv::Mat& p_matImageRIGHT,
                                        const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                        const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                        const double& p_dMotionScaling,
                                        cv::Mat& p_matDisplayLEFT,
                                        cv::Mat& p_matDisplayRIGHT )
{
    assert( 1.0 <= p_dMotionScaling );

    //ds precompute intrinsics
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*p_matTransformationWORLDtoLEFT.matrix( ) );
    const MatrixProjection matProjectionWORLDtoRIGHT( m_pCameraRIGHT->m_matProjection*p_matTransformationWORLDtoLEFT.matrix( ) );

    //ds new active measurement points
    std::vector< CDetectionPoint > vecDetectionPointsActive;

    //ds compute initial sampling line
    const double dHalfLineLength = p_dMotionScaling*10;

    //ds triangulation search length scaling
    const float fTriangulationScale = 1.0+p_dMotionScaling;

    //ds total detections counter
    m_uNumberOfTracksStage3   = 0;
    m_uNumberOfTracksStage2_2 = 0;
    UIDLandmark uNumberOfFailedLandmarkOptimizations = 0;
    UIDLandmark uNumberOfInvalidLandmarks = 0;
    m_vecMeasurementsVisible.clear( );

    //ds timing
    const double dTimeStartSeconds = CTimer::getTimeSeconds( );

    //ds active measurements
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds visible (=in this image detected) active (=not detected in this image but failed detections below threshold)
        std::shared_ptr< std::vector< CLandmark* > >vecActiveLandmarksPerDetectionPoint( std::make_shared< std::vector< CLandmark* > >( ) );

        //ds check relative transform
        const Eigen::Isometry3d matTransformationToNow( p_matTransformationWORLDtoLEFT*cDetectionPoint.matTransformationLEFTtoWORLD );

        //ds compute essential matrix for this detection point
        const Eigen::Matrix3d matRotation( matTransformationToNow.linear( ) );
        const Eigen::Vector3d vecTranslation( matTransformationToNow.translation( ) );
        const Eigen::Matrix3d matEssential( matRotation*CMiniVisionToolbox::getSkew( vecTranslation ) );
        const Eigen::Matrix3d matFundamental( m_pCameraLEFT->m_matIntrinsicPInverseTransposed*matEssential*m_pCameraLEFT->m_matIntrinsicPInverse );

        //ds loop over all points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds check if we can skip this landmark due to failed optimization
            if( 0 < pLandmark->uOptimizationsFailed )
            {
                cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), 4, CColorCodeBGR( 0, 0, 255 ), -1 );
                cv::circle( p_matDisplayRIGHT, pLandmark->getLastDetectionRIGHT( ), 4, CColorCodeBGR( 0, 0, 255 ), -1 );
                pLandmark->bIsCurrentlyVisible = false;
                ++uNumberOfFailedLandmarkOptimizations;
            }

            //ds check if we can skip this landmark due to invalid optimization (at least one time optimized but currently failed)
            else if( 0 < pLandmark->uOptimizationsSuccessful && !pLandmark->bIsOptimal )
            {
                cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), 4, CColorCodeBGR( 0, 255, 255 ), -1 );
                cv::circle( p_matDisplayRIGHT, pLandmark->getLastDetectionRIGHT( ), 4, CColorCodeBGR( 0, 255, 255 ), -1 );
                pLandmark->bIsCurrentlyVisible = false;
                ++uNumberOfInvalidLandmarks;
            }

            //ds process the landmark
            else
            {
                //ds check if already detected (in pose optimization)
                if( pLandmark->bIsCurrentlyVisible )
                {
                    //ds just register the measurement
                    m_vecMeasurementsVisible.push_back( pLandmark->getLastMeasurement( ) );
                    vecActiveLandmarksPerDetectionPoint->push_back( pLandmark );
                }
                else
                {
                    //ds if there was a translation (else the essential matrix is undefined)
                    if( 0.0 < vecTranslation.squaredNorm( ) )
                    {
                        //ds projection from triangulation to estimate epipolar line drawing TODO: remove cast
                        const cv::Point2f ptProjection( m_pCameraLEFT->getProjectionRounded( static_cast< CPoint3DWORLD >( p_matTransformationWORLDtoLEFT*pLandmark->vecPointXYZOptimized ) ) );

                        try
                        {
                            //ds check if the projection is out of the fov
                            if( !m_pCameraLEFT->m_cFieldOfView.contains( ptProjection ) )
                            {
                                throw CExceptionEpipolarLine( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) projection out of sight" );
                            }

                            //ds compute the projection of the point (line) in the current frame
                            const Eigen::Vector3d vecCoefficients( matFundamental*pLandmark->vecUVReferenceLEFT );

                            //ds line length for this projection based on principal weighting
                            const double dHalfLineLengthU = m_dEpipolarLineBaseLength + m_pCameraLEFT->getPrincipalWeightU( ptProjection )*dHalfLineLength;
                            const double dHalfLineLengthV = m_dEpipolarLineBaseLength + m_pCameraLEFT->getPrincipalWeightV( ptProjection )*dHalfLineLength;

                            assert( 0.0 < dHalfLineLengthU );
                            assert( 0.0 < dHalfLineLengthV );

                            //ds raw values
                            const double dUMinimumRAW     = std::max( ptProjection.x-dHalfLineLengthU, 0.0 );
                            const double dUMaximumRAW     = std::min( ptProjection.x+dHalfLineLengthU, m_pCameraLEFT->m_dWidthPixels );
                            const double dVForUMinimumRAW = _getCurveV( vecCoefficients, dUMinimumRAW );
                            const double dVForUMaximumRAW = _getCurveV( vecCoefficients, dUMaximumRAW );

                            assert( 0.0 <= dUMinimumRAW );
                            assert( m_pCameraLEFT->m_dWidthPixels >= dUMinimumRAW );
                            assert( 0.0 <= dUMaximumRAW );
                            assert( m_pCameraLEFT->m_dWidthPixels >= dUMaximumRAW );
                            //assert( dUMinimumRAW     < dUMaximumRAW );
                            //assert( dVForUMinimumRAW != dVForUMaximumRAW );

                            //ds check if line is out of scope
                            if( ( 0.0 > dVForUMinimumRAW && 0.0 > dVForUMaximumRAW )                                                       ||
                                ( m_pCameraLEFT->m_dHeightPixels < dVForUMinimumRAW && m_pCameraLEFT->m_dHeightPixels < dVForUMaximumRAW ) )
                            {
                                //ds landmark out of sight (not visible in this frame, still active though)
                                throw CExceptionEpipolarLine( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) vertical out of sight" );
                            }

                            //ds final values set after if tree TODO reduce if cases for performance
                            double dUMinimum     = -1.0;
                            double dUMaximum     = -1.0;
                            double dVForUMinimum = -1.0;
                            double dVForUMaximum = -1.0;

                            //ds compute v border values
                            const double dVLimitMinimum( std::max( ptProjection.y-dHalfLineLengthV, 0.0 ) );
                            const double dVLimitMaximum( std::min( ptProjection.y+dHalfLineLengthV, m_pCameraLEFT->m_dHeightPixels ) );

                            assert( 0.0 <= dVLimitMinimum && m_pCameraLEFT->m_dHeightPixels >= dVLimitMinimum );
                            assert( 0.0 <= dVLimitMaximum && m_pCameraLEFT->m_dHeightPixels >= dVLimitMaximum );

                            //ds regular U
                            dUMinimum = dUMinimumRAW;
                            dUMaximum = dUMaximumRAW;

                            //ds check line configuration
                            if( dVForUMinimumRAW < dVForUMaximumRAW )
                            {
                                //ds check for invalid border values (bad reprojections)
                                if( dVLimitMinimum > dVForUMaximumRAW || dVLimitMaximum < dVForUMinimumRAW )
                                {
                                    throw CExceptionEpipolarLine( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) caught bad projection negative slope" );
                                }

                                //ds regular case
                                if( dVLimitMinimum > dVForUMinimumRAW )
                                {
                                    dVForUMinimum = dVLimitMinimum;
                                    dUMinimum     = _getCurveU( vecCoefficients, dVForUMinimum );
                                }
                                else
                                {
                                    dVForUMinimum = dVForUMinimumRAW;
                                }
                                if( dVLimitMaximum < dVForUMaximumRAW )
                                {
                                    dVForUMaximum = dVLimitMaximum;
                                    dUMaximum     = _getCurveU( vecCoefficients, dVForUMaximum );
                                }
                                else
                                {
                                    dVForUMaximum = dVForUMaximumRAW;
                                }
                                //std::printf( "sampling case 0: [%6.2f,%6.2f][%6.2f,%6.2f]\n", dUMinimum, dUMaximum, dVForUMinimum, dVForUMaximum );
                            }
                            else
                            {
                                //ds check for invalid border values (bad reprojections)
                                if( dVLimitMinimum > dVForUMinimumRAW || dVLimitMaximum < dVForUMaximumRAW )
                                {
                                    throw CExceptionEpipolarLine( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) caught bad projection positive slope" );
                                }

                                //ds swapped case
                                if( dVLimitMinimum > dVForUMaximumRAW )
                                {
                                    dVForUMinimum = dVLimitMinimum;
                                    dUMaximum     = _getCurveU( vecCoefficients, dVForUMinimum );
                                }
                                else
                                {
                                    dVForUMinimum = dVForUMaximumRAW;
                                }
                                if( dVLimitMaximum < dVForUMinimumRAW )
                                {
                                    dVForUMaximum = dVLimitMaximum;
                                    dUMinimum     = _getCurveU( vecCoefficients, dVForUMaximum );
                                }
                                else
                                {
                                    dVForUMaximum = dVForUMinimumRAW;
                                }
                                //std::printf( "sampling case 1: [%6.2f,%6.2f][%6.2f,%6.2f]\n", dUMinimum, dUMaximum, dVForUMinimum, dVForUMaximum );
                            }

                            assert( 0.0 <= dUMaximum && m_pCameraLEFT->m_dWidthPixels >= dUMaximum );
                            assert( 0.0 <= dUMinimum && m_pCameraLEFT->m_dWidthPixels >= dUMinimum );
                            assert( 0.0 <= dVForUMaximum && m_pCameraLEFT->m_dHeightPixels >= dVForUMaximum );
                            assert( 0.0 <= dVForUMinimum && m_pCameraLEFT->m_dHeightPixels >= dVForUMinimum );
                            assert( 0.0 <= dUMaximum-dUMinimum );
                            assert( 0.0 <= dVForUMaximum-dVForUMinimum );

                            //ds compute pixel ranges to sample
                            const uint32_t uDeltaU = dUMaximum-dUMinimum;
                            const uint32_t uDeltaV = dVForUMaximum-dVForUMinimum;

                            //ds check zero line length (can occur)
                            if( 0 == uDeltaU && 0 == uDeltaV )
                            {
                                throw CExceptionEpipolarLine( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) zero line length" );
                            }

                            //ds the match to find
                            std::shared_ptr< CMatchTracking > pMatchLEFT = 0;

                            //ds sample the larger range
                            if( uDeltaV < uDeltaU )
                            {
                                //ds get the match over U
                                pMatchLEFT = _getMatchSampleRecursiveU( p_matDisplayLEFT, p_matImageLEFT, dUMinimum, uDeltaU, vecCoefficients, pLandmark->getLastDescriptorLEFT( ), pLandmark->matDescriptorReferenceLEFT, pLandmark->dKeyPointSize, 0 );
                            }
                            else
                            {
                                //ds get the match over V
                                pMatchLEFT = _getMatchSampleRecursiveV( p_matDisplayLEFT, p_matImageLEFT, dVForUMinimum, uDeltaV, vecCoefficients, pLandmark->getLastDescriptorLEFT( ), pLandmark->matDescriptorReferenceLEFT, pLandmark->dKeyPointSize, 0 );
                            }

                            assert( 0 != pMatchLEFT );

                            //ds draw epipolar marking circle
                            cv::circle( p_matDisplayLEFT, pMatchLEFT->cKeyPoint.pt, 4, CColorCodeBGR( 255, 0, 0 ), 1 );

                            //ds add this measurement to the landmark
                            _addMeasurementToLandmarkLEFT( p_uFrame,
                                                           pLandmark,
                                                           p_matImageRIGHT,
                                                           pMatchLEFT->cKeyPoint,
                                                           pMatchLEFT->matDescriptor,
                                                           p_matTransformationLEFTtoWORLD,
                                                           p_matTransformationWORLDtoLEFT,
                                                           matProjectionWORLDtoLEFT,
                                                           matProjectionWORLDtoRIGHT,
                                                           p_dMotionScaling );
                            m_vecMeasurementsVisible.push_back( pLandmark->getLastMeasurement( ) );

                            //ds update info
                            ++m_uNumberOfTracksStage3;
                        }
                        catch( const CExceptionEpipolarLine& p_cException )
                        {
                            //std::printf( "[%06lu]<CFundamentalMatcher>(getVisibleLandmarksFundamental) landmark [%06lu] epipolar failure: %s\n", p_uFrame, pLandmark->uID, p_cException.what( ) );
                            ++pLandmark->uFailedSubsequentTrackings;
                            pLandmark->bIsCurrentlyVisible = false;
                        }
                        catch( const CExceptionNoMatchFound& p_cException )
                        {
                            //std::printf( "[%06lu]<CFundamentalMatcher>(getVisibleLandmarksFundamental) landmark [%06lu] matching failure: %s\n", p_uFrame, pLandmark->uID, p_cException.what( ) );
                            ++pLandmark->uFailedSubsequentTrackings;
                            pLandmark->bIsCurrentlyVisible = false;
                        }
                    }
                    else
                    {
                        //ds compute current reprojection points
                        const CPoint3DCAMERA vecPointXYZLEFT( p_matTransformationWORLDtoLEFT*pLandmark->vecPointXYZOptimized );
                        const cv::Point2f ptUVEstimateLEFT( m_pCameraLEFT->getProjectionRounded( vecPointXYZLEFT ) );
                        const cv::Point2f ptUVEstimateRIGHT( m_pCameraRIGHT->getProjectionRounded( vecPointXYZLEFT ) );
                        assert( ptUVEstimateLEFT.y == ptUVEstimateRIGHT.y );

                        //ds regional extraction
                        const float fKeyPointSizePixels       = pLandmark->dKeyPointSize;
                        const float fKeyPointSizePixelsHalf   = 4*fKeyPointSizePixels;
                        const cv::Point2f ptOffsetKeyPointHalf( fKeyPointSizePixelsHalf, fKeyPointSizePixelsHalf );
                        const float fSearchRange = fTriangulationScale*pLandmark->getLastDisparity( );

                        //ds check if we are in tracking range
                        if( m_pCameraLEFT->m_cFieldOfView.contains( ptUVEstimateLEFT ) && m_pCameraRIGHT->m_cFieldOfView.contains( ptUVEstimateRIGHT ) )
                        {
                            //ds STAGE 2: check region on LEFT
                            try
                            {
                                //ds compute search ranges (no subpixel accuracy)
                                const double dScaleSearchULEFT      = std::round( m_pCameraLEFT->getPrincipalWeightU( ptUVEstimateLEFT ) + p_dMotionScaling );
                                const double dScaleSearchVLEFT      = std::round( m_pCameraLEFT->getPrincipalWeightV( ptUVEstimateLEFT ) + p_dMotionScaling );
                                const double dSearchHalfWidthLEFT   = std::round( dScaleSearchULEFT*m_uSearchBlockSizePoseOptimization );
                                const double dSearchHalfHeightLEFT  = std::round( dScaleSearchVLEFT*m_uSearchBlockSizePoseOptimization );

                                //ds corners
                                cv::Point2f ptUpperLeftLEFT( std::max( ptUVEstimateLEFT.x-dSearchHalfWidthLEFT, 0.0 ), std::max( ptUVEstimateLEFT.y-dSearchHalfHeightLEFT, 0.0 ) );
                                cv::Point2f ptLowerRightLEFT( std::min( ptUVEstimateLEFT.x+dSearchHalfWidthLEFT, m_pCameraLEFT->m_dWidthPixels ), std::min( ptUVEstimateLEFT.y+dSearchHalfHeightLEFT, m_pCameraLEFT->m_dHeightPixels ) );

                                //ds search rectangle
                                const cv::Rect cSearchROILEFT( ptUpperLeftLEFT, ptLowerRightLEFT );

                                //ds run detection on current frame
                                std::vector< cv::KeyPoint > vecKeyPointsLEFT;
                                m_pDetector->detect( p_matImageLEFT( cSearchROILEFT ), vecKeyPointsLEFT );

                                //ds if we found some features in the LEFT frame
                                if( 0 < vecKeyPointsLEFT.size( ) )
                                {
                                    //ds compute descriptors in extended ROI
                                    const float fUpperLeftU  = std::max( ptUpperLeftLEFT.x-fKeyPointSizePixelsHalf, 0.0f );
                                    const float fUpperLeftV  = std::max( ptUpperLeftLEFT.y-fKeyPointSizePixelsHalf, 0.0f );
                                    const float fLowerRightU = std::min( ptLowerRightLEFT.x+fKeyPointSizePixelsHalf, m_pCameraLEFT->m_fWidthPixels );
                                    const float fLowerRightV = std::min( ptLowerRightLEFT.y+fKeyPointSizePixelsHalf, m_pCameraLEFT->m_fHeightPixels );
                                    cv::Mat matDescriptorsLEFT;

                                    //ds adjust keypoint offsets before computing descriptors
                                    std::for_each( vecKeyPointsLEFT.begin( ), vecKeyPointsLEFT.end( ), [ &ptOffsetKeyPointHalf ]( cv::KeyPoint& cKeyPoint ){ cKeyPoint.pt += ptOffsetKeyPointHalf; } );
                                    m_pExtractor->compute( p_matImageLEFT( cv::Rect( cv::Point2f( fUpperLeftU, fUpperLeftV ), cv::Point2f( fLowerRightU, fLowerRightV ) ) ), vecKeyPointsLEFT, matDescriptorsLEFT );

                                    //ds check descriptor matches for this landmark
                                    std::vector< cv::DMatch > vecMatchesLEFT;
                                    m_pMatcher->match( pLandmark->getLastDescriptorLEFT( ), matDescriptorsLEFT, vecMatchesLEFT );

                                    //ds if we got a match and the matching distance is within the range
                                    if( 0 < vecMatchesLEFT.size( ) )
                                    {
                                        if( m_dMatchingDistanceCutoffTrackingStage2 > vecMatchesLEFT[0].distance )
                                        {
                                            const cv::Point2f ptBestMatchLEFT( vecKeyPointsLEFT[vecMatchesLEFT[0].trainIdx].pt );
                                            const cv::Point2f ptBestMatchLEFTInCamera( ptUpperLeftLEFT+ptBestMatchLEFT-ptOffsetKeyPointHalf );
                                            const CDescriptor matDescriptorLEFT( matDescriptorsLEFT.row(vecMatchesLEFT[0].trainIdx) );

                                            //ds center V
                                            const float fVReference = ptBestMatchLEFTInCamera.y-fKeyPointSizePixelsHalf;

                                            //ds if in range triangulate the point directly
                                            if( 0.0 <= fVReference )
                                            {
                                                const CMatchTriangulation cMatchRIGHT( m_pTriangulator->getPointTriangulatedInRIGHT( p_matImageRIGHT,
                                                                                                                                     std::max( 0.0f, ptBestMatchLEFTInCamera.x-fSearchRange-fKeyPointSizePixelsHalf ),
                                                                                                                                     fVReference,
                                                                                                                                     fKeyPointSizePixels,
                                                                                                                                     ptBestMatchLEFTInCamera,
                                                                                                                                     matDescriptorLEFT ) );
                                                const CDescriptor matDescriptorRIGHT( cMatchRIGHT.matDescriptorCAMERA );

                                                //ds check depth
                                                const double dDepthMeters = cMatchRIGHT.vecPointXYZCAMERA.z( );
                                                if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                                                {
                                                    throw CExceptionNoMatchFound( "invalid depth" );
                                                }

                                                //ds check if the descriptor match is acceptable
                                                if( m_dMatchingDistanceCutoffTrackingStage2 > cv::norm( pLandmark->getLastDescriptorRIGHT( ), matDescriptorRIGHT, cv::NORM_HAMMING ) )
                                                {
                                                    //ds add measurement to landmark
                                                    _addMeasurementToLandmarkSTEREO( p_uFrame,
                                                                                     pLandmark,
                                                                                     ptBestMatchLEFTInCamera,
                                                                                     cMatchRIGHT.ptUVCAMERA,
                                                                                     cMatchRIGHT.vecPointXYZCAMERA,
                                                                                     matDescriptorLEFT,
                                                                                     matDescriptorRIGHT,
                                                                                     p_matTransformationLEFTtoWORLD,
                                                                                     p_matTransformationWORLDtoLEFT,
                                                                                     matProjectionWORLDtoLEFT,
                                                                                     matProjectionWORLDtoRIGHT );

                                                    //ds add measurement to matcher
                                                    m_vecMeasurementsVisible.push_back( pLandmark->getLastMeasurement( ) );

                                                    //ds info
                                                    cv::circle( p_matDisplayLEFT, ptBestMatchLEFTInCamera, 4, CColorCodeBGR( 255, 0, 0 ), 1 );
                                                    ++m_uNumberOfTracksStage2_2;
                                                }
                                                else
                                                {
                                                    //ds try the RIGHT frame
                                                    throw CExceptionNoMatchFound( "triangulation descriptor mismatch" );
                                                }
                                            }
                                            else
                                            {
                                                //ds try the RIGHT frame
                                                throw CExceptionNoMatchFound( "out of tracking range" );
                                            }
                                        }
                                        else
                                        {
                                            //ds try the RIGHT frame
                                            throw CExceptionNoMatchFound( "descriptor mismatch" );
                                        }
                                    }
                                    else
                                    {
                                        //ds try the RIGHT frame
                                        throw CExceptionNoMatchFound( "no matches found" );
                                    }
                                }
                                else
                                {
                                    //ds try the RIGHT frame
                                    throw CExceptionNoMatchFound( "no features detected" );
                                }
                            }
                            catch( const CExceptionNoMatchFound& p_cExceptionStage2LEFT )
                            {
                                //ds try to find the landmarks on RIGHT
                                try
                                {
                                    //ds compute search ranges (no subpixel accuracy)
                                    const double dScaleSearchURIGHT      = std::round( m_pCameraRIGHT->getPrincipalWeightU( ptUVEstimateRIGHT ) + p_dMotionScaling );
                                    const double dScaleSearchVRIGHT      = std::round( m_pCameraRIGHT->getPrincipalWeightV( ptUVEstimateRIGHT ) + p_dMotionScaling );
                                    const double dSearchHalfWidthRIGHT   = std::round( dScaleSearchURIGHT*m_uSearchBlockSizePoseOptimization );
                                    const double dSearchHalfHeightRIGHT  = std::round( dScaleSearchVRIGHT*m_uSearchBlockSizePoseOptimization );

                                    //ds corners
                                    cv::Point2f ptUpperLeftRIGHT( std::max( ptUVEstimateRIGHT.x-dSearchHalfWidthRIGHT, 0.0 ), std::max( ptUVEstimateRIGHT.y-dSearchHalfHeightRIGHT, 0.0 ) );
                                    cv::Point2f ptLowerRightRIGHT( std::min( ptUVEstimateRIGHT.x+dSearchHalfWidthRIGHT, m_pCameraRIGHT->m_dWidthPixels ), std::min( ptUVEstimateRIGHT.y+dSearchHalfHeightRIGHT, m_pCameraRIGHT->m_dHeightPixels ) );

                                    //ds search rectangle
                                    const cv::Rect cSearchROIRIGHT( ptUpperLeftRIGHT, ptLowerRightRIGHT );

                                    //ds run detection on current frame
                                    std::vector< cv::KeyPoint > vecKeyPointsRIGHT;
                                    m_pDetector->detect( p_matImageRIGHT( cSearchROIRIGHT ), vecKeyPointsRIGHT );

                                    //ds if we found some features in the RIGHT frame
                                    if( 0 < vecKeyPointsRIGHT.size( ) )
                                    {
                                        //ds compute descriptors in extended ROI
                                        const float fUpperLeftU  = std::max( ptUpperLeftRIGHT.x-fKeyPointSizePixelsHalf, 0.0f );
                                        const float fUpperLeftV  = std::max( ptUpperLeftRIGHT.y-fKeyPointSizePixelsHalf, 0.0f );
                                        const float fLowerRightU = std::min( ptLowerRightRIGHT.x+fKeyPointSizePixelsHalf, m_pCameraRIGHT->m_fWidthPixels );
                                        const float fLowerRightV = std::min( ptLowerRightRIGHT.y+fKeyPointSizePixelsHalf, m_pCameraRIGHT->m_fHeightPixels );
                                        cv::Mat matDescriptorsRIGHT;

                                        //ds adjust keypoint offsets before computing descriptors
                                        std::for_each( vecKeyPointsRIGHT.begin( ), vecKeyPointsRIGHT.end( ), [ &ptOffsetKeyPointHalf ]( cv::KeyPoint& cKeyPoint ){ cKeyPoint.pt += ptOffsetKeyPointHalf; } );
                                        m_pExtractor->compute( p_matImageRIGHT( cv::Rect( cv::Point2f( fUpperLeftU, fUpperLeftV ), cv::Point2f( fLowerRightU, fLowerRightV ) ) ), vecKeyPointsRIGHT, matDescriptorsRIGHT );

                                        //ds check descriptor matches for this landmark
                                        std::vector< cv::DMatch > vecMatchesRIGHT;
                                        m_pMatcher->match( pLandmark->getLastDescriptorRIGHT( ), matDescriptorsRIGHT, vecMatchesRIGHT );

                                        //ds if we got a match and the matching distance is within the range
                                        if( 0 < vecMatchesRIGHT.size( ) )
                                        {
                                            if( m_dMatchingDistanceCutoffTrackingStage2 > vecMatchesRIGHT[0].distance )
                                            {
                                                const cv::Point2f ptBestMatchRIGHT( vecKeyPointsRIGHT[vecMatchesRIGHT[0].trainIdx].pt );
                                                const cv::Point2f ptBestMatchRIGHTInCamera( ptUpperLeftRIGHT+ptBestMatchRIGHT-ptOffsetKeyPointHalf );
                                                const CDescriptor matDescriptorRIGHT( matDescriptorsRIGHT.row(vecMatchesRIGHT[0].trainIdx) );

                                                //ds center V
                                                const float fVReference = ptBestMatchRIGHTInCamera.y-fKeyPointSizePixelsHalf;

                                                //ds if in range triangulate the point directly
                                                if( 0.0 <= fVReference )
                                                {
                                                    const CMatchTriangulation cMatchLEFT( m_pTriangulator->getPointTriangulatedInLEFT( p_matImageLEFT,
                                                                                                                                       fSearchRange,
                                                                                                                                       std::max( 0.0f, ptBestMatchRIGHTInCamera.x-fKeyPointSizePixelsHalf ),
                                                                                                                                       fVReference,
                                                                                                                                       fKeyPointSizePixels,
                                                                                                                                       ptBestMatchRIGHTInCamera,
                                                                                                                                       matDescriptorRIGHT ) );
                                                    const CDescriptor matDescriptorLEFT( cMatchLEFT.matDescriptorCAMERA );

                                                    //ds check depth
                                                    const double dDepthMeters = cMatchLEFT.vecPointXYZCAMERA.z( );
                                                    if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                                                    {
                                                        throw CExceptionNoMatchFound( "invalid depth" );
                                                    }

                                                    //ds check if the descriptor match is acceptable
                                                    if( m_dMatchingDistanceCutoffTrackingStage2 > cv::norm( pLandmark->getLastDescriptorLEFT( ), matDescriptorLEFT, cv::NORM_HAMMING ) )
                                                    {
                                                        //ds add measurement to landmark
                                                        _addMeasurementToLandmarkSTEREO( p_uFrame,
                                                                                         pLandmark,
                                                                                         cMatchLEFT.ptUVCAMERA,
                                                                                         ptBestMatchRIGHTInCamera,
                                                                                         cMatchLEFT.vecPointXYZCAMERA,
                                                                                         matDescriptorLEFT,
                                                                                         matDescriptorRIGHT,
                                                                                         p_matTransformationLEFTtoWORLD,
                                                                                         p_matTransformationWORLDtoLEFT,
                                                                                         matProjectionWORLDtoLEFT,
                                                                                         matProjectionWORLDtoRIGHT );

                                                        //ds add measurement to matcher
                                                        m_vecMeasurementsVisible.push_back( pLandmark->getLastMeasurement( ) );

                                                        //ds info
                                                        cv::circle( p_matDisplayRIGHT, ptBestMatchRIGHTInCamera, 4, CColorCodeBGR( 255, 0, 0 ), 1 );
                                                        ++m_uNumberOfTracksStage2_2;
                                                    }
                                                    else
                                                    {
                                                        throw CExceptionNoMatchFound( "triangulation descriptor mismatch" );
                                                    }
                                                }
                                                else
                                                {
                                                    throw CExceptionNoMatchFound( "out of tracking range" );
                                                }
                                            }
                                            else
                                            {
                                                throw CExceptionNoMatchFound( "descriptor mismatch" );
                                            }
                                        }
                                        else
                                        {
                                            throw CExceptionNoMatchFound( "no matches found" );
                                        }
                                    }
                                    else
                                    {
                                        throw CExceptionNoMatchFound( "no features detected" );
                                    }
                                }
                                catch( const CExceptionNoMatchFound& p_cExceptionStage2RIGHT )
                                {
                                    //std::printf( "<CFundamentalMatcher>(getMeasurementsEpipolar) landmark [%06lu] RIGHT failed: %s\n", pLandmark->uID, p_cException.what( ) );
                                    ++pLandmark->uFailedSubsequentTrackings;
                                    pLandmark->bIsCurrentlyVisible = false;
                                }
                            }
                        }
                        else
                        {
                            ++pLandmark->uFailedSubsequentTrackings;
                            pLandmark->bIsCurrentlyVisible = false;
                        }
                    }

                    //ds check activity
                    if( m_uMaximumFailedSubsequentTrackingsPerLandmark > pLandmark->uFailedSubsequentTrackings )
                    {
                        vecActiveLandmarksPerDetectionPoint->push_back( pLandmark );
                    }
                }
            }
        }

        //ds log
        //CLogger::CLogDetectionEpipolar::addEntry( p_uFrame, cDetectionPoint.uID, cDetectionPoint.vecLandmarks->size( ), vecActiveLandmarksPerDetectionPoint->size( ), vecMeasurementsPerDetectionPoint.size( ) );

        //ds check if we can keep the measurement point
        if( 0 < vecActiveLandmarksPerDetectionPoint->size( ) )
        {
            //ds register the measurement point and its active landmarks anew
            vecDetectionPointsActive.push_back( CDetectionPoint( cDetectionPoint.uID, cDetectionPoint.matTransformationLEFTtoWORLD, vecActiveLandmarksPerDetectionPoint ) );
        }
        else
        {
            //std::printf( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) erased detection point [%06lu]\n", cDetectionPoint.uID );
        }
    }

    //ds timing
    m_dDurationTotalSecondsEpipolarTracking += CTimer::getTimeSeconds( )-dTimeStartSeconds;

    //ds info counters
    m_uNumberOfFailedLandmarkOptimizationsTotal += uNumberOfFailedLandmarkOptimizations;
    m_uNumberOfInvalidLandmarksTotal            += uNumberOfInvalidLandmarks;

    /*if( 50 < uNumberOfFailedLandmarkOptimizations+uNumberOfInvalidLandmarks )
    {
        std::printf( "[0][%06lu]<CFundamentalMatcher>(getMeasurementsEpipolar) erased landmarks - failed optimization: %2lu, invalid optimization: %2lu\n", p_uFrame, uNumberOfFailedLandmarkOptimizations, uNumberOfInvalidLandmarks );
    }*/

    //ds update active measurement points
    m_vecDetectionPointsActive.swap( vecDetectionPointsActive );
}

const cv::Mat CFundamentalMatcher::getMaskVisibleLandmarks( ) const
{
    //ds compute mask to avoid detecting similar features
    cv::Mat matMaskDetection( cv::Mat( m_pCameraSTEREO->m_uPixelHeight, m_pCameraSTEREO->m_uPixelWidth, CV_8UC1, cv::Scalar ( 255 ) ) );

    //ds draw black circles for existing landmark positions into the mask (avoid redetection of landmarks)
    for( const CLandmark* pLandmark: m_vecVisibleLandmarks )
    {
        cv::circle( matMaskDetection, pLandmark->getLastDetectionLEFT( ), m_uFeatureRadiusForMask, cv::Scalar ( 0 ), -1 );
    }

    return matMaskDetection;
}

const cv::Mat CFundamentalMatcher::getMaskActiveLandmarks( const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT, cv::Mat& p_matDisplayLEFT ) const
{
    //ds compute mask to avoid detecting similar features
    cv::Mat matMaskDetection( cv::Mat( m_pCameraSTEREO->m_uPixelHeight, m_pCameraSTEREO->m_uPixelWidth, CV_8UC1, cv::Scalar ( 255 ) ) );

    //ds draw black circles for existing landmark positions into the mask (avoid redetection of landmarks)
    //ds active measurements
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds if the landmark is visible we don't have to compute the reprojection
            if( pLandmark->bIsCurrentlyVisible )
            {
                cv::circle( matMaskDetection, pLandmark->getLastDetectionLEFT( ), m_uFeatureRadiusForMask, cv::Scalar ( 0 ), -1 );
                cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), m_uFeatureRadiusForMask, cv::Scalar ( 0 ), -1 );
            }
            else
            {
                //ds get into camera frame
                const CPoint3DCAMERA vecPointXYZ( p_matTransformationWORLDtoLEFT*pLandmark->vecPointXYZOptimized );

                cv::circle( matMaskDetection, m_pCameraLEFT->getUV( vecPointXYZ ), m_uFeatureRadiusForMask, cv::Scalar ( 0 ), -1 );
                cv::circle( p_matDisplayLEFT, m_pCameraLEFT->getUV( vecPointXYZ ), m_uFeatureRadiusForMask, cv::Scalar ( 0 ), -1 );
            }
        }
    }

    return matMaskDetection;
}

void CFundamentalMatcher::drawVisibleLandmarks( cv::Mat& p_matDisplayLEFT, cv::Mat& p_matDisplayRIGHT, const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT ) const
{
    //ds for all visible landmarks
    for( const CLandmark* pLandmark: m_vecVisibleLandmarks )
    {
        //ds compute green brightness based on depth (further away -> darker)
        const uint8_t uGreenValue = 255-pLandmark->getLastDepth( )/100.0*255;
        cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), 2, CColorCodeBGR( 0, uGreenValue, 0 ), -1 );
        cv::circle( p_matDisplayRIGHT, pLandmark->getLastDetectionRIGHT( ), 2, CColorCodeBGR( 0, uGreenValue, 0 ), -1 );

        //ds get 3d position in current camera frame (trivial)
        const CPoint3DCAMERA vecXYZLEFT( p_matTransformationWORLDtoLEFT*pLandmark->getLastPointXYZOptimized( ) );

        //ds also draw reprojections
        cv::circle( p_matDisplayLEFT, m_pCameraLEFT->getProjection( vecXYZLEFT ), 6, CColorCodeBGR( 0, uGreenValue, 0 ), 1 );
        cv::circle( p_matDisplayRIGHT, m_pCameraRIGHT->getProjection( vecXYZLEFT ), 6, CColorCodeBGR( 0, uGreenValue, 0 ), 1 );
    }
}

//ds shifts all active landmarks
void CFundamentalMatcher::shiftActiveLandmarks( const Eigen::Vector3d& p_vecTranslation )
{
    //ds active measurements
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds shift the landmark
            pLandmark->vecPointXYZOptimized += p_vecTranslation;
        }
    }
}

//ds rotates all active landmarks
void CFundamentalMatcher::rotateActiveLandmarks( const Eigen::Matrix3d& p_matRotation )
{
    //ds active measurements
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds shift the landmark
            pLandmark->vecPointXYZOptimized = p_matRotation*pLandmark->vecPointXYZOptimized;
        }
    }
}

void CFundamentalMatcher::clearActiveLandmarksMeasurements( const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT )
{
    //ds precompute intrinsics
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*p_matTransformationWORLDtoLEFT.matrix( ) );
    const MatrixProjection matProjectionWORLDtoRIGHT( m_pCameraRIGHT->m_matProjection*p_matTransformationWORLDtoLEFT.matrix( ) );

    //ds active measurements
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds clear landmark measurements
            pLandmark->clearMeasurements( pLandmark->vecPointXYZOptimized, p_matTransformationWORLDtoLEFT, matProjectionWORLDtoLEFT, matProjectionWORLDtoLEFT );
        }
    }
}

const std::shared_ptr< CMatchTracking > CFundamentalMatcher::_getMatchSampleRecursiveU( cv::Mat& p_matDisplay,
                                                                                                const cv::Mat& p_matImage,
                                                                                                const double& p_dUMinimum,
                                                                                                const uint32_t& p_uDeltaU,
                                                                                                const Eigen::Vector3d& p_vecCoefficients,
                                                                                                const CDescriptor& p_matReferenceDescriptor,
                                                                                                const CDescriptor& p_matOriginalDescriptor,
                                                                                                const float& p_fKeyPointSize,
                                                                                                const uint8_t& p_uRecursionDepth ) const
{
    //ds allocate keypoint pool
    std::vector< cv::KeyPoint > vecPoolKeyPoints( p_uDeltaU );

    //ds determine sampling direction - if even we loop positively
    if( 0 == p_uRecursionDepth%2 )
    {
        const int8_t iSamplingOffset( p_uRecursionDepth );

        //ds sample over U
        for( uint32_t u = 0; u < p_uDeltaU; ++u )
        {
            //ds compute corresponding V coordinate
            const double dU( p_dUMinimum+u );
            const double dV( _getCurveV( p_vecCoefficients, dU ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[u] = cv::KeyPoint( dU, dV, p_fKeyPointSize );
            //cv::circle( p_matDisplay, cv::Point2i( dU, dV ), 1, CColorCodeBGR( 0, 165, 255 ), -1 );
        }
    }
    else
    {
        const int8_t iSamplingOffset( -p_uRecursionDepth );

        //ds sample over U
        for( uint32_t u = 0; u < p_uDeltaU; ++u )
        {
            //ds compute corresponding V coordinate
            const double dU( p_dUMinimum+u );
            const double dV( _getCurveV( p_vecCoefficients, dU ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[u] = cv::KeyPoint( dU, dV, p_fKeyPointSize );
            //cv::circle( p_matDisplay, cv::Point2i( dU, dV ), 1, CColorCodeBGR( 0, 165, 255 ), -1 );
        }
    }

    //ds center point
    const cv::Point2f ptCenter( vecPoolKeyPoints[vecPoolKeyPoints.size( )/2].pt );

    //ds absolute deltas - augmented by key point length to create a larger search region
    const float fDeltaU = std::fabs( vecPoolKeyPoints.front( ).pt.x-vecPoolKeyPoints.back( ).pt.x ) + 16*p_fKeyPointSize;
    const float fDeltaV = std::fabs( vecPoolKeyPoints.front( ).pt.y-vecPoolKeyPoints.back( ).pt.y ) + 16*p_fKeyPointSize;

    //ds compute topleft point coordinates: mid - distances
    const float fUTopLeft = std::max( ptCenter.x-fDeltaU/2, 0.0f );
    const float fVTopLeft = std::max( ptCenter.y-fDeltaV/2, 0.0f );

    //ds compute possible lengths
    const float fWidth  = std::min( fDeltaU, m_pCameraLEFT->m_fWidthPixels-fUTopLeft );
    const float fHeigth = std::min( fDeltaV, m_pCameraLEFT->m_fHeightPixels-fVTopLeft );

    //ds search rectangle
    const cv::Rect cMatchROI( fUTopLeft, fVTopLeft, fWidth, fHeigth );
    //cv::rectangle( p_matDisplay, cMatchROI, CColorCodeBGR( 0, 165, 255 ) );

    //ds adjust keypoint offsets before computing descriptors
    std::for_each( vecPoolKeyPoints.begin( ), vecPoolKeyPoints.end( ), [ &fUTopLeft, &fVTopLeft ]( cv::KeyPoint& cKeyPoint ){ cKeyPoint.pt.x -= fUTopLeft; cKeyPoint.pt.y -= fVTopLeft; } );

    try
    {
        //ds return if we find a match on the epipolar line
        return _getMatch( p_matImage( cMatchROI ), vecPoolKeyPoints, cv::Point2f( fUTopLeft, fVTopLeft ), p_matReferenceDescriptor, p_matOriginalDescriptor );
    }
    catch( const CExceptionNoMatchFoundInternal& p_cException )
    {
        //ds escape if the limit is reached
        if( m_uRecursionLimitEpipolarLines == p_uRecursionDepth )
        {
            throw CExceptionNoMatchFound( p_cException.what( ) );
        }
        else
        {
            //ds sample further
            return _getMatchSampleRecursiveU( p_matDisplay, p_matImage, p_dUMinimum, p_uDeltaU, p_vecCoefficients, p_matReferenceDescriptor, p_matOriginalDescriptor, p_fKeyPointSize, p_uRecursionDepth+m_uRecursionStepSize );
        }
    }
}

const std::shared_ptr< CMatchTracking > CFundamentalMatcher::_getMatchSampleRecursiveV( cv::Mat& p_matDisplay,
                                                                                                const cv::Mat& p_matImage,
                                                                                                const double& p_dVMinimum,
                                                                                                const uint32_t& p_uDeltaV,
                                                                                                const Eigen::Vector3d& p_vecCoefficients,
                                                                                                const CDescriptor& p_matReferenceDescriptor,
                                                                                                const CDescriptor& p_matOriginalDescriptor,
                                                                                                const float& p_fKeyPointSize,
                                                                                                const uint8_t& p_uRecursionDepth ) const
{
    //ds allocate keypoint pool
    std::vector< cv::KeyPoint > vecPoolKeyPoints( p_uDeltaV );

    //ds determine sampling direction - if even we loop positively
    if( 0 == p_uRecursionDepth%2 )
    {
        const int8_t iSamplingOffset( p_uRecursionDepth );

        //ds sample over U
        for( uint32_t v = 0; v < p_uDeltaV; ++v )
        {
            //ds compute corresponding U coordinate
            const double dV( p_dVMinimum+v );
            const double dU( _getCurveU( p_vecCoefficients, dV ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[v] = cv::KeyPoint( dU, dV, p_fKeyPointSize );
            //cv::circle( p_matDisplay, cv::Point2i( dU, dV ), 1, CColorCodeBGR( 219, 112, 147 ), -1 );
        }
    }
    else
    {
        const int8_t iSamplingOffset( -p_uRecursionDepth );

        //ds sample over U
        for( uint32_t v = 0; v < p_uDeltaV; ++v )
        {
            //ds compute corresponding U coordinate
            const double dV( p_dVMinimum+v );
            const double dU( _getCurveU( p_vecCoefficients, dV ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[v] = cv::KeyPoint( dU, dV, p_fKeyPointSize );
            //cv::circle( p_matDisplay, cv::Point2i( dU, dV ), 1, CColorCodeBGR( 219, 112, 147 ), -1 );
        }
    }

    //ds center point
    const cv::Point2f ptCenter( vecPoolKeyPoints[vecPoolKeyPoints.size( )/2].pt );

    //ds absolute deltas - augmented by key point length to create a larger search region
    const float fDeltaU = std::fabs( vecPoolKeyPoints.front( ).pt.x-vecPoolKeyPoints.back( ).pt.x ) + 16*p_fKeyPointSize;
    const float fDeltaV = std::fabs( vecPoolKeyPoints.front( ).pt.y-vecPoolKeyPoints.back( ).pt.y ) + 16*p_fKeyPointSize;

    //ds compute topleft point coordinates: mid - distances
    const float fUTopLeft = std::max( ptCenter.x-fDeltaU/2, 0.0f );
    const float fVTopLeft = std::max( ptCenter.y-fDeltaV/2, 0.0f );

    //ds compute possible lengths
    const float fWidth  = std::min( fDeltaU, m_pCameraLEFT->m_fWidthPixels-fUTopLeft );
    const float fHeigth = std::min( fDeltaV, m_pCameraLEFT->m_fHeightPixels-fVTopLeft );

    //ds search rectangle
    const cv::Rect cMatchROI( fUTopLeft, fVTopLeft, fWidth, fHeigth );
    //cv::rectangle( p_matDisplay, cMatchROI, CColorCodeBGR( 0, 165, 255 ) );

    //ds adjust keypoint offsets before computing descriptors
    std::for_each( vecPoolKeyPoints.begin( ), vecPoolKeyPoints.end( ), [ &fUTopLeft, &fVTopLeft ]( cv::KeyPoint& cKeyPoint ){ cKeyPoint.pt.x -= fUTopLeft; cKeyPoint.pt.y -= fVTopLeft; } );

    try
    {
        //ds return if we find a match on the epipolar line
        return _getMatch( p_matImage( cMatchROI ), vecPoolKeyPoints, cv::Point2f( fUTopLeft, fVTopLeft ), p_matReferenceDescriptor, p_matOriginalDescriptor );
    }
    catch( const CExceptionNoMatchFoundInternal& p_cException )
    {
        //ds escape if the limit is reached
        if( m_uRecursionLimitEpipolarLines == p_uRecursionDepth )
        {
            throw CExceptionNoMatchFound( p_cException.what( ) );
        }
        else
        {
            //ds sample further
            return _getMatchSampleRecursiveV( p_matDisplay, p_matImage, p_dVMinimum, p_uDeltaV, p_vecCoefficients, p_matReferenceDescriptor, p_matOriginalDescriptor, p_fKeyPointSize, p_uRecursionDepth+m_uRecursionStepSize );
        }
    }
}

const std::shared_ptr< CMatchTracking > CFundamentalMatcher::_getMatch( const cv::Mat& p_matImage,
                                                                     std::vector< cv::KeyPoint >& p_vecPoolKeyPoints,
                                                                     const cv::Point2f& p_ptOffsetROI,
                                                                     const CDescriptor& p_matDescriptorReference,
                                                                     const CDescriptor& p_matDescriptorOriginal ) const
{
    //ds descriptor pool
    cv::Mat matPoolDescriptors;

    //ds compute descriptors of current search area
    m_pExtractor->compute( p_matImage, p_vecPoolKeyPoints, matPoolDescriptors );

    //ds escape if we didnt find any descriptors
    if( 0 == p_vecPoolKeyPoints.size( ) )
    {
        throw CExceptionNoMatchFoundInternal( "could not find a matching descriptor (empty key point pool)" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matDescriptorReference, matPoolDescriptors, vecMatches );

    //ds escape for no matches
    if( 0 == vecMatches.size( ) )
    {
        throw CExceptionNoMatchFoundInternal( "could not find any matches (empty matches pool)" );
    }

    //ds buffer first match
    const cv::DMatch& cBestMatch( vecMatches[0] );

    //ds check if we are in the range (works for negative ids as well)
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( cBestMatch.trainIdx ) < p_vecPoolKeyPoints.size( ) );

    //ds bufffer new descriptor
    const CDescriptor matDescriptorNew( matPoolDescriptors.row(cBestMatch.trainIdx) );

    //ds distances
    const double dMatchingDistanceToRelative( cBestMatch.distance );
    const double dMatchingDistanceToOriginal( cv::norm( p_matDescriptorOriginal, matDescriptorNew, cv::NORM_HAMMING ) );

    if( m_dMatchingDistanceCutoffTrackingStage3 > dMatchingDistanceToRelative )
    {
        if( m_dMatchingDistanceCutoffOriginal > dMatchingDistanceToOriginal )
        {
            //ds correct keypoint offset
            cv::KeyPoint cKeyPointShifted( p_vecPoolKeyPoints[cBestMatch.trainIdx] );
            cKeyPointShifted.pt += p_ptOffsetROI;

            //ds return the match
            return std::make_shared< CMatchTracking >( cKeyPointShifted, matDescriptorNew );
        }
        else
        {
            throw CExceptionNoMatchFoundInternal( "could not find a matching descriptor (ORIGINAL matching distance too big)" );
        }
    }
    else
    {
        throw CExceptionNoMatchFoundInternal( "could not find a matching descriptor" );
    }
}

void CFundamentalMatcher::_addMeasurementToLandmarkLEFT( const UIDFrame p_uFrame,
                                                  CLandmark* p_pLandmark,
                                                  const cv::Mat& p_matImageRIGHT,
                                                  const cv::KeyPoint& p_cKeyPoint,
                                                  const CDescriptor& p_matDescriptorLEFT,
                                                  const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                  const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                                  const MatrixProjection& p_matProjectionWORLDtoLEFT,
                                                  const MatrixProjection& p_matProjectionWORLDtoRIGHT,
                                                  const double& p_dMotionScaling )
{
    //ds buffer point
    cv::Point2f ptUVLEFT( p_cKeyPoint.pt );

    //assert( m_pCameraLEFT->m_cFieldOfView.contains( p_cKeyPoint.pt ) );

    //ds triangulate point
    const float fSearchRangePixels = ( 1.0+p_dMotionScaling )*p_pLandmark->getLastDisparity( );
    const CMatchTriangulation cMatchRIGHT( m_pTriangulator->getPointTriangulatedInRIGHT( p_matImageRIGHT,
                                                                                         std::max( 0.0f, ptUVLEFT.x-fSearchRangePixels-4*p_cKeyPoint.size ),
                                                                                         ptUVLEFT.y-4*p_cKeyPoint.size,
                                                                                         p_cKeyPoint.size,
                                                                                         ptUVLEFT,
                                                                                         p_matDescriptorLEFT ) );
    const CPoint3DCAMERA vecPointXYZLEFT( cMatchRIGHT.vecPointXYZCAMERA );
    const cv::Point2f ptUVRIGHT( cMatchRIGHT.ptUVCAMERA );

    //ds depth
    const double dDepthMeters = vecPointXYZLEFT.z( );

    //ds check depth
    if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
    {
        throw CExceptionNoMatchFound( "<CFundamentalMatcher>(_addMeasurementToLandmark) invalid depth" );
    }

    //assert( m_pCameraRIGHT->m_cFieldOfView.contains( ptUVRIGHT ) );

    //ds update landmark (NO EXCEPTIONS HERE)
    p_pLandmark->bIsCurrentlyVisible        = true;
    p_pLandmark->uFailedSubsequentTrackings = 0;
    p_pLandmark->addMeasurement( p_uFrame,
                                 ptUVLEFT,
                                 ptUVRIGHT,
                                 p_matDescriptorLEFT,
                                 cMatchRIGHT.matDescriptorCAMERA,
                                 vecPointXYZLEFT,
                                 p_matTransformationLEFTtoWORLD,
                                 p_matTransformationWORLDtoLEFT,
                                 p_matProjectionWORLDtoLEFT,
                                 p_matProjectionWORLDtoRIGHT );

    //ds add to vector for fast search
    m_vecVisibleLandmarks.push_back( p_pLandmark );
}

void CFundamentalMatcher::_addMeasurementToLandmarkSTEREO( const UIDFrame p_uFrame,
                                                           const CSolverStereoPosit::CMatch& p_cMatchSTEREO,
                                                           const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                           const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                                           const MatrixProjection& p_matProjectionWORLDtoLEFT,
                                                           const MatrixProjection& p_matProjectionWORLDtoRIGHT )
{
    //ds input validation
    //assert( m_pCameraLEFT->m_cFieldOfView.contains( p_cMatchSTEREO.ptUVLEFT ) );
    //assert( m_pCameraRIGHT->m_cFieldOfView.contains( p_cMatchSTEREO.ptUVRIGHT ) );
    assert( m_dMinimumDepthMeters < p_cMatchSTEREO.vecPointXYZLEFT.z( ) );
    assert( m_dMaximumDepthMeters > p_cMatchSTEREO.vecPointXYZLEFT.z( ) );

    //ds update landmark directly (NO EXCEPTIONS HERE)
    assert( 0 != p_cMatchSTEREO.pLandmark );
    p_cMatchSTEREO.pLandmark->bIsCurrentlyVisible        = true;
    p_cMatchSTEREO.pLandmark->uFailedSubsequentTrackings = 0;
    p_cMatchSTEREO.pLandmark->addMeasurement( p_uFrame,
                               p_cMatchSTEREO.ptUVLEFT,
                               p_cMatchSTEREO.ptUVRIGHT,
                               p_cMatchSTEREO.matDescriptorLEFT,
                               p_cMatchSTEREO.matDescriptorRIGHT,
                               p_cMatchSTEREO.vecPointXYZLEFT,
                               p_matTransformationLEFTtoWORLD,
                               p_matTransformationWORLDtoLEFT,
                               p_matProjectionWORLDtoLEFT,
                               p_matProjectionWORLDtoRIGHT );

    //ds add to vector for fast search
    m_vecVisibleLandmarks.push_back( p_cMatchSTEREO.pLandmark );
}

void CFundamentalMatcher::_addMeasurementToLandmarkSTEREO( const UIDFrame p_uFrame,
                                                           CLandmark* p_pLandmark,
                                                           const cv::Point2d& p_ptUVLEFT,
                                                           const cv::Point2d& p_ptUVRIGHT,
                                                           const CPoint3DCAMERA& p_vecPointXYZLEFT,
                                                           const CDescriptor& p_matDescriptorLEFT,
                                                           const CDescriptor& p_matDescriptorRIGHT,
                                                           const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                           const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                                           const MatrixProjection& p_matProjectionWORLDtoLEFT,
                                                           const MatrixProjection& p_matProjectionWORLDtoRIGHT )
{
    //ds input validation
    //assert( m_pCameraLEFT->m_cFieldOfView.contains( p_ptUVLEFT ) );
    //assert( m_pCameraRIGHT->m_cFieldOfView.contains( p_ptUVRIGHT ) );
    assert( m_dMinimumDepthMeters < p_vecPointXYZLEFT.z( ) );
    assert( m_dMaximumDepthMeters > p_vecPointXYZLEFT.z( ) );
    assert( p_ptUVLEFT.y == p_ptUVRIGHT.y );
    assert( p_ptUVLEFT.x > p_ptUVRIGHT.x );

    //ds update landmark (NO EXCEPTIONS HERE)
    p_pLandmark->bIsCurrentlyVisible        = true;
    p_pLandmark->uFailedSubsequentTrackings = 0;
    p_pLandmark->addMeasurement( p_uFrame,
                                 p_ptUVLEFT,
                                 p_ptUVRIGHT,
                                 p_matDescriptorLEFT,
                                 p_matDescriptorRIGHT,
                                 p_vecPointXYZLEFT,
                                 p_matTransformationLEFTtoWORLD,
                                 p_matTransformationWORLDtoLEFT,
                                 p_matProjectionWORLDtoLEFT,
                                 p_matProjectionWORLDtoRIGHT );

    //ds add to vector for fast search
    m_vecVisibleLandmarks.push_back( p_pLandmark );
}

const double CFundamentalMatcher::_getCurveU( const Eigen::Vector3d& p_vecCoefficients, const double& p_dV ) const
{
    return -( p_vecCoefficients(1)*p_dV+p_vecCoefficients(2) )/p_vecCoefficients(0);
}
const double CFundamentalMatcher::_getCurveV( const Eigen::Vector3d& p_vecCoefficients, const double& p_dU ) const
{
    return -( p_vecCoefficients(0)*p_dU+p_vecCoefficients(2) )/p_vecCoefficients(1);
}
